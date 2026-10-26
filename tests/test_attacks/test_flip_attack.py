import ast
import copy
from dataclasses import asdict
import importlib.util
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from adversariallm.attacks import Attack
from adversariallm.attacks.flip_attack import FlipAttack, FlipAttackConfig
from adversariallm.defenses.api_target import APITarget
from adversariallm.defenses.registry import validate_hosted_target
from adversariallm.lm_utils.text_generation import GenerationResult


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "verification/flip_attack/reference/src"
MODES = ["FWO", "FCW", "FCS", "FMM"]
FLAGS = list(product([False, True], repeat=3))
PROMPTS = [
    "Explain how rainbows form, with a simple example.\nUse café windows as context.",
    "  Describe a blue sky\t and white clouds.  \nCompare the colours gently. 🌈",
]
MODELS = ["gpt-4o-2024-08-06", "meta-llama/Meta-Llama-3.1-405B-Instruct", "meta-llama/llama-test"]


def load_reference(name):
    spec = importlib.util.spec_from_file_location("flipattack_reference_" + name, REFERENCE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def author():
    return load_reference("flip_attack").FlipAttack


def config(**overrides):
    settings = asdict(FlipAttackConfig())
    settings.update(overrides)
    return OmegaConf.create(settings)


@pytest.fixture
def hosted(monkeypatch):
    generator = MagicMock()
    generator.generate.return_value = GenerationResult(
        gen=[["A benign answer."]], api_metadata=[[{"id": "request-id", "usage": {"total_tokens": 7}}]],
    )
    monkeypatch.setattr("adversariallm.defenses.api_target.build_text_generator", lambda cfg: generator)
    target = APITarget({"backend": "openrouter", "id": MODELS[0], "provider": "test"})
    return target, generator


def dataset(prompts):
    return [[{"role": "user", "content": prompt}, {"role": "assistant", "content": "Reference answer."}]
            for prompt in prompts]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cot,lang_gpt,few_shot", FLAGS)
@pytest.mark.parametrize("model", MODELS)
def test_every_mode_and_guidance_matches_independent_author(author, hosted, mode, cot, lang_gpt, few_shot, model):
    target, generator = hosted
    target.config["id"] = model
    originals = dataset(PROMPTS)
    before = copy.deepcopy(originals)
    result = FlipAttack(config(flip_mode=mode, cot=cot, lang_gpt=lang_gpt, few_shot=few_shot)).run(target, originals)
    assert len(result.runs) == len(PROMPTS)
    assert generator.generate.call_count == len(PROMPTS)
    for prompt, original, call, run in zip(PROMPTS, before, generator.generate.call_args_list, result.runs):
        # Fresh original builder, not the adapter's imported builder or a generated expected hash.
        _, expected = author(flip_mode=mode, cot=cot, lang_gpt=lang_gpt, few_shot=few_shot,
                             victim_llm=model).generate(prompt)
        assert call.args == ([expected],)
        assert call.kwargs == {"temperature": 0.0, "max_new_tokens": None, "num_return_sequences": 1}
        assert run.original_prompt == original
        assert len(run.steps) == 1
        step = run.steps[0]
        assert step.model_input == expected
        assert step.model_completions == ["A benign answer."]
        assert step.api_metadata == [{"id": "request-id", "usage": {"total_tokens": 7}}]
        assert step.scores == {} and step.loss is None and step.model_input_tokens is None
    assert originals == before


def test_fresh_builders_match_original_runner_loop(author, hosted):
    target, generator = hosted
    replies = ["First answer.", "Second answer."]
    generator.generate.side_effect = [GenerationResult(gen=[[reply]]) for reply in replies]
    # Execute the original main.py loop with its I/O bindings replaced. Its post-loop
    # reporting/evaluation CLI is outside this adapter's scope.
    tree = ast.parse((REFERENCE / "main.py").read_bytes())
    main = next(node for node in tree.body if isinstance(node, ast.If))
    loop = next(node for node in main.body if isinstance(node, ast.For))
    calls = []
    responses = iter(replies)

    def respond(messages):
        calls.append(copy.deepcopy(messages))
        return next(responses)

    namespace = {
        "args": SimpleNamespace(begin=0, end=2, flip_mode="FCS", cot=True, lang_gpt=True,
                                few_shot=True, victim_llm=MODELS[0], eval=False),
        "adv_bench": {"goal": [PROMPTS[0], PROMPTS[0]]}, "FlipAttack": author,
        "victim_llm": SimpleNamespace(response=respond), "tqdm": lambda iterable: iterable,
        "result_dict_list": [],
    }
    exec(compile(ast.Module(body=[loop], type_ignores=[]), "pinned-main-loop", "exec"), namespace)
    result = FlipAttack(config(cot=True, lang_gpt=True, few_shot=True)).run(target, dataset([PROMPTS[0]] * 2))
    assert calls[0] == calls[1]  # Original main.py creates a fresh builder per example.
    assert [call.args[0][0] for call in generator.generate.call_args_list] == calls
    assert [run.steps[0].model_completions[0] for run in result.runs] == [
        row["output"] for row in namespace["result_dict_list"]
    ]


@pytest.mark.parametrize("prompt", ["Hi", ""])
def test_short_few_shot_edge_matches_author_failure(author, hosted, prompt):
    target, generator = hosted
    with pytest.raises((IndexError, ValueError)) as reference_error:
        author(few_shot=True).generate(prompt)
    with pytest.raises(type(reference_error.value)):
        FlipAttack(config(few_shot=True)).run(target, dataset([prompt]))
    generator.generate.assert_not_called()


@pytest.mark.parametrize("settings", [
    {"flip_mode": "unknown"}, {"cot": "false"}, {"lang_gpt": 1}, {"few_shot": None},
    {"generation_config": {"num_return_sequences": 2}},
    {"generation_config": {"max_new_tokens": -1}},
    {"generation_config": {"max_new_tokens": True}},
])
def test_invalid_settings_fail_before_requests(settings, hosted):
    _, generator = hosted
    with pytest.raises(ValueError):
        FlipAttack(config(**settings))
    generator.generate.assert_not_called()


def test_local_target_is_rejected():
    with pytest.raises(ValueError, match="configured OpenRouter target"):
        FlipAttack(config()).run(object(), [])


@pytest.mark.parametrize("completions", [[], [[""]], [["one", "two"]], [[None]]])
def test_invalid_responses_are_not_saved(hosted, completions):
    target, generator = hosted
    generator.generate.return_value = GenerationResult(gen=completions)
    with pytest.raises(ValueError, match="nonempty target response"):
        FlipAttack(config()).run(target, dataset(PROMPTS[:1]))


def test_config_and_registry_match_pinned_cli_defaults():
    assert Attack.from_name("flip_attack") is FlipAttack
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=[
            "hydra/launcher=basic", "models=openrouter", "model=openrouter", "attack=flip_attack",
            "models.openrouter.id=test/model", "models.openrouter.provider=test",
        ])
    # Read literal author argparse defaults independently of our dataclass/YAML.
    tree = ast.parse((REFERENCE / "main.py").read_bytes())
    defaults = {node.args[0].value: {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords
                                   if kw.arg in {"default", "action"}}
                for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"}
    attack = cfg.attacks.flip_attack
    assert attack.flip_mode == defaults["--flip_mode"]["default"]
    assert attack.generation_config.temperature == defaults["--temperature"]["default"]
    assert defaults["--max_token"]["default"] == -1 and attack.generation_config.max_new_tokens is None
    for flag in ("cot", "lang_gpt", "few_shot"):
        assert defaults["--" + flag]["action"] == "store_true" and attack[flag] is False
    assert attack.generation_config.num_return_sequences == 1
    assert attack.seed == 0
    assert cfg.models.openrouter.id == "test/model"
    FlipAttack.validate_config(attack)
    validate_hosted_target(FlipAttack, cfg.models.openrouter, None)


@pytest.mark.parametrize("limit", [None, 500])
@pytest.mark.parametrize("reply", ["A benign answer.", "  A benign answer.\n"])
def test_actual_requests_against_original_gpt_transport(monkeypatch, author, limit, reply):
    import sys

    # The original module imports Anthropic even for GPT. Bind that unused dependency
    # without installing it, then replace only its OpenAI I/O and sleep bindings.
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=MagicMock()))
    original_llm = load_reference("llm")
    reference_client, hosted_client = MagicMock(), MagicMock()
    completion = SimpleNamespace(
        id="request-1", model=MODELS[0], provider="test", usage=None,
        choices=[SimpleNamespace(message=SimpleNamespace(content=reply), finish_reason="stop")],
    )
    reference_client.chat.completions.create.return_value = completion
    hosted_client.chat.completions.create.return_value = completion
    monkeypatch.setattr(original_llm, "OpenAI", lambda **kwargs: reference_client)
    monkeypatch.setattr(original_llm.time, "sleep", lambda seconds: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("adversariallm.lm_utils.text_generation.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr("adversariallm.lm_utils.text_generation.OpenAI", lambda **kwargs: hosted_client)

    _, expected_messages = author(victim_llm=MODELS[0]).generate(PROMPTS[0])
    reference_text = original_llm.LLM(model_id=MODELS[0]).response(expected_messages)
    target = APITarget({"backend": "openrouter", "id": MODELS[0], "provider": "test"})
    settings = config()
    settings.generation_config.max_new_tokens = limit
    step = FlipAttack(settings).run(target, dataset(PROMPTS[:1])).runs[0].steps[0]
    original_request = reference_client.chat.completions.create.call_args.kwargs
    actual_request = hosted_client.chat.completions.create.call_args.kwargs
    expected_request = {
        **original_request,
        "extra_body": {"provider": {"only": ["test"], "allow_fallbacks": False, "require_parameters": True}},
    }
    if limit is not None:
        expected_request["max_tokens"] = limit
    assert actual_request == expected_request
    # Preserve the research adapter's raw response; explicitly prove and document
    # this departure from the original GPT transport's .strip().
    assert step.model_completions == [reply]
    assert reference_text == reply.strip()
    assert hosted_client.chat.completions.create.call_count == 1
    assert reference_client.chat.completions.create.call_count == 1


@pytest.mark.parametrize("original", [[], [{"role": "user", "content": "Hello"}], [
    {"role": "assistant", "content": "Hello"}, {"role": "user", "content": "World"},
]])
def test_invalid_dataset_conversations_fail_before_requests(hosted, original):
    target, generator = hosted
    with pytest.raises(ValueError, match="requires a user request"):
        FlipAttack(config()).run(target, [original])
    generator.generate.assert_not_called()


def test_runner_collects_executes_and_logs_registered_flipattack(monkeypatch, hosted):
    import run_attacks

    target, generator = hosted

    class BenignDataset(list):
        config_idx = [0]

        def __init__(self, cfg):
            super().__init__(dataset(PROMPTS[:1]))

    monkeypatch.setattr(run_attacks.PromptDataset, "from_name", lambda name: BenignDataset)
    monkeypatch.setattr(run_attacks, "filter_config", lambda rc, length, overwrite: rc)
    monkeypatch.setattr(run_attacks, "APITarget", lambda cfg: target)
    monkeypatch.setattr(run_attacks, "load_model_and_tokenizer",
                        lambda cfg: pytest.fail("Hosted FlipAttack attempted to load local weights"))
    logged = []
    monkeypatch.setattr(run_attacks, "log_attack", lambda *args: logged.append(args))
    cfg = OmegaConf.create({
        "models": {"hosted": target.config}, "model": "hosted",
        "datasets": {"benign": {"idx": [0]}}, "dataset": "benign",
        "attacks": {"flip_attack": asdict(FlipAttackConfig())}, "attack": "flip_attack",
        "defenses": {"none": {"name": "none"}}, "defense": "none", "overwrite": True,
    })
    runs = run_attacks.collect_configs(cfg)
    run_attacks.run_attacks(runs, cfg, "test")
    assert len(runs) == len(logged) == generator.generate.call_count == 1
    assert logged[0][0].attack == "flip_attack"
    assert logged[0][1].runs[0].steps[0].model_completions == ["A benign answer."]
