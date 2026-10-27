"""Check hosted targets and shared model configuration without real API calls."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from adversariallm.attacks.direct import DirectAttack, DirectConfig
from adversariallm.defenses.api_target import APITarget
from adversariallm.defenses.registry import validate_hosted_target
from adversariallm.lm_utils.text_generation import GenerationResult, build_text_generator


def test_api_target_normalizes_messages_and_preserves_parameters(monkeypatch):
    """Check placeholders and local batch controls never reach the hosted API."""

    # Use fake generation or SDK responses to isolate the hosted target contract.
    generator = MagicMock()
    generator.generate.return_value = GenerationResult(gen=[["answer"]])
    monkeypatch.setattr("adversariallm.defenses.api_target.build_text_generator", lambda cfg: generator)
    target = APITarget({"backend": "openrouter", "id": "test/model", "provider": "test"})
    conv = [{"role": "user", "content": "question"}, {"role": "assistant", "content": ""}]
    target.generate([conv], max_new_tokens=150, top_k=0, initial_batch_size=1, verbose=True)

    # Strip hosted-inapplicable controls while retaining the caller's original conversation.
    generator.generate.assert_called_once_with([[conv[0]]], max_new_tokens=150)
    assert len(conv) == 2
    assert target.model is None and target.tokenizer is None


def test_api_target_rejects_undeclared_prefilling(monkeypatch):
    """Check that assistant prefixes require an explicit capability setting."""

    # Use fake generation or SDK responses to isolate the hosted target contract.
    generator = MagicMock()
    monkeypatch.setattr("adversariallm.defenses.api_target.build_text_generator", lambda cfg: generator)
    target = APITarget({})
    with pytest.raises(ValueError, match="prefill"):
        target.generate([[{"role": "assistant", "content": "prefix"}]])

    # Reject unsupported assistant prefixes before generation.
    generator.generate.assert_not_called()


def test_api_target_rejects_local_loss(monkeypatch):
    """Check that a hosted target does not invent local loss measurements."""

    # Use fake generation or SDK responses to isolate the hosted target contract.
    monkeypatch.setattr("adversariallm.defenses.api_target.build_text_generator", lambda cfg: MagicMock())

    # Hosted targets must report that local loss measurements are unavailable.
    with pytest.raises(ValueError, match="local model losses"):
        APITarget({}).loss([], [], initial_batch_size=1)


def test_openrouter_generator_pins_provider_and_samples_separately(monkeypatch):
    """Check provider pinning, token controls, independent samples, and metadata."""

    # Use fake generation or SDK responses to isolate the hosted target contract.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("adversariallm.lm_utils.text_generation.load_dotenv", lambda *a, **kw: None)
    client = MagicMock()
    completion = SimpleNamespace(id="test-id", model="test/model", provider="test",
                                 usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 7}),
                                 choices=[SimpleNamespace(message=SimpleNamespace(content="answer"),
                                                          finish_reason="stop")])
    client.chat.completions.create.return_value = completion
    monkeypatch.setattr("adversariallm.lm_utils.text_generation.OpenAI", lambda **kw: client)
    generator = build_text_generator({"backend": "openrouter", "id": "test/model", "provider": "test"})
    result = generator.generate([[{"role": "user", "content": "question"}]],
                                max_new_tokens=150, num_return_sequences=2)

    # Keep provider pinning, separate samples, and API metadata aligned.
    assert result.gen == [["answer", "answer"]]
    assert client.chat.completions.create.call_count == 2
    request = client.chat.completions.create.call_args.kwargs
    assert request["model"] == "test/model" and request["max_tokens"] == 150
    assert "n" not in request and "max_completion_tokens" not in request
    assert request["extra_body"]["provider"] == {
        "only": ["test"], "allow_fallbacks": False, "require_parameters": True,
    }
    assert result.api_metadata[0][0]["usage"] == {"total_tokens": 7}
    generator.generate([[{"role": "user", "content": "question"}]], top_k=7,
                       extra_body=OmegaConf.create({"provider": {"only": ["other-provider"]},
                                                   "reasoning": {"enabled": False}}))
    body = client.chat.completions.create.call_args.kwargs["extra_body"]
    assert body["provider"]["only"] == ["test"] and body["top_k"] == 7
    assert body["reasoning"] == {"enabled": False}
    json.dumps(body)


def test_api_failure_is_not_a_model_response(monkeypatch):
    """Check exhausted requests raise instead of being judged as refusals."""

    # Use fake generation or SDK responses to isolate the hosted target contract.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("adversariallm.lm_utils.text_generation.load_dotenv", lambda *a, **kw: None)
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("request failed")
    monkeypatch.setattr("adversariallm.lm_utils.text_generation.OpenAI", lambda **kw: client)
    monkeypatch.setattr("adversariallm.lm_utils.text_generation.time.sleep", lambda _: None)
    generator = build_text_generator({"backend": "openrouter", "id": "test/model", "provider": "test"})
    with pytest.raises(RuntimeError, match="request failed"):
        generator.generate([[{"role": "user", "content": "question"}]])

    # Propagate transport failure rather than treating it as model output.
    assert client.chat.completions.create.call_count == 3


@pytest.mark.parametrize("fails", [False, True])
def test_retry_controls_are_consumed_before_the_api_call(monkeypatch, fails):
    """Check explicit retry settings avoid duplicate arguments and limit failed calls."""

    # Use fake generation or SDK responses to isolate the hosted target contract.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("adversariallm.lm_utils.text_generation.load_dotenv", lambda *a, **kw: None)
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        id="test", model="test/model", provider="test", usage=None,
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"), finish_reason="stop")],
    )
    if fails:
        client.chat.completions.create.side_effect = RuntimeError("request failed")
    monkeypatch.setattr("adversariallm.lm_utils.text_generation.OpenAI", lambda **kw: client)
    generator = build_text_generator({"backend": "openrouter", "id": "test/model", "provider": "test"})
    conversations = [[{"role": "user", "content": "question"}]]
    retry_settings = {"max_retries": 1, "retry_sleep": 0, "retry_backoff": 1.0}
    if fails:
        with pytest.raises(RuntimeError, match="request failed"):
            generator.generate(conversations, **retry_settings)
    else:
        assert generator.generate(conversations, **retry_settings).gen == [["answer"]]

    # Consume retry settings locally and enforce the selected attempt limit.
    assert client.chat.completions.create.call_count == 1
    assert not set(retry_settings).intersection(client.chat.completions.create.call_args.kwargs)


@pytest.mark.parametrize("content,has_choice", [(None, True), ("", True), (None, False)])
def test_empty_completion_preserves_usage_without_paid_retries(monkeypatch, content, has_choice):
    """Check token-exhausted answers retain diagnostics and are not regenerated."""

    # Use fake generation or SDK responses to isolate the hosted target contract.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("adversariallm.lm_utils.text_generation.load_dotenv", lambda *a, **kw: None)
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        id="empty", model="test/model", provider="test",
        usage=SimpleNamespace(model_dump=lambda: {"completion_tokens": 150}),
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="length")] if has_choice else [],
    )
    monkeypatch.setattr("adversariallm.lm_utils.text_generation.OpenAI", lambda **kw: client)
    generator = build_text_generator({"backend": "openrouter", "id": "test/model", "provider": "test"})
    with pytest.raises(ValueError, match="finish reasons"):
        generator.generate([[{"role": "user", "content": "question"}]])

    # Keep billed usage diagnostics and avoid repeating an empty completion.
    assert client.chat.completions.create.call_count == 1
    assert generator.last_response_metadata["usage"]["completion_tokens"] == 150


def test_local_generator_uses_existing_loader(monkeypatch):
    """Check local role configuration still uses the repository's model loader."""

    # Replace the local loader and generator to check construction without model weights.
    pytest.importorskip("transformers", reason="Requires the local-inference environment")
    loader = MagicMock(return_value=("model", "tokenizer"))
    generator = MagicMock()
    monkeypatch.setattr("adversariallm.lm_utils.text_generation.load_model_and_tokenizer", loader)
    monkeypatch.setattr("adversariallm.lm_utils.text_generation.LocalTextGenerator", generator)
    cfg = {"id": "local/model", "tokenizer_id": "local/model"}
    build_text_generator(cfg)

    # Leave the established local model/tokenizer construction path intact.
    loader.assert_called_once_with(cfg)
    generator.assert_called_once_with("model", "tokenizer", default_generate_kwargs=None)


@pytest.mark.parametrize("attack,defense,provider", [
    ("gcg", "none", "test"), ("direct", "polyguard", "test"), ("direct", "none", None),
])
def test_hosted_validation_rejects_unsupported_combinations(monkeypatch, attack, defense, provider):
    """Check incompatible methods and missing routing settings fail early."""

    # Use fake generation or SDK responses to isolate the hosted target contract.
    attack_cls = SimpleNamespace(__name__=attack, SUPPORTS_HOSTED_TARGET=attack == "direct")

    # Reject incompatible attacks, defenses, or missing provider routing.
    with pytest.raises(ValueError):
        validate_hosted_target(attack_cls, {"backend": "openrouter", "id": "test/model", "provider": provider},
                               {"name": defense})


def test_direct_hosted_target_keeps_measurements_optional():
    """Check the baseline records hosted outputs without trying local tokenization."""

    # Use fake generation or SDK responses to isolate the hosted target contract.
    metadata = [[{"id": "request-1", "usage": {"total_tokens": 7}}]]
    target = SimpleNamespace(model=None, tokenizer=None,
                             generate=lambda convs, **kwargs: GenerationResult(gen=[["answer"]], api_metadata=metadata))
    dataset = [[{"role": "user", "content": "question"}, {"role": "assistant", "content": "target"}]]
    step = DirectAttack(DirectConfig()).run(target, dataset).runs[0].steps[0]

    # Retain hosted answers while leaving unavailable local measurements empty.
    assert step.model_completions == ["answer"]
    assert step.model_input[-1]["content"] == ""
    assert step.loss is None and step.model_input_tokens is None
    assert step.api_metadata == metadata[0]


def test_hosted_metadata_serializes_with_existing_result_format():
    from dataclasses import asdict
    from adversariallm.attacks.attack import AttackStepResult

    # Use fake generation or SDK responses to isolate the hosted target contract.
    step = AttackStepResult(step=0, model_completions=["answer"],
                            api_metadata=[{"id": "request-1", "usage": {"total_tokens": 7}}])
    payload = json.loads(json.dumps(asdict(step)))

    # Preserve request metadata through the existing result serialization format.
    assert payload["api_metadata"] == step.api_metadata
    assert AttackStepResult(step=0, model_completions=["legacy"]).api_metadata is None


def test_openrouter_configuration_requires_explicit_selection():
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from omegaconf.errors import MissingMandatoryValue

    # Use fake generation or SDK responses to isolate the hosted target contract.
    conf_dir = str(Path(__file__).resolve().parents[1] / "conf")
    with initialize_config_dir(config_dir=conf_dir, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[
            "models=openrouter", "model=openrouter", "attack=direct", "hydra/launcher=basic",
        ])
        with pytest.raises(MissingMandatoryValue):
            OmegaConf.to_container(cfg.models, resolve=True, throw_on_missing=True)
        cfg = compose(config_name="config", overrides=[
            "models=openrouter", "model=openrouter", "attack=direct",
            "models.openrouter.id=test/model", "models.openrouter.provider=test",
            "hydra/launcher=basic",
        ])

    # Require explicit hosted model/provider selection in composed configuration.
    assert cfg.models.openrouter.id == "test/model"
    assert cfg.models.openrouter.provider == "test"
    assert cfg.defense == "none"


def test_openrouter_omits_unspecified_generation_settings(monkeypatch):
    # Use fake generation or SDK responses to isolate the hosted target contract.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("adversariallm.lm_utils.text_generation.load_dotenv", lambda *a, **kw: None)
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        id="test", model="test/model", provider="test", usage=None,
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"), finish_reason="stop")],
    )
    monkeypatch.setattr("adversariallm.lm_utils.text_generation.OpenAI", lambda **kw: client)
    generator = build_text_generator({"backend": "openrouter", "id": "test/model", "provider": "test"})
    messages = [{"role": "user", "content": "Hello, 世界\n"}]
    generator.generate([messages])

    # Keep unspecified generation controls absent from the SDK request.
    client.chat.completions.create.assert_called_once_with(
        model="test/model", messages=messages,
        extra_body={"provider": {"only": ["test"], "allow_fallbacks": False, "require_parameters": True}},
    )


def test_hosted_batch_sample_metadata_keeps_request_order(monkeypatch):
    # Use fake generation or SDK responses to isolate the hosted target contract.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("adversariallm.lm_utils.text_generation.load_dotenv", lambda *a, **kw: None)
    client = MagicMock()
    client.chat.completions.create.side_effect = [SimpleNamespace(
        id=f"request-{i}", model="test/model", provider="test", usage=None,
        choices=[SimpleNamespace(message=SimpleNamespace(content=f"answer-{i}"), finish_reason="stop")],
    ) for i in range(4)]
    monkeypatch.setattr("adversariallm.lm_utils.text_generation.OpenAI", lambda **kw: client)
    generator = build_text_generator({"backend": "openrouter", "id": "test/model", "provider": "test"})
    conversations = [[{"role": "user", "content": prompt}] for prompt in ["Hello", "Goodbye"]]
    result = generator.generate(conversations, num_return_sequences=2)

    # Keep completion and metadata rows in conversation/sample order.
    assert result.gen == [["answer-0", "answer-1"], ["answer-2", "answer-3"]]
    assert [[item["id"] for item in row] for row in result.api_metadata] == [
        ["request-0", "request-1"], ["request-2", "request-3"],
    ]
    assert [call.kwargs["messages"] for call in client.chat.completions.create.call_args_list] == [
        conversations[0], conversations[0], conversations[1], conversations[1],
    ]
