"""Verify required OpenRouter routes and repository author-result integration."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from adversariallm.attacks import Attack
from adversariallm.attacks.author_adapter import AuthorRunResult, AuthorStepResult
from adversariallm.attacks.author_execution import validate_author_config
from adversariallm.attacks.flip_attack import FlipAttack
from adversariallm.attacks.equacode import EquaCodeAttack
from adversariallm.attacks.gta import GTAAttack
from adversariallm.defenses.api_target import APITarget
from adversariallm.defenses.registry import validate_hosted_target
from tests.author_test_utils import ROOT, author_payload


@pytest.mark.parametrize('method', ['flip_attack', 'equacode', 'gta'])
def test_registry_and_yaml_retain_original_defaults(method):
    # Keep registry selection and composed settings aligned with the source profile.
    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base='1.3'):
        overrides = ['hydra/launcher=basic', f'models={method}_author', f'model={method}_author',
                     f'attack={method}']
        overrides += [f'models.{method}_author.id=test/target', f'models.{method}_author.provider=target-provider',
                      f'attacks.{method}.judge.id=test/judge', f'attacks.{method}.judge.provider=judge-provider']
        if method == 'gta':
            overrides += ['attacks.gta.attacker.id=test/attacker', 'attacks.gta.attacker.provider=attacker-provider']
        cfg = compose(config_name='config', overrides=overrides)
    settings, model = cfg.attacks[method], cfg.models[method + '_author']
    cls = Attack.from_name(method)
    validate_hosted_target(cls, model, None, settings)

    # Keep registry selection and composed settings aligned with the source profile.
    assert settings.seed is None and 'generation_config' not in settings
    if method == 'flip_attack':
        assert model.author_model == 'gpt-4-0613' and model.id == 'test/target'
        assert settings.retry_time == 1000 and settings.failed_sleep_time == settings.round_sleep_time == 1
        assert settings.max_token == -1 and settings.evaluate is False
    elif method == 'equacode':
        assert model.author_model == 'gpt-3.5-turbo' and settings.judge_model == 'GPT-4'
    elif method == 'gta':
        assert model.author_model == 'deepseek_r1' and settings.rounds == 5


@pytest.mark.parametrize('method', ['flip_attack', 'equacode', 'gta'])
@pytest.mark.parametrize('role,field', [('target', 'id'), ('target', 'provider'), ('judge', 'id'), ('judge', 'provider')])
def test_all_routes_require_explicit_model_and_provider(method, role, field):
    # Prepare the selected author configuration and controlled repository dependencies.
    payload = author_payload(method)
    (payload['model'] if role == 'target' else payload['config']['judge'])[field] = None

    # Require each configured role to specify both routing fields.
    with pytest.raises(ValueError, match='explicit OpenRouter'):
        validate_author_config(method, payload['model'], payload['config'])


@pytest.mark.parametrize('method', ['flip_attack', 'equacode', 'gta'])
@pytest.mark.parametrize('change', ['openrouter', 'model', 'provider', 'seed', 'generation_config'])
def test_substitutions_and_generic_overrides_are_rejected(method, change):
    # Prepare the selected author configuration and controlled repository dependencies.
    payload = author_payload(method)
    if change == 'openrouter':
        payload['model']['backend'] = 'local'
    elif change == 'model':
        payload['model']['author_model'] = 'replacement-model'
    elif change == 'provider':
        payload['model']['provider'] = ''
    elif change == 'seed':
        payload['config']['seed'] = 0
    else:
        payload['config']['generation_config'] = {'max_new_tokens': 150}

    # Reject settings that would silently change the supported author protocol.
    with pytest.raises(ValueError):
        validate_author_config(method, payload['model'], payload['config'])


@pytest.mark.parametrize('method', ['flip_attack', 'equacode', 'gta'])
def test_no_global_rng_seed_is_applied(monkeypatch, method):
    # Prepare the selected author configuration and controlled repository dependencies.
    monkeypatch.setattr('transformers.set_seed', lambda value: pytest.fail('Author adapters must not seed RNG'))
    payload = author_payload(method)
    cls = {'flip_attack': FlipAttack, 'equacode': EquaCodeAttack, 'gta': GTAAttack}[method]
    cls(payload['config'])


def hosted_target(payload, client=None):
    # Supply the existing hosted interface without initializing unrelated provider clients.
    target = APITarget.__new__(APITarget)
    target.config = payload['model']
    target.text_generator = SimpleNamespace(client=client)
    return target


def test_author_records_map_to_repository_results_without_reinterpreting_scores(monkeypatch):
    # Prepare the selected author configuration and controlled repository dependencies.
    payload = author_payload('flip_attack')
    record = {'output': 'Answer.', 'all_prompt': [{'role': 'user', 'content': 'Question'}],
              'judge_success_dict': 1, 'judge_score_gpt4': 10, 'judge_success_gpt4': 1}
    trace = [{'role': 'target', 'raw_text': '  Answer.\n', 'metadata': {'id': 'request'}}]
    result = {'records': [record], 'requests': [trace], 'durations': [1.0],
              'author_summary': {'all_count': 1}, 'environment': {'openai': '1.109.1'}}
    monkeypatch.setattr('adversariallm.attacks.author_adapter.execute', lambda *args, **kwargs: result)
    output = FlipAttack(payload['config']).run(hosted_target(payload), payload['dataset'])
    step = output.runs[0].steps[0]

    # Carry original records, raw replies, and author scores into repository results.
    assert step.model_completions == ['Answer.'] and step.model_completions_raw == ['  Answer.\n']
    assert step.author_result == record and step.author_requests == trace
    assert step.scores['flip_attack_author']['judge_score_gpt4'] == [10.0]
    assert output.author_summary == {'all_count': 1}


def test_original_exception_propagates_without_conversion(monkeypatch):
    # Prepare the selected author configuration and controlled repository dependencies.
    payload = author_payload('flip_attack')
    failure = UnboundLocalError('response')

    def execute(*args, **kwargs):
        raise failure

    monkeypatch.setattr('adversariallm.attacks.author_adapter.execute', execute)
    with pytest.raises(UnboundLocalError) as caught:
        FlipAttack(payload['config']).run(hosted_target(payload), payload['dataset'])

    # Keep the exact original exception object when author execution fails.
    assert caught.value is failure


def test_runner_uses_author_target_without_loading_weights(monkeypatch):
    import run_attacks

    # Prepare the selected author configuration and controlled repository dependencies.
    payload = author_payload('flip_attack')
    target, results = object(), object()

    class FakeAttack:
        def __init__(self, config):
            pass

        def run(self, actual, dataset):
            assert actual is target
            return results

    monkeypatch.setattr(run_attacks, 'APITarget', lambda cfg: target)
    monkeypatch.setattr(run_attacks, 'load_model_and_tokenizer', lambda cfg: pytest.fail('Loaded local weights'))
    monkeypatch.setattr(run_attacks.Attack, 'from_name', lambda name: FakeAttack)
    monkeypatch.setattr(run_attacks.PromptDataset, 'from_name', lambda name: (lambda cfg: []))
    logged = []
    monkeypatch.setattr(run_attacks, 'log_attack', lambda *args: logged.append(args))
    rc = run_attacks.RunConfig(model='author', dataset='d', attack='flip_attack', defense=None,
                              model_params=payload['model'], dataset_params={}, attack_params=payload['config'])
    run_attacks.run_attacks([rc], OmegaConf.create({}), 'test')

    # Use the hosted author target through the normal runner and logging interface.
    assert logged[0][1] is results


@pytest.mark.parametrize('method', ['flip_attack', 'equacode', 'gta'])
def test_repository_binding_reuses_hosted_sdk_client_and_only_openrouter_credentials(monkeypatch, method):
    import httpx
    from openai import OpenAI

    # Prepare the selected author configuration and controlled repository dependencies.
    payload = author_payload(method)
    if method == 'gta':
        payload['config']['rounds'] = 1
    wire = []
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-openrouter-key')
    for key in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'DEEPINFRA_API_KEY'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr('adversariallm.attacks.author_execution.time.sleep', lambda value: None)

    # Use real SDK serialization with scripted in-memory HTTP responses.
    def http(request):
        import json
        body = json.loads(request.content)
        wire.append(body)
        assert request.url.host == 'openrouter.ai'
        assert request.headers['authorization'] == 'Bearer test-openrouter-key'
        if method == 'gta':
            text = ['<think>Plan.</think><response>Explain rainbows.</response>',
                    '<think>Reason.</think>  A benign answer.\n', 'Rating: [[7]]'][(len(wire) - 1) % 3]
        else:
            text = '  A benign answer.\n' if len(wire) == 1 else 'Rating: [[7]]'
        return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 0,
            'model': body['model'], 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text},
            'finish_reason': 'stop'}]}, request=request)

    # Exercise the public attack entry point with the target's existing SDK client.
    with httpx.Client(transport=httpx.MockTransport(http)) as transport:
        client = OpenAI(api_key='test-openrouter-key', base_url='https://openrouter.ai/api/v1', http_client=transport)
        cls = {'flip_attack': FlipAttack, 'equacode': EquaCodeAttack, 'gta': GTAAttack}[method]
        result = cls(payload['config']).run(hosted_target(payload, client), payload['dataset'])
    # All entry points retain full run evidence and serialize through repository dataclasses.
    run = result.runs[0]

    # Keep the real SDK on OpenRouter and retain complete run/step evidence.
    assert isinstance(run, AuthorRunResult)
    assert all(isinstance(step, AuthorStepResult) for step in run.steps)
    assert [step.step for step in run.steps] == list(range(len(run.steps)))
    assert run.author_result and len(run.author_requests) == len(wire)
    serialized = asdict(result)['runs'][0]
    assert serialized['author_result'] == run.author_result
    assert serialized['author_requests'] == run.author_requests
    if method in ('flip_attack', 'equacode'):
        assert run.author_result == run.steps[0].author_result
        assert run.author_requests == run.steps[0].author_requests
        assert run.steps[0].time_taken == run.total_time

    assert result.runs[0].steps[0].model_completions == [
        'A benign answer.' if method == 'flip_attack' else '  A benign answer.\n']
    expected = (['test/target'] if method == 'flip_attack' else
                ['test/target', 'test/judge'] if method == 'equacode' else
                ['test/attacker', 'test/target', 'test/judge'] * 2)
    assert [body['model'] for body in wire] == expected
    if method == 'gta':
        assert len(result.runs[0].steps) == 2
        assert result.runs[0].steps[0].scores == {'gta_author': {'rating': [7.0]}}
        assert result.runs[0].author_result['conversation'][0][0] == 'Thomas Background'
    if method == 'equacode':
        assert result.runs[0].steps[0].author_result['evaluation'] == 'Rating: [[7]]'
        assert result.runs[0].steps[0].scores == {}
    for body in wire:
        role = body['model'].split('/')[-1]
        assert body['provider'] == {'only': [role + '-provider'], 'allow_fallbacks': False, 'require_parameters': True}


@pytest.mark.parametrize('method', ['flip_attack', 'equacode', 'gta'])
def test_collection_rejects_provider_substitution_before_dataset_access(monkeypatch, method):
    import run_attacks

    # Prepare the selected author configuration and controlled repository dependencies.
    payload = author_payload(method)
    payload['model']['backend'] = 'local'
    monkeypatch.setattr(run_attacks.PromptDataset, 'from_name', lambda name: pytest.fail('Dataset loaded first'))
    cfg = OmegaConf.create({'models': {'m': payload['model']}, 'model': 'm',
                           'datasets': {'d': {}}, 'dataset': 'd',
                           'attacks': {method: payload['config']}, 'attack': method,
                           'defenses': {'none': {'name': 'none'}}, 'defense': 'none'})

    # Validate the hosted route before touching the dataset.
    with pytest.raises(ValueError, match='backend=openrouter'):
        run_attacks.collect_configs(cfg)


@pytest.mark.parametrize('method', ['flip_attack', 'equacode', 'gta'])
def test_defaults_match_independent_author_files(method):
    import ast
    from adversariallm.attacks.author_execution import FLIP_MODELS
    from adversariallm.attacks.flip_attack import FlipAttackConfig

    # Prepare the selected author configuration and controlled repository dependencies.
    base = ROOT / 'verification' / method / 'reference'

    # Compare repository defaults with independently bundled original declarations.
    if method == 'flip_attack':
        main = ast.parse((base / 'src/main.py').read_bytes())
        arguments = {node.args[0].value: {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords
                                       if kw.arg in {'default', 'choices', 'action'}}
                     for node in ast.walk(main) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument'}
        defaults = FlipAttackConfig()
        for field in ('temperature', 'max_token', 'retry_time', 'failed_sleep_time', 'round_sleep_time', 'flip_mode'):
            assert getattr(defaults, field) == arguments['--' + field]['default']
        assert defaults.judge_model == arguments['--judge_llm']['default']
        assert set(FLIP_MODELS) == set(arguments['--victim_llm']['choices'])
    elif method == 'equacode':
        from adversariallm.attacks.equacode import EquaCodeConfig
        values = {node.targets[0].id: ast.literal_eval(node.value)
                  for node in ast.parse((base / 'config.py').read_bytes()).body if isinstance(node, ast.Assign)}
        assert EquaCodeConfig().judge_model == values['assessmodel']
        assert values['base_url'] == values['assess_base_url'] == ''
    elif method == 'gta':
        from adversariallm.attacks.gta import GTAConfig
        tree = ast.parse((base / 'utils.py').read_bytes())
        driver = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'ours_attack')
        assert GTAConfig().rounds == ast.literal_eval(driver.args.defaults[0])
        script = (base / 'run_attacks.sh').read_text()
        assert '--target1 deepseek_r1 --target2 deepseek_r1 --attacker gpt-4o-mini' in script



@pytest.mark.parametrize('method', ['flip_attack', 'equacode', 'gta'])
def test_sdk_transient_retry_serialization_and_delay_match_original_baselines(method):
    from tests.author_test_utils import compare_author

    # Prepare the selected author configuration and controlled repository dependencies.
    payload = author_payload(method)
    payload['replies'] = [{'status': 500}, {'text': 'Answer.'}]
    if method == 'equacode':
        payload['replies'].append({'text': 'Rating: [[1]]'})
    elif method == 'gta':
        from tests.test_attacks.test_gta import round_replies
        payload['config']['rounds'] = 1
        payload['replies'] = [{'status': 500}, *round_replies()]

    # Run the independent original and integrated paths with identical scripted replies.
    _, result = compare_author(payload)

    # Keep SDK retry bodies and delays equivalent to the declared baseline.
    assert len(result['sdk_sleeps']) == 1
    assert len(result['wire']) == {'flip_attack': 2, 'equacode': 3, 'gta': 7}[method]


@pytest.mark.parametrize('method', ['flip_attack', 'equacode', 'gta'])
def test_unknown_settings_are_not_silently_ignored(method):
    # Prepare the selected author configuration and controlled repository dependencies.
    payload = author_payload(method)
    payload['config']['unknown_override'] = True

    # Reject undeclared configuration fields rather than silently accepting them.
    with pytest.raises(ValueError, match='Unsupported author settings'):
        validate_author_config(method, payload['model'], payload['config'])


@pytest.mark.parametrize('method', ['flip_attack', 'equacode', 'gta'])
def test_custom_endpoints_cannot_redirect_author_requests(method):
    # Prepare the selected author configuration and controlled repository dependencies.
    payload = author_payload(method)
    payload['model']['base_url'] = 'https://api.openai.com/v1'

    # Reject endpoint overrides on the fixed author OpenRouter transport.
    with pytest.raises(ValueError, match='only https://openrouter.ai'):
        validate_author_config(method, payload['model'], payload['config'])



@pytest.mark.parametrize('field', ['id', 'provider'])
def test_gta_requires_explicit_attacker_routing(field):
    # Prepare the selected author configuration and controlled repository dependencies.
    payload = author_payload('gta')
    payload['config']['attacker'][field] = None

    # Require explicit attacker routing independently of target and judge routes.
    with pytest.raises(ValueError, match='attacker requires an explicit OpenRouter'):
        validate_author_config('gta', payload['model'], payload['config'])


@pytest.mark.parametrize('value', [True, '5', 1.5])
def test_gta_rounds_match_author_cli_integer_type(value):
    # Prepare the selected author configuration and controlled repository dependencies.
    payload = author_payload('gta', rounds=value)

    # Keep the original integer argument semantics for GTA rounds.
    with pytest.raises(ValueError, match='integer'):
        validate_author_config('gta', payload['model'], payload['config'])


def test_gta_preserves_caught_none_response_without_inventing_a_refusal(monkeypatch):
    # Prepare the selected author configuration and controlled repository dependencies.
    payload = author_payload('gta')
    record = {'conversation': [['Thomas', None]], 'judgements': [
        {'role': 'Thomas', 'round': 1, 'response': None, 'rating': -1, 'analysis': '', 'request_end': 1}]}
    result = {'records': [record], 'requests': [[{'role': 'Thomas', 'request': {'messages': []},
                                               'error_type': 'NotFoundError'}]],
              'durations': [1.0], 'author_summary': {}, 'environment': {}}
    monkeypatch.setattr('adversariallm.attacks.author_adapter.execute', lambda *args, **kwargs: result)
    output = GTAAttack(payload['config']).run(hosted_target(payload), payload['dataset'])

    # Retain the source None record and expose an absent repository completion.
    assert output.runs[0].author_result['judgements'][0]['response'] is None
    assert output.runs[0].steps[0].model_completions == []
    assert output.runs[0].steps[0].scores == {'gta_author': {'rating': [-1.0]}}


def test_gta_background_only_results_remain_available_without_target_steps(monkeypatch):
    # Prepare the selected author configuration and controlled repository dependencies.
    payload = author_payload('gta', rounds=0)
    record = {'conversation': [['Thomas Background', 'Original']], 'judgements': []}
    result = {'records': [record], 'requests': [[]], 'durations': [0.0], 'author_summary': {}, 'environment': {}}
    monkeypatch.setattr('adversariallm.attacks.author_adapter.execute', lambda *args, **kwargs: result)
    output = GTAAttack(payload['config']).run(hosted_target(payload), payload['dataset'])

    # Keep the original background record even when no target turn exists.
    assert output.runs[0].steps == [] and output.runs[0].author_result == record


@pytest.mark.parametrize('method', ['flip_attack', 'equacode', 'gta'])
def test_shared_entry_point_rejects_non_hosted_targets_before_execution(monkeypatch, method):
    # Prepare the selected author configuration and controlled repository dependencies.
    payload = author_payload(method)
    monkeypatch.setattr('adversariallm.attacks.author_adapter.execute',
                        lambda *args, **kwargs: pytest.fail('Executed with a non-hosted target'))

    # Reject incompatible targets before the original driver is called.
    with pytest.raises(ValueError, match='requires the OpenRouter author-driver target'):
        Attack.from_name(method)(payload['config']).run(object(), payload['dataset'])


@pytest.mark.parametrize('method', ['flip_attack', 'equacode', 'gta'])
@pytest.mark.parametrize('fails', [False, True])
def test_result_mapping_does_not_consume_author_rng_state(monkeypatch, method, fails):
    import random
    from adversariallm.attacks.author_adapter import adapt_results

    # Prepare the selected author configuration and controlled repository dependencies.
    original_state = random.getstate()
    payload = author_payload(method)
    finished_state = random.Random(123).getstate()

    def execute(*args, **kwargs):
        random.setstate(finished_state)
        return {'records': [], 'requests': [], 'durations': [], 'author_summary': {}, 'environment': {}}

    # Simulate result bookkeeping that consumes RNG state and can also fail.
    def mapping(*args):
        random.random()
        if fails:
            raise ValueError('Bookkeeping failed')
        return adapt_results(*args)

    monkeypatch.setattr('adversariallm.attacks.author_adapter.execute', execute)
    monkeypatch.setattr('adversariallm.attacks.author_adapter.adapt_results', mapping)

    # Keep the post-driver RNG state intact on successful and failed result conversion.
    try:
        if fails:
            with pytest.raises(ValueError, match='Bookkeeping failed'):
                Attack.from_name(method)(payload['config']).run(hosted_target(payload), payload['dataset'])
        else:
            Attack.from_name(method)(payload['config']).run(hosted_target(payload), payload['dataset'])
        assert random.getstate() == finished_state
    finally:
        random.setstate(original_state)
