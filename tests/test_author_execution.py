"""Verify required OpenRouter routes and repository author-result integration."""

from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from adversariallm.attacks import Attack
from adversariallm.attacks.author_execution import validate_author_config
from adversariallm.attacks.flip_attack import FlipAttack
from adversariallm.attacks.equacode import EquaCodeAttack
from adversariallm.defenses.api_target import APITarget
from adversariallm.defenses.registry import validate_hosted_target
from tests.author_test_utils import ROOT, author_payload


@pytest.mark.parametrize('method', ['flip_attack', 'equacode'])
def test_registry_and_yaml_retain_original_defaults(method):
    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base='1.3'):
        overrides = ['hydra/launcher=basic', f'models={method}_author', f'model={method}_author',
                     f'attack={method}']
        overrides += [f'models.{method}_author.id=test/target', f'models.{method}_author.provider=target-provider',
                      f'attacks.{method}.judge.id=test/judge', f'attacks.{method}.judge.provider=judge-provider']
        cfg = compose(config_name='config', overrides=overrides)
    settings, model = cfg.attacks[method], cfg.models[method + '_author']
    cls = Attack.from_name(method)
    validate_hosted_target(cls, model, None, settings)
    assert settings.seed is None and 'generation_config' not in settings
    if method == 'flip_attack':
        assert model.author_model == 'gpt-4-0613' and model.id == 'test/target'
        assert settings.retry_time == 1000 and settings.failed_sleep_time == settings.round_sleep_time == 1
        assert settings.max_token == -1 and settings.evaluate is False
    else:
        assert model.author_model == 'gpt-3.5-turbo' and settings.judge_model == 'GPT-4'


@pytest.mark.parametrize('method', ['flip_attack', 'equacode'])
@pytest.mark.parametrize('role,field', [('target', 'id'), ('target', 'provider'), ('judge', 'id'), ('judge', 'provider')])
def test_all_routes_require_explicit_model_and_provider(method, role, field):
    payload = author_payload(method)
    (payload['model'] if role == 'target' else payload['config']['judge'])[field] = None
    with pytest.raises(ValueError, match='explicit OpenRouter'):
        validate_author_config(method, payload['model'], payload['config'])


@pytest.mark.parametrize('method', ['flip_attack', 'equacode'])
@pytest.mark.parametrize('change', ['openrouter', 'model', 'provider', 'seed', 'generation_config'])
def test_substitutions_and_generic_overrides_are_rejected(method, change):
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
    with pytest.raises(ValueError):
        validate_author_config(method, payload['model'], payload['config'])


@pytest.mark.parametrize('method', ['flip_attack', 'equacode'])
def test_no_global_rng_seed_is_applied(monkeypatch, method):
    monkeypatch.setattr('transformers.set_seed', lambda value: pytest.fail('Author adapters must not seed RNG'))
    payload = author_payload(method)
    cls = FlipAttack if method == 'flip_attack' else EquaCodeAttack
    cls(payload['config'])


def hosted_target(payload, client=None):
    # Supply the existing hosted interface without initializing unrelated provider clients.
    target = APITarget.__new__(APITarget)
    target.config = payload['model']
    target.text_generator = SimpleNamespace(client=client)
    return target


def test_author_records_map_to_repository_results_without_reinterpreting_scores(monkeypatch):
    payload = author_payload('flip_attack')
    record = {'output': 'Answer.', 'all_prompt': [{'role': 'user', 'content': 'Question'}],
              'judge_success_dict': 1, 'judge_score_gpt4': 10, 'judge_success_gpt4': 1}
    trace = [{'role': 'target', 'raw_text': '  Answer.\n', 'metadata': {'id': 'request'}}]
    result = {'records': [record], 'requests': [trace], 'durations': [1.0],
              'author_summary': {'all_count': 1}, 'environment': {'openai': '1.109.1'}}
    monkeypatch.setattr('adversariallm.attacks.author_adapter.execute', lambda *args, **kwargs: result)
    output = FlipAttack(payload['config']).run(hosted_target(payload), payload['dataset'])
    step = output.runs[0].steps[0]
    assert step.model_completions == ['Answer.'] and step.model_completions_raw == ['  Answer.\n']
    assert step.author_result == record and step.author_requests == trace
    assert step.scores['flip_attack_author']['judge_score_gpt4'] == [10.0]
    assert output.author_summary == {'all_count': 1}


def test_original_exception_propagates_without_conversion(monkeypatch):
    payload = author_payload('flip_attack')
    failure = UnboundLocalError('response')

    def execute(*args, **kwargs):
        raise failure

    monkeypatch.setattr('adversariallm.attacks.author_adapter.execute', execute)
    with pytest.raises(UnboundLocalError) as caught:
        FlipAttack(payload['config']).run(hosted_target(payload), payload['dataset'])
    assert caught.value is failure


def test_runner_uses_author_target_without_loading_weights(monkeypatch):
    import run_attacks
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
    assert logged[0][1] is results


@pytest.mark.parametrize('method', ['flip_attack', 'equacode'])
def test_repository_binding_reuses_hosted_sdk_client_and_only_openrouter_credentials(monkeypatch, method):
    import httpx
    from openai import OpenAI
    payload = author_payload(method)
    wire = []
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-openrouter-key')
    for key in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'DEEPINFRA_API_KEY'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr('adversariallm.attacks.author_execution.time.sleep', lambda value: None)

    def http(request):
        import json
        body = json.loads(request.content)
        wire.append(body)
        assert request.url.host == 'openrouter.ai'
        assert request.headers['authorization'] == 'Bearer test-openrouter-key'
        text = '  A benign answer.\n' if len(wire) == 1 else 'Rating: [[7]]'
        return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 0,
            'model': body['model'], 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text},
            'finish_reason': 'stop'}]}, request=request)

    with httpx.Client(transport=httpx.MockTransport(http)) as transport:
        client = OpenAI(api_key='test-openrouter-key', base_url='https://openrouter.ai/api/v1', http_client=transport)
        cls = FlipAttack if method == 'flip_attack' else EquaCodeAttack
        result = cls(payload['config']).run(hosted_target(payload, client), payload['dataset'])
    assert result.runs[0].steps[0].model_completions == [
        'A benign answer.' if method == 'flip_attack' else '  A benign answer.\n']
    assert [body['model'] for body in wire] == (['test/target'] if method == 'flip_attack' else ['test/target', 'test/judge'])
    if method == 'equacode':
        assert result.runs[0].steps[0].author_result['evaluation'] == 'Rating: [[7]]'
        assert result.runs[0].steps[0].scores == {}
    for body in wire:
        role = 'target' if body['model'] == 'test/target' else 'judge'
        assert body['provider'] == {'only': [role + '-provider'], 'allow_fallbacks': False, 'require_parameters': True}


@pytest.mark.parametrize('method', ['flip_attack', 'equacode'])
def test_collection_rejects_provider_substitution_before_dataset_access(monkeypatch, method):
    import run_attacks
    payload = author_payload(method)
    payload['model']['backend'] = 'local'
    monkeypatch.setattr(run_attacks.PromptDataset, 'from_name', lambda name: pytest.fail('Dataset loaded first'))
    cfg = OmegaConf.create({'models': {'m': payload['model']}, 'model': 'm',
                           'datasets': {'d': {}}, 'dataset': 'd',
                           'attacks': {method: payload['config']}, 'attack': method,
                           'defenses': {'none': {'name': 'none'}}, 'defense': 'none'})
    with pytest.raises(ValueError, match='backend=openrouter'):
        run_attacks.collect_configs(cfg)


@pytest.mark.parametrize('method', ['flip_attack', 'equacode'])
def test_defaults_match_independent_author_files(method):
    import ast
    from adversariallm.attacks.author_execution import FLIP_MODELS
    from adversariallm.attacks.flip_attack import FlipAttackConfig
    base = ROOT / 'verification' / method / 'reference'
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
    else:
        from adversariallm.attacks.equacode import EquaCodeConfig
        values = {node.targets[0].id: ast.literal_eval(node.value)
                  for node in ast.parse((base / 'config.py').read_bytes()).body if isinstance(node, ast.Assign)}
        assert EquaCodeConfig().judge_model == values['assessmodel']
        assert values['base_url'] == values['assess_base_url'] == ''


@pytest.mark.parametrize('method', ['flip_attack', 'equacode'])
def test_sdk_transient_retry_serialization_and_delay_match_original_baselines(method):
    from tests.author_test_utils import compare_author
    payload = author_payload(method)
    payload['replies'] = [{'status': 500}, {'text': 'Answer.'}]
    if method == 'equacode':
        payload['replies'].append({'text': 'Rating: [[1]]'})
    _, result = compare_author(payload)
    assert len(result['sdk_sleeps']) == 1
    assert len(result['wire']) == (2 if method == 'flip_attack' else 3)


@pytest.mark.parametrize('method', ['flip_attack', 'equacode'])
def test_unknown_settings_are_not_silently_ignored(method):
    payload = author_payload(method)
    payload['config']['unknown_override'] = True
    with pytest.raises(ValueError, match='Unsupported author settings'):
        validate_author_config(method, payload['model'], payload['config'])


@pytest.mark.parametrize('method', ['flip_attack', 'equacode'])
def test_custom_endpoints_cannot_redirect_author_requests(method):
    payload = author_payload(method)
    payload['model']['base_url'] = 'https://api.openai.com/v1'
    with pytest.raises(ValueError, match='only https://openrouter.ai'):
        validate_author_config(method, payload['model'], payload['config'])

