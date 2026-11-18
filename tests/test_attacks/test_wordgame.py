"""WordGame source-derived golden cases and hosted integration, without live model calls."""

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import pytest

from tests.wordgame_test_utils import CORE


ROOT = Path(__file__).resolve().parents[2]
CASES = json.loads((ROOT / 'tests/wordgame_cases.json').read_text())['cases']
MODEL = {'backend': 'openrouter', 'id': 'test/target', 'provider': 'target-provider'}


def config(**changes):
    return {**asdict(CORE.config()), 'attacker': {'id': 'test/attacker', 'provider': 'attacker-provider'}, **changes}


class ScriptedClient:
    base_url = 'https://openrouter.ai/api/v1'

    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []
        self.options = []
        self.chat = SimpleNamespace(completions=self)

    def with_options(self, **options):
        self.options.append(options)
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2,
                                completion_tokens_details=None, model_dump=lambda: {'total_tokens': 2})
        return SimpleNamespace(id='test', model=kwargs['model'], usage=usage,
                               choices=[SimpleNamespace(message=SimpleNamespace(content=reply), finish_reason='stop')])


def run_case(case, **changes):
    client = ScriptedClient([*case['replies'], '  A benign answer.\n'])
    dataset = [[{'role': 'user', 'content': case['goal']}, {'role': 'assistant', 'content': 'Reference.'}]]
    sleeps = []
    result = CORE.execute(CORE.driver, MODEL, config(**{**case['settings'], **changes}), dataset,
                     client=client, sleep=sleeps.append)
    return result, client, sleeps


@pytest.mark.parametrize('case', CASES, ids=lambda case: case['name'])
def test_matches_independent_pinned_jbf_golden(case):
    result, client, sleeps = run_case(case)
    record = result['records'][0]
    assert record == {'question': case['goal'], 'response': '  A benign answer.\n', **case['expected']}
    assert [call['messages'] for call in client.calls] == case['messages']
    assert sleeps == []
    assert client.options == [{'max_retries': 0}]
    assert [item['role'] for item in result['requests'][0]] == ['attacker'] * len(case['replies']) + ['target']
    for call in client.calls:
        role = 'target' if call is client.calls[-1] else 'attacker'
        assert call['model'] == f'test/{role}'
        assert call['temperature'] == 1
        assert 'max_tokens' not in call and 'top_p' not in call
        assert call['extra_body']['provider'] == {
            'only': [f'{role}-provider'], 'allow_fallbacks': False, 'require_parameters': True}


@pytest.mark.parametrize('field,value', [('num_hints', True), ('num_hints', 0), ('num_hints', 1.5),
    ('num_hints', '5'), ('use_wordgame_plus', 'false'), ('use_wordgame_plus', 0),
    ('source_profile', 'other'), ('seed', 0), ('generation_config', {})])
def test_invalid_configuration_fails_before_requests(field, value):
    with pytest.raises(ValueError):
        CORE.validate_config(MODEL, config(**{field: value}))


@pytest.mark.parametrize('role', ['target', 'attacker'])
@pytest.mark.parametrize('field', ['id', 'provider'])
def test_explicit_routes_required(role, field):
    model, settings = dict(MODEL), config()
    (model if role == 'target' else settings['attacker'])[field] = None
    with pytest.raises(ValueError, match='explicit OpenRouter'):
        CORE.validate_config(model, settings)


@pytest.mark.parametrize('bad_reply', [None, RuntimeError('temporary')])
@pytest.mark.parametrize('role', ['attacker', 'target'])
def test_original_five_attempt_limit_and_error_trace(bad_reply, role):
    prefix = ['rainbows', 'Hint.'] if role == 'target' else []
    client = ScriptedClient([*prefix, *([bad_reply] * 5)])
    sleeps = []
    with pytest.raises((RuntimeError, ValueError)) as error:
        CORE.execute(CORE.driver, MODEL, config(),
                [[{'role': 'user', 'content': 'Explain rainbows.'}, {'role': 'assistant', 'content': 'Reference.'}]],
                client=client, sleep=sleeps.append)
    assert len(client.calls) == len(prefix) + 5
    assert sleeps == [1, 2, 4, 8]
    assert len(error.value.author_requests) == len(client.calls)
    assert error.value.author_requests[-1]['role'] == role


def test_retry_recovery_and_empty_dataset():
    client = ScriptedClient([RuntimeError('temporary'), 'rainbows', 'Hint.', 'Answer.'])
    sleeps = []
    result = CORE.execute(CORE.driver, MODEL, config(),
        [[{'role': 'user', 'content': 'Explain rainbows.'}, {'role': 'assistant', 'content': 'Reference.'}]],
        client=client, sleep=sleeps.append)
    assert result['records'][0]['response'] == 'Answer.' and sleeps == [1]
    result = CORE.execute(CORE.driver, MODEL, config(), [], client=ScriptedClient([]))
    assert result['records'] == result['requests'] == result['durations'] == []


@pytest.mark.parametrize('failure', [False, True])
def test_real_sdk_serialization_and_no_nested_retries(failure):
    import httpx
    from openai import APIStatusError, OpenAI

    wire, sleeps = [], []
    replies = iter(['rainbows', 'Hint.', '  Answer.\n'])

    def http(request):
        body = json.loads(request.content)
        wire.append(body)
        assert request.url.host == 'openrouter.ai'
        assert request.headers['authorization'] == 'Bearer test-key'
        if failure:
            return httpx.Response(500, json={'error': {'message': 'temporary'}}, request=request)
        return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 0,
            'model': body['model'], 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': next(replies)},
            'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}},
            request=request)

    with httpx.Client(transport=httpx.MockTransport(http)) as http_client:
        client = OpenAI(api_key='test-key', base_url='https://openrouter.ai/api/v1', http_client=http_client)
        dataset = [[{'role': 'user', 'content': 'Explain rainbows.'}, {'role': 'assistant', 'content': 'Reference.'}]]
        if failure:
            with pytest.raises(APIStatusError):
                CORE.execute(CORE.driver, MODEL, config(), dataset, client=client, sleep=sleeps.append)
            assert len(wire) == 5 and sleeps == [1, 2, 4, 8]
        else:
            result = CORE.execute(CORE.driver, MODEL, config(), dataset, client=client, sleep=sleeps.append)
            assert result['records'][0]['response'] == '  Answer.\n'
            assert [body['model'] for body in wire] == ['test/attacker', 'test/attacker', 'test/target']
        for body in wire:
            role = body['model'].split('/')[-1]
            assert body['provider'] == {'only': [f'{role}-provider'],
                                         'allow_fallbacks': False, 'require_parameters': True}
            assert body['temperature'] == 1 and 'max_tokens' not in body


def test_public_results_and_multiple_examples():
    from adversariallm.attacks.wordgame import WordGameAttack
    from adversariallm.attacks.utils.author_runtime import OpenRouterTarget

    case = CASES[0]
    target = OpenRouterTarget.__new__(OpenRouterTarget)
    target.config = MODEL
    target.client = ScriptedClient([*case['replies'], '  Answer.\n', *case['replies'], ''])
    conversation = [{'role': 'user', 'content': case['goal']}, {'role': 'assistant', 'content': 'Reference.'}]
    result = WordGameAttack(config()).run(target, [conversation, conversation])
    assert len(result.runs) == 2
    for i, run in enumerate(result.runs):
        step = run.steps[0]
        assert step.step == 0 and step.scores == {}
        assert step.model_completions == step.model_completions_raw == [['  Answer.\n'], ['']][i]
        assert step.model_input == case['messages'][-1]
        assert all(request['example'] == i for request in run.author_requests)
        assert step.author_result == run.author_result
        assert asdict(run)['steps'][0]['author_requests'] == run.author_requests


def test_registry_and_hydra_configuration():
    from adversariallm.attacks import Attack
    from adversariallm.attacks.wordgame import WordGameAttack, WordGameConfig

    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base='1.3'):
        cfg = compose(config_name='config', overrides=['hydra/launcher=basic', 'attack=wordgame',
            'model=openrouter', 'models.openrouter.id=test/target', 'models.openrouter.provider=target-provider',
            'attacks.wordgame.attacker.id=test/attacker', 'attacks.wordgame.attacker.provider=attacker-provider'])
    assert Attack.from_name('wordgame') is WordGameAttack
    assert set(cfg.attacks.wordgame) == set(asdict(WordGameConfig()))
    WordGameAttack.validate_config(cfg.models.openrouter, cfg.attacks.wordgame)
    assert cfg.attacks.wordgame.num_hints == 5 and cfg.attacks.wordgame.use_wordgame_plus is True
