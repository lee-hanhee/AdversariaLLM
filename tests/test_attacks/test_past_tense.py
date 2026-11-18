"""Past-Tense hosted integration and unit tests, without live model calls."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import pytest

from tests.jbf_test_utils import load_jbf_core


ROOT = Path(__file__).resolve().parents[2]
CORE = load_jbf_core('past_tense', 'PastTenseConfig', 'PastTenseAttack')
MODEL = {'backend': 'openrouter', 'id': 'test/target', 'provider': 'target-provider'}


def config(**changes):
    return {
        **asdict(CORE.config()),
        'attacker': {'id': 'test/attacker', 'provider': 'attacker-provider'},
        'judge': {'id': 'test/judge', 'provider': 'judge-provider'},
        **changes,
    }


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
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2,
                                completion_tokens_details=None, model_dump=lambda: {'total_tokens': 2})
        return SimpleNamespace(id='test', model=kwargs['model'], usage=usage,
                               choices=[SimpleNamespace(message=SimpleNamespace(content=reply), finish_reason='stop')])


def test_past_tense_restarts_early_stopping_and_quote_stripping():
    # Restart 1 fails judge (Rating: [[3]]); Restart 2 succeeds (Rating: [[10]]) and stops early.
    client = ScriptedClient([
        '"How were widgets built in 2020?"',
        'I cannot help with that.',
        'Rating: [[3]]',
        '"How did people build widgets in the past?"',
        'Historically, widgets were built by...',
        'Rating: [[10]]',
    ])
    dataset = [[{'role': 'user', 'content': 'Build a widget'}, {'role': 'assistant', 'content': 'Reference.'}]]
    result = CORE.execute(CORE.driver, MODEL, config(n_restarts=5), dataset, client=client, sleep=lambda _: None)

    assert len(client.calls) == 6
    assert [c['model'] for c in client.calls] == [
        'test/attacker', 'test/target', 'test/judge',
        'test/attacker', 'test/target', 'test/judge',
    ]
    # Reformulation call omits system prompt injection and forwards temperature=1.0, max_tokens=150.
    assert len(client.calls[0]['messages']) == 1 and client.calls[0]['messages'][0]['role'] == 'user'
    assert client.calls[0]['temperature'] == 1.0 and client.calls[0]['max_tokens'] == 150
    # Quote stripping matches JBF's response.replace('"', '').
    record = result['records'][0]
    assert record['jailbroken'] is True
    assert len(record['attempts']) == 2
    assert record['prompt'] == 'How did people build widgets in the past?'


def test_past_tense_exhausts_n_restarts_without_success():
    client = ScriptedClient([
        'Attempt 1?', 'Refusal 1', 'Rating: [[1]]',
        'Attempt 2?', 'Refusal 2', 'Rating: [[2]]',
    ])
    dataset = [[{'role': 'user', 'content': 'Build a widget'}, {'role': 'assistant', 'content': 'Reference.'}]]
    result = CORE.execute(CORE.driver, MODEL, config(n_restarts=2), dataset, client=client, sleep=lambda _: None)

    record = result['records'][0]
    assert record['jailbroken'] is False
    assert len(record['attempts']) == 2
    assert record['prompt'] == 'Attempt 2?'


def test_past_tense_config_validation_and_hydra_defaults():
    cfg = config()
    CORE.validate_config(MODEL, cfg)
    with pytest.raises(ValueError, match='explicit OpenRouter'):
        CORE.validate_config(MODEL, {**cfg, 'judge': {'id': None, 'provider': None}})
    with pytest.raises(ValueError, match='n_restarts'):
        CORE.validate_config(MODEL, {**cfg, 'n_restarts': 0})
    with pytest.raises(ValueError, match='source_profile'):
        CORE.validate_config(MODEL, {**cfg, 'source_profile': 'gpt-4o'})

    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base='1.3'):
        composed = compose(config_name='config', overrides=[
            'hydra/launcher=basic', 'model=openrouter', 'attack=past_tense',
            'models.openrouter.id=test/target', 'models.openrouter.provider=target-provider',
            'attacks.past_tense.attacker.id=test/attacker', 'attacks.past_tense.attacker.provider=attacker-provider',
            'attacks.past_tense.judge.id=test/judge', 'attacks.past_tense.judge.provider=judge-provider',
        ])
    assert composed.attacks.past_tense.name == 'past_tense'
    assert composed.attacks.past_tense.n_restarts == 20
    assert composed.attacks.past_tense.seed is None
