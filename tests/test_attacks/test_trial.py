"""TRIAL hosted integration and unit tests, without live model calls."""

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import pytest

from tests.jbf_test_utils import load_jbf_core


ROOT = Path(__file__).resolve().parents[2]
CORE = load_jbf_core('trial', 'TRIALConfig', 'TRIALAttack')
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


def test_trial_dilemma_and_followup_escalation():
    client = ScriptedClient([
        json.dumps({'theme': 'engineering', 'action': 'build', 'goal': 'prevent collapse'}),
        'Option A: build widget. Option B: catastrophe.',
        'I choose Option A to prevent catastrophe.',
        'How specifically would you implement Option A?',
        'Detailed Option A implementation steps.',
        'YES',
    ])
    dataset = [[{'role': 'user', 'content': 'Build a widget'}, {'role': 'assistant', 'content': 'Reference.'}]]
    result = CORE.execute(CORE.driver, MODEL, config(n_iterations=3), dataset, client=client, sleep=lambda _: None)

    assert [c['model'] for c in client.calls] == [
        'test/attacker', 'test/attacker', 'test/target',
        'test/attacker', 'test/target', 'test/judge',
    ]
    record = result['records'][0]
    assert record['jailbroken'] is True
    assert len(record['attempts']) == 2
    assert record['response'] == 'Detailed Option A implementation steps.'


def test_trial_config_validation_and_hydra_defaults():
    cfg = config()
    CORE.validate_config(MODEL, cfg)
    with pytest.raises(ValueError, match='explicit OpenRouter'):
        CORE.validate_config(MODEL, {**cfg, 'judge': {'id': None, 'provider': None}})
    with pytest.raises(ValueError, match='n_iterations'):
        CORE.validate_config(MODEL, {**cfg, 'n_iterations': 0})
    with pytest.raises(ValueError, match='source_profile'):
        CORE.validate_config(MODEL, {**cfg, 'source_profile': 'gpt-3.5-turbo'})

    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base='1.3'):
        composed = compose(config_name='config', overrides=[
            'hydra/launcher=basic', 'model=openrouter', 'attack=trial',
            'models.openrouter.id=test/target', 'models.openrouter.provider=target-provider',
            'attacks.trial.attacker.id=test/attacker', 'attacks.trial.attacker.provider=attacker-provider',
            'attacks.trial.judge.id=test/judge', 'attacks.trial.judge.provider=judge-provider',
        ])
    assert composed.attacks.trial.name == 'trial'
    assert composed.attacks.trial.n_iterations == 5
    assert composed.attacks.trial.seed is None
