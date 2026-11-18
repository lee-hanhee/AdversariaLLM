"""RTS-Attack hosted integration and unit tests, without live model calls."""

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import pytest

from tests.jbf_test_utils import load_jbf_core


ROOT = Path(__file__).resolve().parents[2]
CORE = load_jbf_core('rts', 'RTSConfig', 'RTSAttack')
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
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2,
                                completion_tokens_details=None, model_dump=lambda: {'total_tokens': 2})
        return SimpleNamespace(id='test', model=kwargs['model'], usage=usage,
                               choices=[SimpleNamespace(message=SimpleNamespace(content=reply), finish_reason='stop')])


def test_rts_classification_scenario_and_target_execution():
    client = ScriptedClient([
        json.dumps({'type': 1, 'intent': 'build a widget'}),
        'A detailed crime news report about a widget incident.',
        'Natural rewritten instruction.',
        'Target RTS response.',
    ])
    dataset = [[{'role': 'user', 'content': 'How to build a widget'}, {'role': 'assistant', 'content': 'Reference.'}]]
    result = CORE.execute(CORE.driver, MODEL, config(), dataset, client=client, sleep=lambda _: None)

    assert client.calls[-1]['model'] == 'test/target'
    assert all(c['model'] == 'test/attacker' for c in client.calls[:-1])
    assert result['records'][0]['response'] == 'Target RTS response.'


def test_rts_config_validation_and_hydra_defaults():
    cfg = config()
    CORE.validate_config(MODEL, cfg)
    with pytest.raises(ValueError, match='explicit OpenRouter'):
        CORE.validate_config(MODEL, {**cfg, 'attacker': {'id': None, 'provider': None}})
    with pytest.raises(ValueError, match='num_rewrites'):
        CORE.validate_config(MODEL, {**cfg, 'num_rewrites': 0})
    with pytest.raises(ValueError, match='scenario_genre'):
        CORE.validate_config(MODEL, {**cfg, 'scenario_genre': '   '})

    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base='1.3'):
        composed = compose(config_name='config', overrides=[
            'hydra/launcher=basic', 'model=openrouter', 'attack=rts',
            'models.openrouter.id=test/target', 'models.openrouter.provider=target-provider',
            'attacks.rts.attacker.id=test/attacker', 'attacks.rts.attacker.provider=attacker-provider',
        ])
    assert composed.attacks.rts.name == 'rts'
    assert composed.attacks.rts.scenario_genre == 'Crime News Report'
    assert composed.attacks.rts.seed is None
