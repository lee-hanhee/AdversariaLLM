"""S2C (Structured Semantic Cloaking) hosted integration and unit tests, without live model calls."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import pytest

from tests.jbf_test_utils import load_jbf_core


ROOT = Path(__file__).resolve().parents[2]
CORE = load_jbf_core('s2c', 'S2CConfig', 'S2CAttack')
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


def test_s2c_cloaking_and_target_execution():
    client = ScriptedClient([
        'Urgent engineering scenario requiring a widget.',
        'widget',
        'Target S2C answer.',
    ])
    dataset = [[{'role': 'user', 'content': 'Build a widget'}, {'role': 'assistant', 'content': 'Reference.'}]]
    result = CORE.execute(CORE.driver, MODEL, config(obfuscation_method='base64'), dataset,
                          client=client, sleep=lambda _: None)

    assert client.calls[-1]['model'] == 'test/target'
    assert all(c['model'] == 'test/attacker' for c in client.calls[:-1])
    assert result['records'][0]['response'] == 'Target S2C answer.'


def test_s2c_config_validation_and_hydra_defaults():
    cfg = config()
    CORE.validate_config(MODEL, cfg)
    with pytest.raises(ValueError, match='explicit OpenRouter'):
        CORE.validate_config(MODEL, {**cfg, 'attacker': {'id': None, 'provider': None}})
    with pytest.raises(ValueError, match='obfuscation_method'):
        CORE.validate_config(MODEL, {**cfg, 'obfuscation_method': 'invalid'})

    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base='1.3'):
        composed = compose(config_name='config', overrides=[
            'hydra/launcher=basic', 'model=openrouter', 'attack=s2c',
            'models.openrouter.id=test/target', 'models.openrouter.provider=target-provider',
            'attacks.s2c.attacker.id=test/attacker', 'attacks.s2c.attacker.provider=attacker-provider',
        ])
    assert composed.attacks.s2c.name == 's2c'
    assert composed.attacks.s2c.obfuscation_method == 'random'
    assert composed.attacks.s2c.seed is None
