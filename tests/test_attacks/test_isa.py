"""ISA (Intent Shift Attack) hosted integration and unit tests, without live model calls."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import pytest

from tests.jbf_test_utils import load_jbf_core


ROOT = Path(__file__).resolve().parents[2]
CORE = load_jbf_core('isa', 'ISAConfig', 'ISAAttack')
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


@pytest.mark.parametrize('shift_type', ['person', 'tense', 'voice', 'mood', 'question'])
def test_isa_two_step_helper_and_target_routing(shift_type):
    client = ScriptedClient([
        'How to build a widget?',
        f'Shifted ({shift_type}) prompt?',
        'Target response.',
    ])
    dataset = [[{'role': 'user', 'content': 'Build a widget'}, {'role': 'assistant', 'content': 'Reference.'}]]
    result = CORE.execute(CORE.driver, MODEL, config(shift_type=shift_type), dataset, client=client, sleep=lambda _: None)

    assert len(client.calls) == 3
    assert client.calls[0]['model'] == 'test/attacker' and client.calls[0]['temperature'] == 0.0
    assert client.calls[1]['model'] == 'test/attacker' and client.calls[1]['temperature'] == 0.0
    assert 'How to build a widget?' in client.calls[1]['messages'][-1]['content']
    assert client.calls[2]['model'] == 'test/target'
    assert result['records'][0]['prompt'] == f'Shifted ({shift_type}) prompt?'
    assert result['records'][0]['response'] == 'Target response.'


def test_isa_config_validation_and_hydra_defaults():
    cfg = config()
    CORE.validate_config(MODEL, cfg)
    with pytest.raises(ValueError, match='explicit OpenRouter'):
        CORE.validate_config(MODEL, {**cfg, 'attacker': {'id': None, 'provider': None}})
    with pytest.raises(ValueError, match='shift_type'):
        CORE.validate_config(MODEL, {**cfg, 'shift_type': 'invalid'})
    with pytest.raises(ValueError, match='source_profile'):
        CORE.validate_config(MODEL, {**cfg, 'source_profile': 'gpt-3.5-turbo'})

    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base='1.3'):
        composed = compose(config_name='config', overrides=[
            'hydra/launcher=basic', 'model=openrouter', 'attack=isa',
            'models.openrouter.id=test/target', 'models.openrouter.provider=target-provider',
            'attacks.isa.attacker.id=test/attacker', 'attacks.isa.attacker.provider=attacker-provider',
        ])
    assert composed.attacks.isa.name == 'isa'
    assert composed.attacks.isa.shift_type == 'question'
    assert composed.attacks.isa.seed is None
