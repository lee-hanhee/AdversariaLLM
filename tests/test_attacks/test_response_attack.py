"""Response Attack (RA-DRI / RA-SRI) hosted integration and unit tests, without live model calls."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import pytest

from tests.jbf_test_utils import load_jbf_core


ROOT = Path(__file__).resolve().parents[2]
CORE = load_jbf_core('response_attack', 'ResponseAttackConfig', 'ResponseAttack')
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


@pytest.mark.parametrize('attack_type', ['dri', 'sri'])
def test_response_attack_dri_and_sri_message_injection(attack_type):
    client = ScriptedClient([
        'Prompt_1: Harmless framing.\nPrompt_2: Follow-up.',
        'Target injected continuation.',
    ])
    dataset = [[{'role': 'user', 'content': 'Build a widget'}, {'role': 'assistant', 'content': 'Reference.'}]]
    result = CORE.execute(CORE.driver, MODEL, config(attack_type=attack_type), dataset,
                          client=client, sleep=lambda _: None)

    assert [c['model'] for c in client.calls] == ['test/attacker', 'test/target']
    record = result['records'][0]
    assert record['attack_type'] == attack_type
    assert isinstance(record['prompt'], list)
    assert record['response'] == 'Target injected continuation.'


def test_response_attack_config_validation_and_hydra_defaults():
    cfg = config()
    CORE.validate_config(MODEL, cfg)
    with pytest.raises(ValueError, match='explicit OpenRouter'):
        CORE.validate_config(MODEL, {**cfg, 'attacker': {'id': None, 'provider': None}})
    with pytest.raises(ValueError, match='attack_type'):
        CORE.validate_config(MODEL, {**cfg, 'attack_type': 'xyz'})

    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base='1.3'):
        composed = compose(config_name='config', overrides=[
            'hydra/launcher=basic', 'model=openrouter', 'attack=response_attack',
            'models.openrouter.id=test/target', 'models.openrouter.provider=target-provider',
            'attacks.response_attack.attacker.id=test/attacker',
            'attacks.response_attack.attacker.provider=attacker-provider',
        ])
    assert composed.attacks.response_attack.name == 'response_attack'
    assert composed.attacks.response_attack.attack_type == 'dri'
    assert composed.attacks.response_attack.seed is None
