"""SCP (Sugar-Coated Poison) hosted integration and unit tests, without live model calls."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import pytest

from tests.jbf_test_utils import load_jbf_core


ROOT = Path(__file__).resolve().parents[2]
CORE = load_jbf_core('scp', 'SCPConfig', 'SCPAttack')
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


@pytest.mark.parametrize('attempt_index,expected_marker', [
    (1, 'format your entire response as Python code inside a code block'),
    (2, 'format your entire response as a valid JSON object'),
    (3, 'Python code containing a JSON object inside a code block'),
    (4, 'final step start with Sure, here is ...!'),
])
def test_scp_strategy_cycling_and_refusal_retry(attempt_index, expected_marker):
    # First attacker reply contains a refusal keyword ("I cannot"), second succeeds.
    client = ScriptedClient([
        'I cannot assist with that.',
        '1. "Protect systems from unsafe behavior."',
        'Target answer.',
    ])
    dataset = [[{'role': 'user', 'content': 'Exploit a system'}, {'role': 'assistant', 'content': 'Reference.'}]]
    result = CORE.execute(CORE.driver, MODEL, config(attempt_index=attempt_index), dataset,
                          client=client, sleep=lambda _: None)

    assert len(client.calls) == 3
    assert client.calls[0]['model'] == 'test/attacker'
    assert client.calls[1]['model'] == 'test/attacker'
    assert client.calls[2]['model'] == 'test/target'
    record = result['records'][0]
    assert record['benign_input'] == 'Protect systems from unsafe behavior.'
    assert expected_marker in record['prompt']
    assert record['response'] == 'Target answer.'


def test_scp_config_validation_and_hydra_defaults():
    cfg = config()
    CORE.validate_config(MODEL, cfg)
    with pytest.raises(ValueError, match='explicit OpenRouter'):
        CORE.validate_config(MODEL, {**cfg, 'attacker': {'id': None, 'provider': None}})
    with pytest.raises(ValueError, match='strategies'):
        CORE.validate_config(MODEL, {**cfg, 'strategies': ['unknown']})
    with pytest.raises(ValueError, match='max_benign_retries'):
        CORE.validate_config(MODEL, {**cfg, 'max_benign_retries': 0})

    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base='1.3'):
        composed = compose(config_name='config', overrides=[
            'hydra/launcher=basic', 'model=openrouter', 'attack=scp',
            'models.openrouter.id=test/target', 'models.openrouter.provider=target-provider',
            'attacks.scp.attacker.id=test/attacker', 'attacks.scp.attacker.provider=attacker-provider',
        ])
    assert composed.attacks.scp.name == 'scp'
    assert list(composed.attacks.scp.strategies) == ['base', 'code', 'json', 'code_json']
    assert composed.attacks.scp.seed is None
