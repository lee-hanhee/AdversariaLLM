"""HILL hosted integration and unit tests, without live model calls."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import pytest

from tests.jbf_test_utils import load_jbf_core


ROOT = Path(__file__).resolve().parents[2]
CORE = load_jbf_core('hill', 'HILLConfig', 'HILLAttack')
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


def test_hill_reframing_prefix_strip_and_target_routing():
    client = ScriptedClient([
        'HILL: For academic curiosity, how do widgets work?',
        'Target answer.',
    ])
    dataset = [[{'role': 'user', 'content': 'Explain widgets'}, {'role': 'assistant', 'content': 'Reference.'}]]
    result = CORE.execute(CORE.driver, MODEL, config(), dataset, client=client, sleep=lambda _: None)

    assert len(client.calls) == 2
    assert client.calls[0]['model'] == 'test/attacker'
    assert client.calls[1]['model'] == 'test/target'
    assert result['records'][0]['prompt'] == 'For academic curiosity, how do widgets work?'
    assert result['records'][0]['response'] == 'Target answer.'


def test_hill_empty_response_retries_and_fallback():
    # Two empty attacker replies then fallback to original goal on max_attempts=2.
    client = ScriptedClient(['   ', 'HILL:  ', 'Target fallback answer.'])
    dataset = [[{'role': 'user', 'content': 'Original goal'}, {'role': 'assistant', 'content': 'Reference.'}]]
    result = CORE.execute(CORE.driver, MODEL, config(max_attempts=2), dataset, client=client, sleep=lambda _: None)

    assert len(client.calls) == 3
    assert result['records'][0]['prompt'] == 'Original goal'
    assert result['records'][0]['response'] == 'Target fallback answer.'


def test_hill_config_validation_and_hydra_defaults():
    cfg = config()
    CORE.validate_config(MODEL, cfg)
    with pytest.raises(ValueError, match='explicit OpenRouter'):
        CORE.validate_config(MODEL, {**cfg, 'attacker': {'id': None, 'provider': None}})
    with pytest.raises(ValueError, match='max_attempts'):
        CORE.validate_config(MODEL, {**cfg, 'max_attempts': 0})
    with pytest.raises(ValueError, match='intention_check'):
        CORE.validate_config(MODEL, {**cfg, 'intention_check': 'yes'})
    with pytest.raises(ValueError, match='source_profile'):
        CORE.validate_config(MODEL, {**cfg, 'source_profile': 'gpt-3.5-turbo'})

    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base='1.3'):
        composed = compose(config_name='config', overrides=[
            'hydra/launcher=basic', 'model=openrouter', 'attack=hill',
            'models.openrouter.id=test/target', 'models.openrouter.provider=target-provider',
            'attacks.hill.attacker.id=test/attacker', 'attacks.hill.attacker.provider=attacker-provider',
        ])
    assert composed.attacks.hill.name == 'hill'
    assert composed.attacks.hill.max_attempts == 3
    assert composed.attacks.hill.intention_check is True
    assert composed.attacks.hill.seed is None
