"""TrojFill hosted integration and unit tests, without live model calls."""

import base64
from dataclasses import asdict
from pathlib import Path
import threading
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import pytest

from tests.jbf_test_utils import load_jbf_core


ROOT = Path(__file__).resolve().parents[2]
CORE = load_jbf_core('trojfill', 'TrojFillConfig', 'TrojFillAttack')
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


@pytest.mark.parametrize('transform_type', ['Caesar', 'Base64', 'splitting'])
def test_trojfill_transforms_and_target_execution(transform_type):
    client = ScriptedClient([
        '{"keyword": "widget", "template": "How to build [MASK]?"}',
        'Target TrojFill answer.',
    ])
    dataset = [[{'role': 'user', 'content': 'Build a widget'}, {'role': 'assistant', 'content': 'Reference.'}]]
    result = CORE.execute(CORE.driver, MODEL, config(transform_type=transform_type), dataset,
                          client=client, sleep=lambda _: None)

    assert client.calls[0]['model'] == 'test/attacker'
    assert client.calls[-1]['model'] == 'test/target'
    assert result['records'][0]['transform_type'] == transform_type
    assert result['records'][0]['response'] == 'Target TrojFill answer.'


def test_trojfill_config_validation_and_hydra_defaults():
    cfg = config()
    CORE.validate_config(MODEL, cfg)
    with pytest.raises(ValueError, match='explicit OpenRouter'):
        CORE.validate_config(MODEL, {**cfg, 'attacker': {'id': None, 'provider': None}})
    with pytest.raises(ValueError, match='transform_type'):
        CORE.validate_config(MODEL, {**cfg, 'transform_type': 'ROT13'})

    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base='1.3'):
        composed = compose(config_name='config', overrides=[
            'hydra/launcher=basic', 'model=openrouter', 'attack=trojfill',
            'models.openrouter.id=test/target', 'models.openrouter.provider=target-provider',
            'attacks.trojfill.attacker.id=test/attacker', 'attacks.trojfill.attacker.provider=attacker-provider',
        ])
    assert composed.attacks.trojfill.name == 'trojfill'
    assert composed.attacks.trojfill.transform_type == 'Caesar'
    assert composed.attacks.trojfill.seed is None
