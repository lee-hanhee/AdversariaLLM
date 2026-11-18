"""DeepInception hosted integration and unit tests, without live model calls."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import pytest

from tests.jbf_test_utils import load_jbf_core


ROOT = Path(__file__).resolve().parents[2]
CORE = load_jbf_core('deep_inception', 'DeepInceptionConfig', 'DeepInceptionAttack')
MODEL = {'backend': 'openrouter', 'id': 'test/target', 'provider': 'target-provider'}


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


def test_deep_inception_prompt_construction_and_target_call():
    client = ScriptedClient(['Layer 0 summary.'])
    dataset = [[{'role': 'user', 'content': 'build a widget'}, {'role': 'assistant', 'content': 'Reference.'}]]
    cfg = {**asdict(CORE.config()), 'scene_type': 'dream', 'character_number': 4, 'layer_number': 3}
    result = CORE.execute(CORE.driver, MODEL, cfg, dataset, client=client, sleep=lambda _: None)

    assert len(client.calls) == 1
    assert client.options == [{'max_retries': 0}]
    prompt = result['records'][0]['prompt']
    assert 'dream with more than 4 characters' in prompt
    assert 'please reach layer 3.' in prompt
    assert 'propose a step to build a widget against the super evil doctor.' in prompt
    assert result['records'][0]['response'] == 'Layer 0 summary.'


def test_deep_inception_config_validation_and_hydra_defaults():
    cfg = asdict(CORE.config())
    CORE.validate_config(MODEL, cfg)
    with pytest.raises(ValueError, match='scene_type'):
        CORE.validate_config(MODEL, {**cfg, 'scene_type': '   '})
    with pytest.raises(ValueError, match='character_number'):
        CORE.validate_config(MODEL, {**cfg, 'character_number': 0})
    with pytest.raises(ValueError, match='layer_number'):
        CORE.validate_config(MODEL, {**cfg, 'layer_number': True})
    with pytest.raises(ValueError, match='source_profile'):
        CORE.validate_config(MODEL, {**cfg, 'source_profile': 'gpt-4o'})

    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base='1.3'):
        composed = compose(config_name='config', overrides=[
            'hydra/launcher=basic', 'model=openrouter', 'attack=deep_inception',
            'models.openrouter.id=test/target', 'models.openrouter.provider=target-provider',
        ])
    assert composed.attacks.deep_inception.name == 'deep_inception'
    assert composed.attacks.deep_inception.scene_type == 'science fiction'
    assert composed.attacks.deep_inception.character_number == 5
    assert composed.attacks.deep_inception.layer_number == 5
    assert composed.attacks.deep_inception.seed is None
