"""AIM (Alphabet Index Mapping) hosted integration and unit tests, without live model calls."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import pytest

from tests.jbf_test_utils import load_jbf_core


ROOT = Path(__file__).resolve().parents[2]
CORE = load_jbf_core('aim', 'AIMConfig', 'AIMAttack')
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
        if isinstance(reply, Exception):
            raise reply
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2,
                                completion_tokens_details=None, model_dump=lambda: {'total_tokens': 2})
        return SimpleNamespace(id='test', model=kwargs['model'], usage=usage,
                               choices=[SimpleNamespace(message=SimpleNamespace(content=reply), finish_reason='stop')])


@pytest.mark.parametrize('mode,expected_encoded', [
    ('AIM', '1:2: 3:4:'),
    ('AIM+FWO', '3:4: 1:2:'),
    ('AIM+FCW', '2:1: 4:3:'),
])
def test_aim_modes_and_system_user_message_splitting(mode, expected_encoded):
    client = ScriptedClient(['  Decoded reply.\n'])
    dataset = [[{'role': 'user', 'content': 'ab cd'}, {'role': 'assistant', 'content': 'Reference.'}]]
    cfg = {**asdict(CORE.config()), 'mode': mode}
    result = CORE.execute(CORE.driver, MODEL, cfg, dataset, client=client, sleep=lambda _: None)

    assert len(client.calls) == 1
    assert client.options == [{'max_retries': 0}]
    messages = client.calls[0]['messages']
    assert messages[0]['role'] == 'system'
    assert messages[1] == {'role': 'user', 'content': expected_encoded}
    assert result['records'][0]['mode'] == mode
    assert result['records'][0]['response'] == '  Decoded reply.\n'


def test_aim_config_validation_and_hydra_defaults():
    cfg = asdict(CORE.config())
    CORE.validate_config(MODEL, cfg)
    with pytest.raises(ValueError, match='mode'):
        CORE.validate_config(MODEL, {**cfg, 'mode': 'INVALID'})
    with pytest.raises(ValueError, match='source_profile'):
        CORE.validate_config(MODEL, {**cfg, 'source_profile': 'gpt-3.5-turbo'})
    with pytest.raises(ValueError, match='generation_config'):
        CORE.validate_config(MODEL, {**cfg, 'generation_config': {}})

    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base='1.3'):
        composed = compose(config_name='config', overrides=[
            'hydra/launcher=basic', 'model=openrouter', 'attack=aim',
            'models.openrouter.id=test/target', 'models.openrouter.provider=target-provider',
        ])
    assert composed.attacks.aim.name == 'aim'
    assert composed.attacks.aim.mode == 'AIM'
    assert composed.attacks.aim.seed is None
