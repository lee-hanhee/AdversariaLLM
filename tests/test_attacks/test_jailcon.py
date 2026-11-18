"""JAIL-CON hosted integration and unit tests, without live model calls."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import pytest

from tests.jbf_test_utils import load_jbf_core


ROOT = Path(__file__).resolve().parents[2]
CORE = load_jbf_core('jailcon', 'JAILCONConfig', 'JAILCONAttack')
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


@pytest.mark.parametrize('mode', ['cvt', 'cit'])
def test_jailcon_modes_and_offline_aux_dataset(mode):
    client = ScriptedClient(['Target interleaved answer.'])
    dataset = [[{'role': 'user', 'content': 'Build a widget'}, {'role': 'assistant', 'content': 'Reference.'}]]
    cfg = {**asdict(CORE.config()), 'mode': mode}
    result = CORE.execute(CORE.driver, MODEL, cfg, dataset, client=client, sleep=lambda _: None)

    assert len(client.calls) == 1
    assert client.calls[0]['model'] == 'test/target'
    assert '{' in result['records'][0]['prompt']
    assert result['records'][0]['response'] == 'Target interleaved answer.'


def test_jailcon_config_validation_and_hydra_defaults():
    cfg = asdict(CORE.config())
    CORE.validate_config(MODEL, cfg)
    with pytest.raises(ValueError, match='mode'):
        CORE.validate_config(MODEL, {**cfg, 'mode': 'invalid'})
    with pytest.raises(ValueError, match='separator'):
        CORE.validate_config(MODEL, {**cfg, 'separator': '{'})

    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base='1.3'):
        composed = compose(config_name='config', overrides=[
            'hydra/launcher=basic', 'model=openrouter', 'attack=jailcon',
            'models.openrouter.id=test/target', 'models.openrouter.provider=target-provider',
        ])
    assert composed.attacks.jailcon.name == 'jailcon'
    assert composed.attacks.jailcon.mode == 'cvt'
    assert composed.attacks.jailcon.seed is None
