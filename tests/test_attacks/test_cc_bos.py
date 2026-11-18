"""CC-BOS hosted integration and unit tests, without live model calls."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import pytest

from tests.jbf_test_utils import load_jbf_core


ROOT = Path(__file__).resolve().parents[2]
CORE = load_jbf_core('cc_bos', 'CCBOSConfig', 'CCBOSAttack')
MODEL = {'backend': 'openrouter', 'id': 'test/target', 'provider': 'target-provider'}


def config(**changes):
    return {
        **asdict(CORE.config()),
        'attacker': {'id': 'test/attacker', 'provider': 'attacker-provider'},
        'judge': {'id': 'test/judge', 'provider': 'judge-provider'},
        **changes,
    }


class ScriptedClient:
    base_url = 'https://openrouter.ai/api/v1'

    def __init__(self, replies_by_model):
        self.replies_by_model = {k: iter(v) for k, v in replies_by_model.items()}
        self.calls = []
        self.options = []
        self.chat = SimpleNamespace(completions=self)

    def with_options(self, **options):
        self.options.append(options)
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = next(self.replies_by_model[kwargs['model']])
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2,
                                completion_tokens_details=None, model_dump=lambda: {'total_tokens': 2})
        return SimpleNamespace(id='test', model=kwargs['model'], usage=usage,
                               choices=[SimpleNamespace(message=SimpleNamespace(content=reply), finish_reason='stop')])


def test_cc_bos_foa_routing_and_execution():
    client = ScriptedClient({
        'test/attacker': ['文言文提示词。', 'Translated English response.'] * 10,
        'test/target': ['Sure, here is the detailed answer.'] * 10,
        'test/judge': ['10'] * 10,
    })
    dataset = [[{'role': 'user', 'content': 'Build a widget'}, {'role': 'assistant', 'content': 'Reference.'}]]
    result = CORE.execute(
        CORE.driver, MODEL, config(population_size=1, max_iterations=1), dataset,
        client=client, sleep=lambda _: None,
    )

    models_called = {c['model'] for c in client.calls}
    assert {'test/attacker', 'test/target', 'test/judge'} <= models_called
    assert result['records'][0]['response'] == 'Sure, here is the detailed answer.'


def test_cc_bos_config_validation_and_hydra_defaults():
    cfg = config()
    CORE.validate_config(MODEL, cfg)
    with pytest.raises(ValueError, match='explicit OpenRouter'):
        CORE.validate_config(MODEL, {**cfg, 'judge': {'id': None, 'provider': None}})
    with pytest.raises(ValueError, match='population_size'):
        CORE.validate_config(MODEL, {**cfg, 'population_size': 0})

    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base='1.3'):
        composed = compose(config_name='config', overrides=[
            'hydra/launcher=basic', 'model=openrouter', 'attack=cc_bos',
            'models.openrouter.id=test/target', 'models.openrouter.provider=target-provider',
            'attacks.cc_bos.attacker.id=test/attacker', 'attacks.cc_bos.attacker.provider=attacker-provider',
            'attacks.cc_bos.judge.id=test/judge', 'attacks.cc_bos.judge.provider=judge-provider',
        ])
    assert composed.attacks.cc_bos.name == 'cc_bos'
    assert composed.attacks.cc_bos.seed is None
