"""BPJ (Boundary Point Jailbreaking) hosted integration and unit tests, without live model calls."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import pytest

from tests.jbf_test_utils import load_jbf_core


ROOT = Path(__file__).resolve().parents[2]
CORE = load_jbf_core('bpj', 'BPJConfig', 'BPJAttack')
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


def test_bpj_target_only_boundary_search():
    # Supply enough benign/non-refusal replies for small budget run.
    client = ScriptedClient(['Sure, here is the response.'] * 30)
    dataset = [[{'role': 'user', 'content': 'Build a widget'}, {'role': 'assistant', 'content': 'Reference.'}]]
    cfg = {**asdict(CORE.config()), 'population_size': 2, 'bp_target_size': 2, 'eval_sample_size': 1, 'query_budget': 4}
    result = CORE.execute(CORE.driver, MODEL, cfg, dataset, client=client, sleep=lambda _: None)

    assert len(client.calls) >= 1
    assert all(c['model'] == 'test/target' for c in client.calls)
    assert result['records'][0]['response'] == 'Sure, here is the response.'


def test_bpj_config_validation_and_hydra_defaults():
    cfg = asdict(CORE.config())
    CORE.validate_config(MODEL, cfg)
    with pytest.raises(ValueError, match='population_size'):
        CORE.validate_config(MODEL, {**cfg, 'population_size': 0})

    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base='1.3'):
        composed = compose(config_name='config', overrides=[
            'hydra/launcher=basic', 'model=openrouter', 'attack=bpj',
            'models.openrouter.id=test/target', 'models.openrouter.provider=target-provider',
        ])
    assert composed.attacks.bpj.name == 'bpj'
    assert composed.attacks.bpj.seed is None
