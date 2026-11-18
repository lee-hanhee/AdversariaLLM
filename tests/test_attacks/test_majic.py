"""MAJIC hosted integration and unit tests, without live model calls."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import pytest

from tests.jbf_test_utils import load_jbf_core


ROOT = Path(__file__).resolve().parents[2]
CORE = load_jbf_core('majic', 'MAJICConfig', 'MAJICAttack')
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


def test_majic_q_learning_loop_and_early_stop():
    # Iteration 1 fails judge ('0'); Iteration 2 succeeds ('1') and stops early.
    client = ScriptedClient([
        'Rewritten prompt 1',
        'I cannot assist.',
        'NO',
        'Rewritten prompt 2',
        'Sure, here is the detailed answer.',
        'YES',
    ])
    dataset = [[{'role': 'user', 'content': 'Build a widget'}, {'role': 'assistant', 'content': 'Reference.'}]]
    result = CORE.execute(CORE.driver, MODEL, config(max_queries=5), dataset, client=client, sleep=lambda _: None)

    assert [c['model'] for c in client.calls] == [
        'test/attacker', 'test/target', 'test/judge',
        'test/attacker', 'test/target', 'test/judge',
    ]
    record = result['records'][0]
    assert record['jailbroken'] is True
    assert len(record['attempts']) == 2
    assert record['prompt'] == 'Rewritten prompt 2'


def test_majic_config_validation_and_hydra_defaults():
    cfg = config()
    CORE.validate_config(MODEL, cfg)
    with pytest.raises(ValueError, match='explicit OpenRouter'):
        CORE.validate_config(MODEL, {**cfg, 'judge': {'id': None, 'provider': None}})
    with pytest.raises(ValueError, match='max_queries'):
        CORE.validate_config(MODEL, {**cfg, 'max_queries': 0})
    with pytest.raises(ValueError, match='source_profile'):
        CORE.validate_config(MODEL, {**cfg, 'source_profile': 'gpt-4o'})

    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base='1.3'):
        composed = compose(config_name='config', overrides=[
            'hydra/launcher=basic', 'model=openrouter', 'attack=majic',
            'models.openrouter.id=test/target', 'models.openrouter.provider=target-provider',
            'attacks.majic.attacker.id=test/attacker', 'attacks.majic.attacker.provider=attacker-provider',
            'attacks.majic.judge.id=test/judge', 'attacks.majic.judge.provider=judge-provider',
        ])
    assert composed.attacks.majic.name == 'majic'
    assert composed.attacks.majic.max_queries == 20
    assert composed.attacks.majic.seed is None
