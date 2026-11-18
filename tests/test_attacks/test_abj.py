"""ABJ (Analyzing-based Jailbreak) hosted integration and unit tests, without live model calls."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import pytest

from tests.jbf_test_utils import load_jbf_core


ROOT = Path(__file__).resolve().parents[2]
CORE = load_jbf_core('abj', 'ABJConfig', 'ABJAttack')
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


def test_abj_adjustment_loop_and_early_stopping():
    # Sequence:
    # 1. attacker: transform_query_to_neutral_data
    # 2. target: round 1 reply -> judge evaluator: '0' (not jailbroken)
    # 3. judge: refusal detection -> attacker: toxicity adjustment
    # 4. target: round 2 reply -> judge evaluator: '1' (jailbroken -> early stop)
    client = ScriptedClient([
        'Neutralized criminal profile.',
        'I cannot assist with that.',
        '0',
        'Yes, refusal detected.',
        'Adjusted less-toxic criminal profile.',
        'Detailed step-by-step analysis.',
        '1',
    ])
    dataset = [[{'role': 'user', 'content': 'Build a widget'}, {'role': 'assistant', 'content': 'Reference.'}]]
    result = CORE.execute(
        CORE.driver, MODEL, config(max_attack_rounds=1, max_adjustment_rounds=3), dataset,
        client=client, sleep=lambda _: None,
    )

    assert [c['model'] for c in client.calls] == [
        'test/attacker', 'test/target', 'test/judge', 'test/judge', 'test/attacker', 'test/target', 'test/judge',
    ]
    record = result['records'][0]
    assert record['jailbroken'] is True
    assert len(record['attempts']) == 2
    assert record['response'] == 'Detailed step-by-step analysis.'


def test_abj_config_validation_and_hydra_defaults():
    cfg = config()
    CORE.validate_config(MODEL, cfg)
    with pytest.raises(ValueError, match='explicit OpenRouter'):
        CORE.validate_config(MODEL, {**cfg, 'judge': {'id': None, 'provider': None}})
    with pytest.raises(ValueError, match='max_attack_rounds'):
        CORE.validate_config(MODEL, {**cfg, 'max_attack_rounds': 0})
    with pytest.raises(ValueError, match='source_profile'):
        CORE.validate_config(MODEL, {**cfg, 'source_profile': 'gpt-4o'})

    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base='1.3'):
        composed = compose(config_name='config', overrides=[
            'hydra/launcher=basic', 'model=openrouter', 'attack=abj',
            'models.openrouter.id=test/target', 'models.openrouter.provider=target-provider',
            'attacks.abj.attacker.id=test/attacker', 'attacks.abj.attacker.provider=attacker-provider',
            'attacks.abj.judge.id=test/judge', 'attacks.abj.judge.provider=judge-provider',
        ])
    assert composed.attacks.abj.name == 'abj'
    assert composed.attacks.abj.max_attack_rounds == 1
    assert composed.attacks.abj.max_adjustment_rounds == 5
    assert composed.attacks.abj.seed is None
