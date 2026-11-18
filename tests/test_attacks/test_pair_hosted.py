"""PAIR Hosted integration and unit tests, without live model calls."""

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import pytest

from tests.jbf_test_utils import load_jbf_core


ROOT = Path(__file__).resolve().parents[2]
CORE = load_jbf_core('pair_hosted', 'PAIRHostedConfig', 'PAIRHostedAttack')
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


def test_pair_hosted_refinement_and_early_stopping():
    # n_streams=1, n_iterations=3:
    # Iteration 1: attacker JSON -> target -> judge Rating: [[3]]
    # Iteration 2: attacker JSON -> target -> judge Rating: [[10]] -> stops early!
    client = ScriptedClient([
        json.dumps({'improvement': 'Try roleplay', 'prompt': 'First attempt prompt'}),
        'I cannot help with that.',
        'Rating: [[3]]',
        json.dumps({'improvement': 'Add historical context', 'prompt': 'Winning attempt prompt'}),
        'Sure, here is the detailed answer.',
        'Rating: [[10]]',
    ])
    dataset = [[{'role': 'user', 'content': 'Build a widget'}, {'role': 'assistant', 'content': 'Sure, here is'}]]
    result = CORE.execute(
        CORE.driver, MODEL, config(n_streams=1, n_iterations=3), dataset,
        client=client, sleep=lambda _: None,
    )

    assert [c['model'] for c in client.calls] == [
        'test/attacker', 'test/target', 'test/judge',
        'test/attacker', 'test/target', 'test/judge',
    ]
    record = result['records'][0]
    assert record['prompt'] == 'Winning attempt prompt'
    assert record['best_score'] == 10
    assert len(record['attempts']) == 2
    assert record['response'] == 'Sure, here is the detailed answer.'


def test_pair_hosted_config_validation_and_hydra_defaults():
    cfg = config()
    CORE.validate_config(MODEL, cfg)
    with pytest.raises(ValueError, match='explicit OpenRouter'):
        CORE.validate_config(MODEL, {**cfg, 'attacker': {'id': None, 'provider': None}})
    with pytest.raises(ValueError, match='n_streams'):
        CORE.validate_config(MODEL, {**cfg, 'n_streams': 0})
    with pytest.raises(ValueError, match='source_profile'):
        CORE.validate_config(MODEL, {**cfg, 'source_profile': 'gpt-4o'})

    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base='1.3'):
        composed = compose(config_name='config', overrides=[
            'hydra/launcher=basic', 'model=openrouter', 'attack=pair_hosted',
            'models.openrouter.id=test/target', 'models.openrouter.provider=target-provider',
            'attacks.pair_hosted.attacker.id=test/attacker', 'attacks.pair_hosted.attacker.provider=attacker-provider',
            'attacks.pair_hosted.judge.id=test/judge', 'attacks.pair_hosted.judge.provider=judge-provider',
        ])
    assert composed.attacks.pair_hosted.name == 'pair_hosted'
    assert composed.attacks.pair_hosted.n_streams == 10
    assert composed.attacks.pair_hosted.n_iterations == 3
    assert composed.attacks.pair_hosted.seed is None
