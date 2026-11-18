"""ReNeLLM hosted integration and unit tests, without live model calls."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import pytest

from tests.jbf_test_utils import load_jbf_core


ROOT = Path(__file__).resolve().parents[2]
CORE = load_jbf_core('renellm', 'ReNeLLMConfig', 'ReNeLLMAttack')
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


def test_renellm_rewrite_judge_and_target_execution():
    client = ScriptedClient([
        'Rewritten harmful task.',
        '1',
        'Target ReNeLLM answer.',
    ])
    dataset = [[{'role': 'user', 'content': 'Build a widget'}, {'role': 'assistant', 'content': 'Reference.'}]]
    result = CORE.execute(CORE.driver, MODEL, config(), dataset, client=client, sleep=lambda _: None)

    assert [c['model'] for c in client.calls] == ['test/attacker', 'test/judge', 'test/target']
    assert 'Rewritten harmful task.' in result['records'][0]['prompt']
    assert result['records'][0]['response'] == 'Target ReNeLLM answer.'


def test_renellm_config_validation_and_hydra_defaults():
    cfg = config()
    CORE.validate_config(MODEL, cfg)
    with pytest.raises(ValueError, match='explicit OpenRouter'):
        CORE.validate_config(MODEL, {**cfg, 'judge': {'id': None, 'provider': None}})
    with pytest.raises(ValueError, match='max_iterations'):
        CORE.validate_config(MODEL, {**cfg, 'max_iterations': 0})

    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base='1.3'):
        composed = compose(config_name='config', overrides=[
            'hydra/launcher=basic', 'model=openrouter', 'attack=renellm',
            'models.openrouter.id=test/target', 'models.openrouter.provider=target-provider',
            'attacks.renellm.attacker.id=test/attacker', 'attacks.renellm.attacker.provider=attacker-provider',
            'attacks.renellm.judge.id=test/judge', 'attacks.renellm.judge.provider=judge-provider',
        ])
    assert composed.attacks.renellm.name == 'renellm'
    assert composed.attacks.renellm.max_iterations == 5
    assert composed.attacks.renellm.seed is None
