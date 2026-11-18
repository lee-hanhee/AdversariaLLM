"""JailExpert hosted integration and unit tests, without live model calls."""

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import pytest

from tests.jbf_test_utils import load_jbf_core


ROOT = Path(__file__).resolve().parents[2]
CORE = load_jbf_core('jailexpert', 'JailExpertConfig', 'JailExpertAttack')
MODEL = {'backend': 'openrouter', 'id': 'test/target', 'provider': 'target-provider'}


def config(**changes):
    return {
        **asdict(CORE.config()),
        'attacker': {'id': 'test/attacker', 'provider': 'attacker-provider'},
        'embedding': {'id': 'test/embedding', 'provider': 'embedding-provider'},
        **changes,
    }


class ScriptedClient:
    base_url = 'https://openrouter.ai/api/v1'

    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []
        self.embedding_calls = []
        self.options = []
        self.chat = SimpleNamespace(completions=self)
        self.embeddings = SimpleNamespace(create=self.create_embedding)

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

    def create_embedding(self, **kwargs):
        self.embedding_calls.append(kwargs)
        return SimpleNamespace(
            id='emb-test',
            model=kwargs['model'],
            usage=SimpleNamespace(model_dump=lambda: {'total_tokens': 2}),
            data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3, 0.4])],
        )


def test_jailexpert_embedding_route_mutation_and_target_execution(tmp_path):
    exp_file = tmp_path / 'exp.json'
    exp_file.write_text(json.dumps([
        {'mutation': 'rephrase', 'method': '{query}', 'pre_query': 'a', 'full_query': 'a', 'harmfulness_score': 4},
        {'mutation': 'roleplay', 'method': 'In a story: {query}', 'pre_query': 'b', 'full_query': 'b', 'harmfulness_score': 5},
    ]))
    client = ScriptedClient([
        'Mutated adversarial prompt.',
        'Target JailExpert response.',
    ])
    dataset = [[{'role': 'user', 'content': 'Build a widget'}, {'role': 'assistant', 'content': 'Reference.'}]]
    result = CORE.execute(
        CORE.driver, MODEL, config(experience_path=str(exp_file), n_clusters=1), dataset,
        client=client, sleep=lambda _: None,
    )

    assert len(client.embedding_calls) >= 2
    assert all(c['model'] == 'test/embedding' for c in client.embedding_calls)
    assert client.calls[-1]['model'] == 'test/target'
    assert result['records'][0]['response'] == 'Target JailExpert response.'


def test_jailexpert_config_validation_and_hydra_defaults():
    cfg = config()
    CORE.validate_config(MODEL, cfg)
    with pytest.raises(ValueError, match='explicit OpenRouter'):
        CORE.validate_config(MODEL, {**cfg, 'embedding': {'id': None, 'provider': None}})
    with pytest.raises(ValueError, match='n_clusters'):
        CORE.validate_config(MODEL, {**cfg, 'n_clusters': 0})

    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base='1.3'):
        composed = compose(config_name='config', overrides=[
            'hydra/launcher=basic', 'model=openrouter', 'attack=jailexpert',
            'models.openrouter.id=test/target', 'models.openrouter.provider=target-provider',
            'attacks.jailexpert.attacker.id=test/attacker', 'attacks.jailexpert.attacker.provider=attacker-provider',
            'attacks.jailexpert.embedding.id=test/embedding', 'attacks.jailexpert.embedding.provider=embedding-provider',
        ])
    assert composed.attacks.jailexpert.name == 'jailexpert'
    assert composed.attacks.jailexpert.n_clusters == 10
    assert composed.attacks.jailexpert.seed is None
