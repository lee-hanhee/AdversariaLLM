"""Unit tests for the dual OpenRouter / LiteLLM router and JBF multi-attempt loop."""

from dataclasses import asdict
from types import SimpleNamespace
import pytest

from tests.jbf_test_utils import load_jbf_core


CORE = load_jbf_core('intent_laundering', 'IntentLaunderingConfig', 'IntentLaunderingAttack')


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


def test_openrouter_vs_litellm_routing():
    cfg = {
        **asdict(CORE.config()),
        'attacker': {'id': 'gpt-4o', 'provider': 'openai'},
    }
    dataset = [[{'role': 'user', 'content': 'Build a widget'}, {'role': 'assistant', 'content': 'Reference.'}]]

    # 1. OpenRouter backend pins extra_body.provider.only and uses bare id
    or_client = ScriptedClient(['Revision 1', 'Target 1'])
    or_model = {'backend': 'openrouter', 'id': 'gpt-4o', 'provider': 'openai'}
    CORE.validate_config(or_model, cfg)
    CORE.execute(CORE.driver, or_model, cfg, dataset, client=or_client, sleep=lambda _: None)
    assert or_client.calls[0]['model'] == 'gpt-4o'
    assert or_client.calls[0]['extra_body'] == {
        'provider': {'only': ['openai'], 'allow_fallbacks': False, 'require_parameters': True}
    }

    # 2. LiteLLM backend formats provider/id and omits OpenRouter extra_body
    ll_client = ScriptedClient(['Revision 1', 'Target 1'])
    ll_model = {'backend': 'litellm', 'id': 'gpt-4o', 'provider': 'openai'}
    CORE.validate_config(ll_model, cfg)
    CORE.execute(CORE.driver, ll_model, cfg, dataset, client=ll_client, sleep=lambda _: None)
    assert ll_client.calls[0]['model'] == 'openai/gpt-4o'
    assert 'extra_body' not in ll_client.calls[0]
    assert ll_client.calls[1]['model'] == 'openai/gpt-4o'


def test_intent_laundering_multi_attempt_subsequent_revision():
    cfg = {
        **asdict(CORE.config()),
        'attempts_per_query': 2,
        'attacker': {'id': 'gpt-4o', 'provider': 'openai'},
    }
    dataset = [[{'role': 'user', 'content': 'Build a widget'}, {'role': 'assistant', 'content': 'Reference.'}]]
    client = ScriptedClient([
        'First laundered revision.',
        'I cannot assist with that.',
        'Second laundered revision after failure feedback.',
        'Detailed second response.',
    ])
    model = {'backend': 'openrouter', 'id': 'gpt-4o', 'provider': 'openai'}
    result = CORE.execute(CORE.driver, model, cfg, dataset, client=client, sleep=lambda _: None)

    assert len(client.calls) == 4
    # Second attacker call includes the first failed revision in SUBSEQUENT_REVISION_PROMPT
    assert 'First laundered revision.' in client.calls[2]['messages'][-1]['content']
    record = result['records'][0]
    assert len(record['attempts']) == 2
    assert record['prompt'] == 'Second laundered revision after failure feedback.'
    assert record['response'] == 'Detailed second response.'
