"""Hosted binding for Jailbreak Foundry's Intent Laundering implementation."""

from dataclasses import dataclass, field
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, Optional

from .attack import Attack
from .utils.author_runtime import load_symbols, normalize_config, run_attack, validate_routes
from .utils.jbf_runtime import make_jbf_query_client, run_jbf_universal_attempts


@dataclass
class IntentLaunderingConfig:
    name: str = 'intent_laundering'
    type: str = 'discrete'
    version: str = '0.1.0'
    seed: None = None
    # JBF launderer default label; hosted OpenRouter routes are configured separately.
    source_profile: str = 'gpt-4o'
    attempts_per_query: int = 1
    attacker: dict = field(default_factory=lambda: {'id': None, 'provider': None})


class IntentLaunderingAttack(Attack):
    """Run the preserved JBF Intent Laundering generator with explicit hosted model routes."""

    REQUIRES_HOSTED_TARGET = True

    def __init__(self, config):
        self.config = normalize_config(config)

    def run(self, target, dataset):
        return run_attack(self, target, dataset)

    @staticmethod
    def validate_config(model, config):
        validate_routes(model, config, ('attacker',))
        if config.get('source_profile') != 'gpt-4o':
            raise ValueError('IntentLaundering source_profile must retain gpt-4o.')
        if type(config.get('attempts_per_query', 1)) is not int or config.get('attempts_per_query', 1) < 1:
            raise ValueError('IntentLaundering attempts_per_query must be an integer >= 1.')
        if 'generation_config' in config:
            raise ValueError('IntentLaundering retains JBF generation settings.')

    @staticmethod
    def _add_steps(run, record, trace, duration):
        target_requests = [item for item in trace if item['role'] == 'target' and 'metadata' in item]
        for index, request in enumerate(target_requests):
            is_last = index == len(target_requests) - 1
            raw = request.get('raw_text')
            comp = record['response'] if is_last else (raw or '')
            run.add_step(
                request,
                messages=request['request']['messages'],
                completions=[comp],
                raw_completions=[] if raw is None else [raw],
                time_taken=duration if is_last else 0.0,
                author_result=record if is_last else {},
                author_requests=trace,
            )

    @staticmethod
    def _execute(transport, model, config, questions, sleep=None):
        transport.routes['attacker'] = config['attacker']
        transport.client = transport.client.with_options(max_retries=0)
        helper = make_jbf_query_client(transport, 'attacker', config['source_profile'], sleep or time.sleep)
        target = make_jbf_query_client(transport, 'target', config['source_profile'], sleep or time.sleep)

        il_cfg = {**config, 'launderer_model': {'model': config['source_profile'], 'provider': 'openai'}}

        class _ModernBase:
            def __init__(self, args=None, **kwargs):
                self.args = args
                self.get_parameter_value = il_cfg.__getitem__

        source = load_symbols(
            'intent_laundering/intent_laundering_pipeline.py',
            ['FIRST_REVISION_PROMPT', 'SUBSEQUENT_REVISION_PROMPT', 'IntentLaunderingGen'],
            {
                'ModernBaseAttack': _ModernBase, 'AttackParameter': lambda **kwargs: kwargs,
                'LLMLiteLLM': SimpleNamespace(from_config=lambda **_kw: helper),
                'threading': threading, 'Dict': Dict, 'Any': Any, 'Optional': Optional,
            },
        )['IntentLaunderingGen']

        author = source()
        records = []
        for idx, (question, reference) in enumerate(transport.examples(questions)):
            attempts = run_jbf_universal_attempts(
                author, target, question, reference,
                attempts_per_query=config.get('attempts_per_query', 1),
                example_id=idx,
            )
            records.append({
                'question': question,
                'prompt': attempts[-1]['prompt'],
                'response': attempts[-1]['response'],
                'attempts': attempts,
            })
        return records, {}
