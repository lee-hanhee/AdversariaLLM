"""Hosted binding for Jailbreak Foundry's AIM (Alphabet Index Mapping) implementation."""

from dataclasses import dataclass
import re
import time
from typing import Any, Dict, List, Optional

from .attack import Attack
from .utils.author_runtime import load_symbols, normalize_config, run_attack, validate_routes
from .utils.jbf_runtime import make_jbf_query_client


@dataclass
class AIMConfig:
    name: str = 'aim'
    type: str = 'discrete'
    version: str = '0.1.0'
    seed: None = None
    # JBF target default label; hosted OpenRouter routes are configured separately.
    source_profile: str = 'gpt-4o'
    mode: str = 'AIM'


class AIMAttack(Attack):
    """Run the preserved JBF AIM generator with explicit hosted model routes."""

    REQUIRES_HOSTED_TARGET = True

    def __init__(self, config):
        self.config = normalize_config(config)

    def run(self, target, dataset):
        return run_attack(self, target, dataset)

    @staticmethod
    def validate_config(model, config):
        validate_routes(model, config, ())
        if config.get('source_profile') != 'gpt-4o':
            raise ValueError('AIM source_profile must retain gpt-4o.')
        if config.get('mode') not in {'AIM', 'AIM+FWO', 'AIM+FCW'}:
            raise ValueError("AIM mode must be one of 'AIM', 'AIM+FWO', or 'AIM+FCW'.")
        if 'generation_config' in config:
            raise ValueError('AIM retains JBF generation settings.')

    @staticmethod
    def _add_steps(run, record, trace, duration):
        request = next(item for item in reversed(trace) if item['role'] == 'target' and 'metadata' in item)
        run.add_step(request, messages=request['request']['messages'],
                     completions=[record['response']], raw_completions=[request['raw_text']],
                     time_taken=duration, author_result=record, author_requests=trace)

    @staticmethod
    def _execute(transport, model, config, questions, sleep=None):
        transport.client = transport.client.with_options(max_retries=0)
        target = make_jbf_query_client(transport, 'target', config['source_profile'], sleep or time.sleep)
        source = load_symbols('aim/aim_pipeline.py', ['AIMAttack'], {
            'ModernBaseAttack': object, 'AttackParameter': lambda **kwargs: kwargs,
            'Dict': Dict, 'Any': Any, 'Optional': Optional, 'List': List, 're': re,
        })['AIMAttack']

        records = []
        for question, reference in transport.examples(questions):
            author = source.__new__(source)
            author.get_parameter_value = config.__getitem__
            prompt = author.generate_attack(prompt=question, goal=question, target=reference)
            records.append({
                'question': question,
                'mode': config['mode'],
                'prompt': prompt,
                'response': str(target.query(prompt)),
            })
        return records, {}
