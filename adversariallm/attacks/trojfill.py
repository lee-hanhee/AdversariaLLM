"""Hosted binding for Jailbreak Foundry's TrojFill implementation."""

import base64
from dataclasses import dataclass, field
import logging
import random
import re
import threading
import time
from typing import Any, Dict, List, Optional

from .attack import Attack
from .utils.author_runtime import load_symbols, normalize_config, run_attack, validate_routes
from .utils.jbf_runtime import make_jbf_query_client


@dataclass
class TrojFillConfig:
    name: str = 'trojfill'
    type: str = 'discrete'
    version: str = '0.1.0'
    seed: None = None
    # JBF attacker default label; hosted OpenRouter routes are configured separately.
    source_profile: str = 'gemini-2.5-flash'
    transform_type: str = 'Caesar'
    internal_retries: int = 3
    attacker: dict = field(default_factory=lambda: {'id': None, 'provider': None})


class TrojFillAttack(Attack):
    """Run the preserved JBF TrojFill generator with explicit hosted model routes."""

    REQUIRES_HOSTED_TARGET = True

    def __init__(self, config):
        self.config = normalize_config(config)

    def run(self, target, dataset):
        return run_attack(self, target, dataset)

    @staticmethod
    def validate_config(model, config):
        validate_routes(model, config, ('attacker',))
        if config.get('source_profile') != 'gemini-2.5-flash':
            raise ValueError('TrojFill source_profile must retain gemini-2.5-flash.')
        if config.get('transform_type') not in {'Caesar', 'Base64', 'splitting'}:
            raise ValueError("TrojFill transform_type must be 'Caesar', 'Base64', or 'splitting'.")
        if type(config.get('internal_retries')) is not int or config['internal_retries'] < 1:
            raise ValueError('TrojFill internal_retries must be an integer >= 1.')
        if 'generation_config' in config:
            raise ValueError('TrojFill retains JBF generation settings.')

    @staticmethod
    def _add_steps(run, record, trace, duration):
        request = next(item for item in reversed(trace) if item['role'] == 'target' and 'metadata' in item)
        run.add_step(request, messages=request['request']['messages'],
                     completions=[record['response']], raw_completions=[request['raw_text']],
                     time_taken=duration, author_result=record, author_requests=trace)

    @staticmethod
    def _execute(transport, model, config, questions, sleep=None):
        transport.routes['attacker'] = config['attacker']
        transport.client = transport.client.with_options(max_retries=0)
        helper = make_jbf_query_client(transport, 'attacker', config['source_profile'], sleep or time.sleep)
        target = make_jbf_query_client(transport, 'target', config['source_profile'], sleep or time.sleep)

        source = load_symbols('trojfill/trojfill_pipeline.py', ['TrojFillGen'], {
            'ModernBaseAttack': object, 'AttackParameter': lambda **kwargs: kwargs, 'LLMLiteLLM': object,
            'Dict': Dict, 'Any': Any, 'Optional': Optional, 'List': List,
            're': re, 'random': random, 'base64': base64, 'threading': threading,
            'logger': logging.getLogger(__name__),
        })['TrojFillGen']

        history = {}
        lock = threading.Lock()
        records = []
        for question, reference in transport.examples(questions):
            author = source.__new__(source)
            author.get_parameter_value = config.__getitem__
            author.attacker_llm = helper
            author.history = history
            author.lock = lock
            prompt = author.generate_attack(prompt=question, goal=question, target=reference)
            records.append({
                'question': question,
                'transform_type': config['transform_type'],
                'prompt': prompt,
                'response': str(target.query(prompt)),
            })
        return records, {}
