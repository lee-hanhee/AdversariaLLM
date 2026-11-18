"""Hosted binding for Jailbreak Foundry's SATA (ELP / MLM) implementation."""

from dataclasses import dataclass, field
import json
import os
import random
import re
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from .attack import Attack
from .utils.author_runtime import (
    AUTHOR_IMPLEMENTATIONS, load_symbols, normalize_config, run_attack, validate_routes,
)
from .utils.jbf_runtime import make_jbf_query_client


@dataclass
class SATAConfig:
    name: str = 'sata'
    type: str = 'discrete'
    version: str = '0.1.0'
    seed: None = None
    # JBF attack helper default label; hosted OpenRouter routes are configured separately.
    source_profile: str = 'gpt-3.5-turbo-1106'
    method: str = 'elp'
    granularity: str = 'sw'
    num_selections: int = 10
    temperature: float = 0.3
    attacker: dict = field(default_factory=lambda: {'id': None, 'provider': None})


class SATAAttack(Attack):
    """Run the preserved JBF SATA generator with explicit hosted model routes."""

    REQUIRES_HOSTED_TARGET = True

    def __init__(self, config):
        self.config = normalize_config(config)

    def run(self, target, dataset):
        return run_attack(self, target, dataset)

    @staticmethod
    def validate_config(model, config):
        validate_routes(model, config, ('attacker',))
        if config.get('source_profile') != 'gpt-3.5-turbo-1106':
            raise ValueError('SATA source_profile must retain gpt-3.5-turbo-1106.')
        if config.get('method') not in {'elp', 'mlm'}:
            raise ValueError("SATA method must be 'elp' or 'mlm'.")
        if config.get('granularity') not in {'sw', 'sp', 'mw', 'mp'}:
            raise ValueError("SATA granularity must be 'sw', 'sp', 'mw', or 'mp'.")
        if type(config.get('num_selections')) is not int or config['num_selections'] < 1:
            raise ValueError('SATA num_selections must be an integer >= 1.')
        temp = config.get('temperature')
        if type(temp) not in (int, float) or temp < 0:
            raise ValueError('SATA temperature must be >= 0.')
        if 'generation_config' in config:
            raise ValueError('SATA retains JBF generation settings.')

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
        helper = make_jbf_query_client(
            transport, 'attacker', config['source_profile'], sleep or time.sleep,
            temperature=config['temperature'],
        )
        target = make_jbf_query_client(transport, 'target', config['source_profile'], sleep or time.sleep)

        source = load_symbols('sata/sata_pipeline.py', ['SATAAttack'], {
            'ModernBaseAttack': object, 'AttackParameter': lambda **kwargs: kwargs,
            'LLMLiteLLM': SimpleNamespace(from_config=lambda **_kw: helper),
            'Dict': Dict, 'Any': Any, 'Optional': Optional, 'List': List,
            're': re, 'random': random, 'json': json, 'os': os,
            '__file__': str(AUTHOR_IMPLEMENTATIONS / 'sata/sata_pipeline.py'),
        })['SATAAttack']

        sata_cfg = {
            **config,
            'attack_model': {'model': config['source_profile'], 'provider': 'openai'},
            'wiki_model': {'model': config['source_profile'], 'provider': 'openai'},
        }
        records = []
        for question, reference in transport.examples(questions):
            author = source.__new__(source)
            author.get_parameter_value = sata_cfg.__getitem__
            author._get_param = sata_cfg.__getitem__
            author.commendatory_words = author._load_commendatory_words()
            prompt = author.generate_attack(prompt=question, goal=question, target=reference)
            records.append({
                'question': question,
                'method': config['method'],
                'granularity': config['granularity'],
                'prompt': prompt,
                'response': str(target.query(prompt)),
            })
        return records, {}
