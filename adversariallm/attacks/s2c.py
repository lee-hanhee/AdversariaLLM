"""Hosted binding for Jailbreak Foundry's S2C (Structured Semantic Cloaking) implementation."""

import base64
from dataclasses import dataclass, field
import random
import re
import time
from types import SimpleNamespace
from typing import Any, Dict, List

from .attack import Attack
from .utils.author_runtime import load_symbols, normalize_config, run_attack, validate_routes
from .utils.jbf_runtime import make_jbf_query_client


@dataclass
class S2CConfig:
    name: str = 's2c'
    type: str = 'discrete'
    version: str = '0.1.0'
    seed: None = None
    # JBF attacker default label; hosted OpenRouter routes are configured separately.
    source_profile: str = 'gpt-4o-mini'
    obfuscation_method: str = 'random'
    attacker: dict = field(default_factory=lambda: {'id': None, 'provider': None})


class S2CAttack(Attack):
    """Run the preserved JBF S2C generator with explicit hosted model routes."""

    REQUIRES_HOSTED_TARGET = True

    def __init__(self, config):
        self.config = normalize_config(config)

    def run(self, target, dataset):
        return run_attack(self, target, dataset)

    @staticmethod
    def validate_config(model, config):
        validate_routes(model, config, ('attacker',))
        if config.get('source_profile') != 'gpt-4o-mini':
            raise ValueError('S2C source_profile must retain gpt-4o-mini.')
        valid = {'char_noise', 'reversal', 'caesar', 'base64', 'reversal_char_noise', 'random'}
        if config.get('obfuscation_method') not in valid:
            raise ValueError(f'S2C obfuscation_method must be one of {sorted(valid)}.')
        if 'generation_config' in config:
            raise ValueError('S2C retains JBF generation settings.')

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

        s2c_cfg = {**config, 'attack_model': {'model': config['source_profile'], 'provider': 'openai'}}
        source = load_symbols('s2c/s2c_pipeline.py', ['S2CGen'], {
            'ModernBaseAttack': object, 'AttackParameter': lambda **kwargs: kwargs,
            'LLMLiteLLM': SimpleNamespace(from_config=lambda **_kw: helper),
            'Dict': Dict, 'Any': Any, 'List': List, 're': re, 'random': random, 'base64': base64,
        })['S2CGen']

        records = []
        for question, reference in transport.examples(questions):
            author = source.__new__(source)
            author.get_parameter_value = s2c_cfg.__getitem__
            prompt = author.generate_attack(prompt=question, goal=question, target=reference)
            records.append({
                'question': question,
                'obfuscation_method': config['obfuscation_method'],
                'prompt': prompt,
                'response': str(target.query(prompt)),
            })
        return records, {}
