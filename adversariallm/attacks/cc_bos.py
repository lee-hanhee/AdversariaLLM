"""Hosted binding for Jailbreak Foundry's CC-BOS implementation."""

from dataclasses import dataclass, field
import logging
import math
import random
import re
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Tuple

from .attack import Attack
from .utils.author_runtime import load_symbols, normalize_config, run_attack, validate_routes
from .utils.jbf_runtime import make_jbf_query_client


@dataclass
class CCBOSConfig:
    name: str = 'cc_bos'
    type: str = 'discrete'
    version: str = '0.1.0'
    seed: None = None
    # JBF attacker default label; hosted OpenRouter routes are configured separately.
    source_profile: str = 'gpt-4o-mini'
    population_size: int = 5
    max_iterations: int = 5
    stagnation_threshold: int = 2
    step_size_decay: float = 0.95
    cauchy_mutation_scale: float = 0.2
    max_resample_attempts: int = 5
    early_stop_threshold: int = 120
    attacker: dict = field(default_factory=lambda: {'id': None, 'provider': None})
    judge: dict = field(default_factory=lambda: {'id': None, 'provider': None})


class CCBOSAttack(Attack):
    """Run the preserved JBF CC-BOS FOA optimizer with explicit hosted model routes."""

    REQUIRES_HOSTED_TARGET = True

    def __init__(self, config):
        self.config = normalize_config(config)

    def run(self, target, dataset):
        return run_attack(self, target, dataset)

    @staticmethod
    def validate_config(model, config):
        validate_routes(model, config, ('attacker', 'judge'))
        if config.get('source_profile') != 'gpt-4o-mini':
            raise ValueError('CC-BOS source_profile must retain gpt-4o-mini.')
        for key in ('population_size', 'max_iterations', 'stagnation_threshold', 'max_resample_attempts'):
            if type(config.get(key)) is not int or config[key] < 1:
                raise ValueError(f'CC-BOS {key} must be an integer >= 1.')
        if 'generation_config' in config:
            raise ValueError('CC-BOS retains JBF generation settings.')

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
        transport.routes['judge'] = config['judge']
        transport.client = transport.client.with_options(max_retries=0)
        attacker_llm = make_jbf_query_client(transport, 'attacker', config['source_profile'], sleep or time.sleep)
        judge_llm = make_jbf_query_client(
            transport, 'judge', config['source_profile'], sleep or time.sleep, temperature=0.0,
        )
        target_llm = make_jbf_query_client(transport, 'target', config['source_profile'], sleep or time.sleep)

        cc_cfg = {
            **config,
            'attack_model': {'model': 'attacker_role', 'provider': 'openai'},
            'translation_model': {'model': 'attacker_role', 'provider': 'openai'},
            'judge_model': {'model': 'judge_role', 'provider': 'openai'},
        }

        def from_config(model_name=None, **_kw):
            if model_name == 'judge_role':
                return judge_llm
            if model_name == 'target_role':
                return target_llm
            return attacker_llm

        class _ModernBase:
            def __init__(self, args=None, **kwargs):
                self.args = args
                self.get_parameter_value = cc_cfg.__getitem__

        source = load_symbols('cc_bos/cc_bos_pipeline.py', ['Strategy', 'CCBOSAttack'], {
            'ModernBaseAttack': _ModernBase, 'AttackParameter': lambda **kwargs: kwargs,
            'LLMLiteLLM': SimpleNamespace(from_config=from_config),
            'Dict': Dict, 'Any': Any, 'List': List, 'Optional': Optional, 'Tuple': Tuple, 'Set': Set,
            're': re, 'math': math, 'random': random, 'logger': logging.getLogger(__name__),
        })['CCBOSAttack']

        records = []
        args = SimpleNamespace(model='target_role', provider='openai', api_key=None, api_base=None)
        for question, reference in transport.examples(questions):
            author = source(args=args)
            prompt = author.generate_attack(prompt=question, goal=question, target=reference)
            records.append({
                'question': question,
                'prompt': prompt,
                'response': str(target_llm.query(prompt)),
            })
        return records, {}
