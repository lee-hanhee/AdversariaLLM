"""Hosted binding for Jailbreak Foundry's MAJIC implementation."""

from dataclasses import dataclass, field
import logging
import random
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .attack import Attack
from .utils.author_runtime import load_symbols, normalize_config, run_attack, validate_routes
from .utils.jbf_runtime import make_jbf_query_client


@dataclass
class MAJICConfig:
    name: str = 'majic'
    type: str = 'discrete'
    version: str = '0.1.0'
    seed: None = None
    # JBF attacker default label; hosted OpenRouter routes are configured separately.
    source_profile: str = 'gpt-4o-mini'
    max_queries: int = 20
    learning_rate: float = 0.1
    discount_factor: float = 0.9
    decay_factor: float = 0.99
    temperature: float = 1.0
    reset_factor: float = 0.1
    attacker: dict = field(default_factory=lambda: {'id': None, 'provider': None})
    judge: dict = field(default_factory=lambda: {'id': None, 'provider': None})


class MAJICAttack(Attack):
    """Run the preserved JBF MAJIC Q-learning loop with explicit hosted model routes."""

    REQUIRES_HOSTED_TARGET = True

    def __init__(self, config):
        self.config = normalize_config(config)

    def run(self, target, dataset):
        return run_attack(self, target, dataset)

    @staticmethod
    def validate_config(model, config):
        validate_routes(model, config, ('attacker', 'judge'))
        if config.get('source_profile') != 'gpt-4o-mini':
            raise ValueError('MAJIC source_profile must retain gpt-4o-mini.')
        if type(config.get('max_queries')) is not int or config['max_queries'] < 1:
            raise ValueError('MAJIC max_queries must be an integer >= 1.')
        for key in ('learning_rate', 'discount_factor', 'decay_factor', 'temperature', 'reset_factor'):
            val = config.get(key)
            if type(val) not in (int, float) or val < 0:
                raise ValueError(f'MAJIC {key} must be >= 0.')
        if 'generation_config' in config:
            raise ValueError('MAJIC retains JBF generation settings.')

    @staticmethod
    def _add_steps(run, record, trace, duration):
        target_requests = [item for item in trace if item['role'] == 'target' and 'metadata' in item]
        for index, attempt in enumerate(record['attempts']):
            request = target_requests[index]
            response = attempt['response']
            raw = request.get('raw_text')
            run.add_step(
                request,
                messages=request['request']['messages'],
                completions=[] if response is None else [response],
                raw_completions=[] if raw is None else [raw],
                scores={'majic_author': {'jailbroken': [float(attempt['jailbroken'])]}},
                time_taken=duration if index == len(record['attempts']) - 1 else 0.0,
                author_result=attempt,
                author_requests=trace,
            )

    @staticmethod
    def _execute(transport, model, config, questions, sleep=None):
        transport.routes['attacker'] = config['attacker']
        transport.routes['judge'] = config['judge']
        transport.client = transport.client.with_options(max_retries=0)
        attacker_llm = make_jbf_query_client(transport, 'attacker', config['source_profile'], sleep or time.sleep)
        target_llm = make_jbf_query_client(transport, 'target', config['source_profile'], sleep or time.sleep)
        evaluator_llm = make_jbf_query_client(
            transport, 'judge', config['source_profile'], sleep or time.sleep, temperature=0.0,
        )

        source = load_symbols('majic/majic_pipeline.py', ['MajicGenAttack'], {
            'ModernBaseAttack': object, 'AttackParameter': lambda **kwargs: kwargs, 'LLMLiteLLM': object,
            'Dict': Dict, 'Any': Any, 'Optional': Optional, 'List': List,
            'threading': threading, 'np': np, 'random': random, 'logger': logging.getLogger(__name__),
        })['MajicGenAttack']

        # Share Q-learning transition matrix across dataset examples as in JBF __init__.
        author = source.__new__(source)
        author.max_queries = config['max_queries']
        author.learning_rate = config['learning_rate']
        author.discount_factor = config['discount_factor']
        author.decay_factor = config['decay_factor']
        author.temperature = config['temperature']
        author.reset_factor = config['reset_factor']
        author.strategy_keys = list(author.STRATEGIES.keys())
        author.num_strategies = len(author.strategy_keys)
        author.matrix = np.ones((author.num_strategies, author.num_strategies)) * 0.1
        author.lock = threading.Lock()
        author.current_learning_rate = author.learning_rate
        author.attacker_llm = attacker_llm
        author.target_llm = target_llm
        author.evaluator_llm = evaluator_llm

        records = []
        for question, reference in transport.examples(questions):
            attempts = []
            original_query = target_llm.query
            original_eval = author._evaluate_response

            def tracked_query(prompt, **kwargs):
                resp = str(original_query(prompt, **kwargs))
                attempts.append({'prompt': prompt, 'response': resp, 'jailbroken': False})
                return resp

            def tracked_eval(goal, response, cost_tracker=None):
                verdict = bool(original_eval(goal, response, cost_tracker=cost_tracker))
                if attempts:
                    attempts[-1]['jailbroken'] = verdict
                return verdict

            target_llm.query = tracked_query
            author._evaluate_response = tracked_eval
            final_prompt = author.generate_attack(prompt=question, goal=question, target=reference)
            target_llm.query = original_query
            author._evaluate_response = original_eval

            records.append({
                'question': question,
                'prompt': final_prompt,
                'response': attempts[-1]['response'] if attempts else '',
                'jailbroken': any(a['jailbroken'] for a in attempts),
                'attempts': attempts,
            })
        return records, {}
