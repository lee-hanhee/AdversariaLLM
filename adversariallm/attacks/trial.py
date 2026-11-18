"""Hosted binding for Jailbreak Foundry's TRIAL implementation."""

from dataclasses import dataclass, field
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from .attack import Attack
from .utils.author_runtime import load_symbols, normalize_config, run_attack, validate_routes
from .utils.jbf_runtime import make_jbf_query_client


@dataclass
class TRIALConfig:
    name: str = 'trial'
    type: str = 'discrete'
    version: str = '0.1.0'
    seed: None = None
    # JBF attacker/helper default label; hosted OpenRouter routes are configured separately.
    source_profile: str = 'gpt-4o'
    n_iterations: int = 5
    max_retries: int = 5
    attack_temperature: float = 1.0
    target_temperature: float = 1.0
    judge_temperature: float = 0.0
    attacker: dict = field(default_factory=lambda: {'id': None, 'provider': None})
    judge: dict = field(default_factory=lambda: {'id': None, 'provider': None})


class TRIALAttack(Attack):
    """Run the preserved JBF TRIAL interactive loop with explicit hosted model routes."""

    REQUIRES_HOSTED_TARGET = True

    def __init__(self, config):
        self.config = normalize_config(config)

    def run(self, target, dataset):
        return run_attack(self, target, dataset)

    @staticmethod
    def validate_config(model, config):
        validate_routes(model, config, ('attacker', 'judge'))
        if config.get('source_profile') != 'gpt-4o':
            raise ValueError('TRIAL source_profile must retain gpt-4o.')
        for key in ('n_iterations', 'max_retries'):
            if type(config.get(key)) is not int or config[key] < 1:
                raise ValueError(f'TRIAL {key} must be an integer >= 1.')
        for key in ('attack_temperature', 'target_temperature', 'judge_temperature'):
            val = config.get(key)
            if type(val) not in (int, float) or val < 0:
                raise ValueError(f'TRIAL {key} must be >= 0.')
        if 'generation_config' in config:
            raise ValueError('TRIAL retains JBF generation settings.')

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
                scores={'trial_author': {'jailbroken': [float(attempt['jailbroken'])]}},
                time_taken=duration if index == len(record['attempts']) - 1 else 0.0,
                author_result=attempt,
                author_requests=trace,
            )

    @staticmethod
    def _execute(transport, model, config, questions, sleep=None):
        transport.routes['attacker'] = config['attacker']
        transport.routes['judge'] = config['judge']
        transport.client = transport.client.with_options(max_retries=0)
        attack_llm = make_jbf_query_client(
            transport, 'attacker', config['source_profile'], sleep or time.sleep,
            temperature=config['attack_temperature'],
        )
        helper_llm = make_jbf_query_client(
            transport, 'attacker', config['source_profile'], sleep or time.sleep,
            temperature=config['attack_temperature'],
        )
        judge_llm = make_jbf_query_client(
            transport, 'judge', config['source_profile'], sleep or time.sleep,
            temperature=config['judge_temperature'],
        )
        target_llm = make_jbf_query_client(
            transport, 'target', config['source_profile'], sleep or time.sleep,
            temperature=config['target_temperature'],
        )

        source = load_symbols('trial/trial_pipeline.py', ['TrialGen'], {
            'ModernBaseAttack': object, 'AttackParameter': lambda **kwargs: kwargs, 'LLMLiteLLM': object,
            'Dict': Dict, 'Any': Any, 'Optional': Optional, 'List': List,
            'json': json, 're': re, 'logger': logging.getLogger(__name__),
        })['TrialGen']

        records = []
        for question, reference in transport.examples(questions):
            author = source.__new__(source)
            author.args = None
            author.n_iterations = config['n_iterations']
            author.max_retries = config['max_retries']
            author.get_parameter_value = config.__getitem__
            author.attack_llm = attack_llm
            author.helper_llm = helper_llm
            author.judge_llm = judge_llm
            author.target_llm = target_llm

            attempts = []
            original_query = target_llm.query
            original_judge = author._judge_response

            def tracked_query(messages, **kwargs):
                resp = str(original_query(messages, **kwargs))
                attempts.append({'messages': list(messages), 'response': resp, 'jailbroken': False})
                return resp

            def tracked_judge(goal, response, cost_tracker=None):
                verdict = bool(original_judge(goal, response, cost_tracker=cost_tracker))
                if attempts:
                    attempts[-1]['jailbroken'] = verdict
                return verdict

            target_llm.query = tracked_query
            author._judge_response = tracked_judge
            final_messages = author.generate_attack(prompt=question, goal=question, target=reference)
            target_llm.query = original_query

            records.append({
                'question': question,
                'prompt': final_messages,
                'response': attempts[-1]['response'] if attempts else '',
                'jailbroken': any(a['jailbroken'] for a in attempts),
                'attempts': attempts,
            })
        return records, {}
