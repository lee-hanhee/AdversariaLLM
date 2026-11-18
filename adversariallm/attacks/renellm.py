"""Hosted binding for Jailbreak Foundry's ReNeLLM implementation."""

from dataclasses import dataclass, field
import logging
import random
import time
from typing import Any, Dict, List, Optional

from .attack import Attack
from .utils.author_runtime import load_symbols, normalize_config, run_attack, validate_routes
from .utils.jbf_runtime import make_jbf_query_client


@dataclass
class ReNeLLMConfig:
    name: str = 'renellm'
    type: str = 'discrete'
    version: str = '0.1.0'
    seed: None = None
    # JBF rewrite helper default label; hosted OpenRouter routes are configured separately.
    source_profile: str = 'gpt-3.5-turbo'
    rewrite_temperature: float = 0.5
    max_iterations: int = 5
    attacker: dict = field(default_factory=lambda: {'id': None, 'provider': None})
    judge: dict = field(default_factory=lambda: {'id': None, 'provider': None})


class ReNeLLMAttack(Attack):
    """Run the preserved JBF ReNeLLM generator with explicit hosted model routes."""

    REQUIRES_HOSTED_TARGET = True

    def __init__(self, config):
        self.config = normalize_config(config)

    def run(self, target, dataset):
        return run_attack(self, target, dataset)

    @staticmethod
    def validate_config(model, config):
        validate_routes(model, config, ('attacker', 'judge'))
        if config.get('source_profile') != 'gpt-3.5-turbo':
            raise ValueError('ReNeLLM source_profile must retain gpt-3.5-turbo.')
        if type(config.get('max_iterations')) is not int or config['max_iterations'] < 1:
            raise ValueError('ReNeLLM max_iterations must be an integer >= 1.')
        temp = config.get('rewrite_temperature')
        if type(temp) not in (int, float) or temp < 0:
            raise ValueError('ReNeLLM rewrite_temperature must be >= 0.')
        if 'generation_config' in config:
            raise ValueError('ReNeLLM retains JBF generation settings.')

    @staticmethod
    def _add_steps(run, record, trace, duration):
        request = next(item for item in reversed(trace) if item['role'] == 'target' and 'metadata' in item)
        run.add_step(request, messages=request['request']['messages'],
                     completions=[record['response']], raw_completions=[request['raw_text']],
                     time_taken=duration, author_result=record, author_requests=trace)

    @staticmethod
    def _execute(transport, model, config, questions, sleep=None):
        transport.routes['attacker'] = config['attacker']
        transport.routes['judge'] = config['judge']
        transport.client = transport.client.with_options(max_retries=0)
        rewrite_llm = make_jbf_query_client(
            transport, 'attacker', config['source_profile'], sleep or time.sleep,
            temperature=config['rewrite_temperature'],
        )
        judge_llm = make_jbf_query_client(
            transport, 'judge', config['source_profile'], sleep or time.sleep, temperature=0.0,
        )
        target = make_jbf_query_client(transport, 'target', config['source_profile'], sleep or time.sleep)

        source = load_symbols('renellm/renellm_pipeline.py', ['ReNeLLMAttack'], {
            'ModernBaseAttack': object, 'AttackParameter': lambda **kwargs: kwargs, 'LLMLiteLLM': object,
            'Dict': Dict, 'Any': Any, 'Optional': Optional, 'List': List,
            'random': random, 'logger': logging.getLogger(__name__),
        })['ReNeLLMAttack']

        records = []
        for question, reference in transport.examples(questions):
            author = source.__new__(source)
            author.get_parameter_value = config.__getitem__
            author.rewrite_llm = rewrite_llm
            author.judge_llm = judge_llm
            prompt = author.generate_attack(prompt=question, goal=question, target=reference)
            records.append({
                'question': question,
                'prompt': prompt,
                'response': str(target.query(prompt)),
            })
        return records, {}
