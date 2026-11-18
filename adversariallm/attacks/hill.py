"""Hosted binding for Jailbreak Foundry's HILL implementation."""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, Optional

from .attack import Attack
from .utils.author_runtime import load_symbols, normalize_config, run_attack, validate_routes
from .utils.jbf_runtime import make_jbf_query_client


@dataclass
class HILLConfig:
    name: str = 'hill'
    type: str = 'discrete'
    version: str = '0.1.0'
    seed: None = None
    # JBF helper default label; hosted OpenRouter routes are configured separately.
    source_profile: str = 'gpt-4o'
    max_attempts: int = 3
    intention_check: bool = True
    attacker: dict = field(default_factory=lambda: {'id': None, 'provider': None})


class HILLAttack(Attack):
    """Run the preserved JBF HILL generator with explicit hosted model routes."""

    REQUIRES_HOSTED_TARGET = True

    def __init__(self, config):
        self.config = normalize_config(config)

    def run(self, target, dataset):
        return run_attack(self, target, dataset)

    @staticmethod
    def validate_config(model, config):
        validate_routes(model, config, ('attacker',))
        if config.get('source_profile') != 'gpt-4o':
            raise ValueError('HILL source_profile must retain gpt-4o.')
        if type(config.get('max_attempts')) is not int or config['max_attempts'] < 1:
            raise ValueError('HILL max_attempts must be an integer >= 1.')
        if type(config.get('intention_check')) is not bool:
            raise ValueError('HILL intention_check must be a boolean.')
        if 'generation_config' in config:
            raise ValueError('HILL retains JBF generation settings.')

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
        source = load_symbols('hill/hill_pipeline.py', ['HillAttack'], {
            'ModernBaseAttack': object, 'AttackParameter': lambda **kwargs: kwargs,
            'Dict': Dict, 'Any': Any, 'Optional': Optional,
        })['HillAttack']

        records = []
        for question, reference in transport.examples(questions):
            author = source.__new__(source)
            author.attacker_llm = helper
            author.max_attempts = config['max_attempts']
            author.intention_check = config['intention_check']
            prompt = author.generate_attack(prompt=question, goal=question, target=reference)
            records.append({
                'question': question,
                'prompt': prompt,
                'response': str(target.query(prompt)),
            })
        return records, {}
