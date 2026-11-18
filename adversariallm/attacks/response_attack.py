"""Hosted binding for Jailbreak Foundry's Response Attack (RA-DRI / RA-SRI) implementation."""

from dataclasses import dataclass, field
import re
import time
from typing import Any, Dict, List, Optional

from .attack import Attack
from .utils.author_runtime import load_symbols, normalize_config, run_attack, validate_routes
from .utils.jbf_runtime import make_jbf_query_client


@dataclass
class ResponseAttackConfig:
    name: str = 'response_attack'
    type: str = 'discrete'
    version: str = '0.1.0'
    seed: None = None
    # JBF auxiliary helper default label; hosted OpenRouter routes are configured separately.
    source_profile: str = 'gpt-4o'
    attack_type: str = 'dri'
    max_retries: int = 3
    attacker: dict = field(default_factory=lambda: {'id': None, 'provider': None})


class ResponseAttack(Attack):
    """Run the preserved JBF Response Attack (DRI/SRI) generator with explicit hosted model routes."""

    REQUIRES_HOSTED_TARGET = True

    def __init__(self, config):
        self.config = normalize_config(config)

    def run(self, target, dataset):
        return run_attack(self, target, dataset)

    @staticmethod
    def validate_config(model, config):
        validate_routes(model, config, ('attacker',))
        if config.get('source_profile') != 'gpt-4o':
            raise ValueError('ResponseAttack source_profile must retain gpt-4o.')
        if config.get('attack_type') not in {'dri', 'sri'}:
            raise ValueError("ResponseAttack attack_type must be 'dri' or 'sri'.")
        if type(config.get('max_retries')) is not int or config['max_retries'] < 1:
            raise ValueError('ResponseAttack max_retries must be an integer >= 1.')
        if 'generation_config' in config:
            raise ValueError('ResponseAttack retains JBF generation settings.')

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

        source = load_symbols(
            'response_attack/response_attack_pipeline.py',
            ['TEMPLATE_REWRITE', 'TEMPLATE_DRI', 'TEMPLATE_SRI', 'ResponseAttackGen'],
            {
                'ModernBaseAttack': object, 'AttackParameter': lambda **kwargs: kwargs, 'LLMLiteLLM': object,
                'Dict': Dict, 'Any': Any, 'Optional': Optional, 'List': List, 're': re,
            },
        )['ResponseAttackGen']

        records = []
        for question, reference in transport.examples(questions):
            author = source.__new__(source)
            author.get_parameter_value = config.__getitem__
            author.aux_llm = helper
            messages = author.generate_attack(prompt=question, goal=question, target=reference)
            records.append({
                'question': question,
                'attack_type': config['attack_type'],
                'prompt': messages,
                'response': str(target.query(messages)),
            })
        return records, {}
