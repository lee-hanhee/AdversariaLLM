"""Hosted binding for Jailbreak Foundry's SCP (Sugar-Coated Poison) implementation."""

from dataclasses import dataclass, field
import random
import time
from typing import Any, Dict, List, Optional

from .attack import Attack
from .utils.author_runtime import load_symbols, normalize_config, run_attack, validate_routes
from .utils.jbf_runtime import make_jbf_query_client


@dataclass
class SCPConfig:
    name: str = 'scp'
    type: str = 'discrete'
    version: str = '0.1.0'
    seed: None = None
    # JBF helper default label; hosted OpenRouter routes are configured separately.
    source_profile: str = 'gpt-3.5-turbo'
    max_benign_retries: int = 3
    strategies: list[str] = field(default_factory=lambda: ['base', 'code', 'json', 'code_json'])
    attempt_index: int = 1
    attacker: dict = field(default_factory=lambda: {'id': None, 'provider': None})


class SCPAttack(Attack):
    """Run the preserved JBF SCP generator with explicit hosted model routes."""

    REQUIRES_HOSTED_TARGET = True

    def __init__(self, config):
        self.config = normalize_config(config)

    def run(self, target, dataset):
        return run_attack(self, target, dataset)

    @staticmethod
    def validate_config(model, config):
        validate_routes(model, config, ('attacker',))
        if config.get('source_profile') != 'gpt-3.5-turbo':
            raise ValueError('SCP source_profile must retain gpt-3.5-turbo.')
        if type(config.get('max_benign_retries')) is not int or config['max_benign_retries'] < 1:
            raise ValueError('SCP max_benign_retries must be an integer >= 1.')
        if type(config.get('attempt_index')) is not int or config['attempt_index'] < 1:
            raise ValueError('SCP attempt_index must be an integer >= 1.')
        strategies = config.get('strategies')
        valid = {'base', 'code', 'json', 'code_json'}
        if not isinstance(strategies, list) or not strategies or not all(s in valid for s in strategies):
            raise ValueError('SCP strategies must be a non-empty subset of base, code, json, code_json.')
        if 'generation_config' in config:
            raise ValueError('SCP retains JBF generation settings.')

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
        source = load_symbols('scp/scp_pipeline.py', ['BENIGN_TOKEN_POOL', 'REFUSAL_KEYWORDS', 'SCPAttack'], {
            'ModernBaseAttack': object, 'AttackParameter': lambda **kwargs: kwargs, 'LLMLiteLLM': object,
            'Dict': Dict, 'Any': Any, 'Optional': Optional, 'List': List, 'random': random,
        })['SCPAttack']

        records = []
        benign_cache = {}
        for question, reference in transport.examples(questions):
            author = source.__new__(source)
            author.attacker_llm = helper
            author.max_benign_retries = config['max_benign_retries']
            author.strategies = list(config['strategies'])
            author._benign_cache = benign_cache
            prompt = author.generate_attack(
                prompt=question, goal=question, target=reference,
                attempt_index=config['attempt_index'],
            )
            records.append({
                'question': question,
                'benign_input': benign_cache.get(question, ''),
                'prompt': prompt,
                'response': str(target.query(prompt)),
            })
        return records, {}
