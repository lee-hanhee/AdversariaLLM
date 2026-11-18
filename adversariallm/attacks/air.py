"""Hosted binding for Jailbreak Foundry's AIR implementation."""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional

from .attack import Attack
from .utils.author_runtime import load_symbols, normalize_config, run_attack, validate_routes
from .utils.jbf_runtime import make_jbf_query_client


@dataclass
class AIRConfig:
    name: str = 'air'
    type: str = 'discrete'
    version: str = '0.1.0'
    seed: None = None
    # JBF rewrite helper default label; hosted OpenRouter routes are configured separately.
    source_profile: str = 'gpt-4o-mini'
    attack_method: str = 'k4'
    use_target_for_rewrite: bool = False
    target_max_tokens: int = 300
    clean_conversation: bool = False
    attacker: dict = field(default_factory=lambda: {'id': None, 'provider': None})


class AIRAttack(Attack):
    """Run the preserved JBF AIR generator with explicit hosted model routes."""

    REQUIRES_HOSTED_TARGET = True

    def __init__(self, config):
        self.config = normalize_config(config)

    def run(self, target, dataset):
        return run_attack(self, target, dataset)

    @staticmethod
    def validate_config(model, config):
        validate_routes(model, config, ('attacker',))
        if config.get('source_profile') != 'gpt-4o-mini':
            raise ValueError('AIR source_profile must retain gpt-4o-mini.')
        if config.get('attack_method') not in {'k2', 'k3', 'k4', 'k5', 'k6'}:
            raise ValueError("AIR attack_method must be one of 'k2', 'k3', 'k4', 'k5', or 'k6'.")
        if type(config.get('target_max_tokens')) is not int or config['target_max_tokens'] < 1:
            raise ValueError('AIR target_max_tokens must be an integer >= 1.')
        for key in ('use_target_for_rewrite', 'clean_conversation'):
            if type(config.get(key)) is not bool:
                raise ValueError(f'AIR {key} must be a boolean.')
        if 'generation_config' in config:
            raise ValueError('AIR retains JBF generation settings.')

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
        transport.client = transport.client.with_options(max_retries=0)
        helper = make_jbf_query_client(transport, 'attacker', config['source_profile'], sleep or time.sleep)
        target = make_jbf_query_client(transport, 'target', config['source_profile'], sleep or time.sleep)

        source = load_symbols('air/air_pipeline.py', ['AirGen'], {
            'ModernBaseAttack': object, 'AttackParameter': lambda **kwargs: kwargs, 'LLMLiteLLM': object,
            'Dict': Dict, 'Any': Any, 'Optional': Optional, 'List': List,
        })['AirGen']

        records = []
        for question, reference in transport.examples(questions):
            author = source.__new__(source)
            author.get_parameter_value = config.__getitem__
            author._get_target_llm = lambda: target
            author._get_rewrite_llm = lambda: (target if config['use_target_for_rewrite'] else helper)
            messages = author.generate_attack(prompt=question, goal=question, target=reference)
            records.append({
                'question': question,
                'attack_method': config['attack_method'],
                'prompt': messages,
                'response': str(target.query(messages)),
            })
        return records, {}
