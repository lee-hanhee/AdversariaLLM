"""Hosted binding for Jailbreak Foundry's BPJ (Boundary Point Jailbreaking) implementation."""

from dataclasses import dataclass
import random
import string
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from .attack import Attack
from .utils.author_runtime import load_symbols, normalize_config, run_attack, validate_routes
from .utils.jbf_runtime import make_jbf_query_client


@dataclass
class BPJConfig:
    name: str = 'bpj'
    type: str = 'discrete'
    version: str = '0.1.0'
    seed: None = None
    # JBF target default label; hosted OpenRouter routes are configured separately.
    source_profile: str = 'gpt-4o'
    population_size: int = 8
    bp_target_size: int = 16
    initial_noise: float = 0.4
    noise_reduction_step: float = 0.05
    noise_threshold: float = 0.1
    eval_sample_size: int = 10
    query_budget: int = 100
    prefix_length: int = 10


class BPJAttack(Attack):
    """Run the preserved JBF BPJ boundary search with explicit hosted model routes."""

    REQUIRES_HOSTED_TARGET = True

    def __init__(self, config):
        self.config = normalize_config(config)

    def run(self, target, dataset):
        return run_attack(self, target, dataset)

    @staticmethod
    def validate_config(model, config):
        validate_routes(model, config, ())
        if config.get('source_profile') != 'gpt-4o':
            raise ValueError('BPJ source_profile must retain gpt-4o.')
        for key in ('population_size', 'bp_target_size', 'eval_sample_size', 'query_budget', 'prefix_length'):
            if type(config.get(key)) is not int or config[key] < 1:
                raise ValueError(f'BPJ {key} must be an integer >= 1.')
        if 'generation_config' in config:
            raise ValueError('BPJ retains JBF generation settings.')

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
        transport.client = transport.client.with_options(max_retries=0)
        target = make_jbf_query_client(transport, 'target', config['source_profile'], sleep or time.sleep)
        source = load_symbols('bpj/bpj_pipeline.py', ['BPJGen'], {
            'ModernBaseAttack': object, 'AttackParameter': lambda **kwargs: kwargs,
            'LLMLiteLLM': SimpleNamespace(from_config=lambda **_kw: target),
            'Dict': Dict, 'Any': Any, 'Optional': Optional, 'List': List,
            'random': random, 'string': string,
        })['BPJGen']

        records = []
        for question, reference in transport.examples(questions):
            author = source.__new__(source)
            author.args = SimpleNamespace(model=config['source_profile'], provider='openai', api_key=None, api_base=None)
            author.get_parameter_value = config.__getitem__
            author.population_size = config['population_size']
            author.bp_target_size = config['bp_target_size']
            author.initial_noise = config['initial_noise']
            author.noise_reduction_step = config['noise_reduction_step']
            author.noise_threshold = config['noise_threshold']
            author.eval_sample_size = config['eval_sample_size']
            author.query_budget = config['query_budget']
            author.prefix_length = config['prefix_length']
            prompt = author.generate_attack(prompt=question, goal=question, target=reference)
            records.append({
                'question': question,
                'prompt': prompt,
                'response': str(target.query(prompt)),
            })
        return records, {}
