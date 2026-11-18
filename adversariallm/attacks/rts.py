"""Hosted binding for Jailbreak Foundry's RTS-Attack implementation."""

from dataclasses import dataclass, field
import json
import re
import time
from typing import Any, Dict, Optional

from .attack import Attack
from .utils.author_runtime import load_symbols, normalize_config, run_attack, validate_routes
from .utils.jbf_runtime import make_jbf_query_client


@dataclass
class RTSConfig:
    name: str = 'rts'
    type: str = 'discrete'
    version: str = '0.1.0'
    seed: None = None
    # JBF attack helper default label; hosted OpenRouter routes are configured separately.
    source_profile: str = 'qwen3-14b'
    num_rewrites: int = 1
    scenario_genre: str = 'Crime News Report'
    use_json_format: bool = True
    attacker: dict = field(default_factory=lambda: {'id': None, 'provider': None})


class RTSAttack(Attack):
    """Run the preserved JBF RTS-Attack generator with explicit hosted model routes."""

    REQUIRES_HOSTED_TARGET = True

    def __init__(self, config):
        self.config = normalize_config(config)

    def run(self, target, dataset):
        return run_attack(self, target, dataset)

    @staticmethod
    def validate_config(model, config):
        validate_routes(model, config, ('attacker',))
        if config.get('source_profile') != 'qwen3-14b':
            raise ValueError('RTS source_profile must retain qwen3-14b.')
        if type(config.get('num_rewrites')) is not int or config['num_rewrites'] < 1:
            raise ValueError('RTS num_rewrites must be an integer >= 1.')
        genre = config.get('scenario_genre')
        if not isinstance(genre, str) or not genre.strip():
            raise ValueError('RTS scenario_genre must be a non-empty string.')
        if type(config.get('use_json_format')) is not bool:
            raise ValueError('RTS use_json_format must be a boolean.')
        if 'generation_config' in config:
            raise ValueError('RTS retains JBF generation settings.')

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

        source = load_symbols('rts/rts_pipeline.py', ['RTSGen'], {
            'ModernBaseAttack': object, 'AttackParameter': lambda **kwargs: kwargs, 'LLMLiteLLM': object,
            'Dict': Dict, 'Any': Any, 'Optional': Optional, 'json': json, 're': re,
        })['RTSGen']

        records = []
        for question, reference in transport.examples(questions):
            author = source.__new__(source)
            author.get_parameter_value = config.__getitem__
            author.attack_llm = helper
            prompt = author.generate_attack(prompt=question, goal=question, target=reference)
            records.append({
                'question': question,
                'prompt': prompt,
                'response': str(target.query(prompt)),
            })
        return records, {}
