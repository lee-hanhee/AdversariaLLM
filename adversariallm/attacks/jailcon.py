"""Hosted binding for Jailbreak Foundry's JAIL-CON implementation."""

from dataclasses import dataclass
import json
import random
import re
import time
from typing import Any, Dict, List, Optional

from .attack import Attack
from .utils.author_runtime import (
    AUTHOR_IMPLEMENTATIONS, load_symbols, normalize_config, run_attack, validate_routes,
)
from .utils.jbf_runtime import make_jbf_query_client


@dataclass
class JAILCONConfig:
    name: str = 'jailcon'
    type: str = 'discrete'
    version: str = '0.1.0'
    seed: None = None
    # JBF target default label; hosted OpenRouter routes are configured separately.
    source_profile: str = 'gpt-4o'
    mode: str = 'cvt'
    separator: str = '{}'
    aux_dataset: str = 'jbb-benign'
    attempt_index: int = 1


class JAILCONAttack(Attack):
    """Run the preserved JBF JAIL-CON generator with explicit hosted model routes."""

    REQUIRES_HOSTED_TARGET = True

    def __init__(self, config):
        self.config = normalize_config(config)

    def run(self, target, dataset):
        return run_attack(self, target, dataset)

    @staticmethod
    def validate_config(model, config):
        validate_routes(model, config, ())
        if config.get('source_profile') != 'gpt-4o':
            raise ValueError('JAIL-CON source_profile must retain gpt-4o.')
        if config.get('mode') not in {'cvt', 'cit', 'random'}:
            raise ValueError("JAIL-CON mode must be 'cvt', 'cit', or 'random'.")
        sep = config.get('separator')
        if not isinstance(sep, str) or len(sep) != 2:
            raise ValueError('JAIL-CON separator must be a 2-character string.')
        if type(config.get('attempt_index')) is not int or config['attempt_index'] < 1:
            raise ValueError('JAIL-CON attempt_index must be an integer >= 1.')
        if 'generation_config' in config:
            raise ValueError('JAIL-CON retains JBF generation settings.')

    @staticmethod
    def _add_steps(run, record, trace, duration):
        request = next(item for item in reversed(trace) if item['role'] == 'target' and 'metadata' in item)
        run.add_step(request, messages=request['request']['messages'],
                     completions=[record['response']], raw_completions=[request['raw_text']],
                     time_taken=duration, author_result=record, author_requests=trace)

    @staticmethod
    def _execute(transport, model, config, questions, sleep=None):
        transport.client = transport.client.with_options(max_retries=0)
        target = make_jbf_query_client(transport, 'target', config['source_profile'], sleep or time.sleep)
        source = load_symbols('jailcon/jailcon_pipeline.py', ['JailConAttack'], {
            'ModernBaseAttack': object, 'AttackParameter': lambda **kwargs: kwargs, 'JBBBehaviorsLoader': object,
            'Dict': Dict, 'Any': Any, 'Optional': Optional, 'List': List, 'random': random, 're': re,
        })['JailConAttack']

        aux_dataset = json.loads((AUTHOR_IMPLEMENTATIONS / 'jailcon/jbb_benign.json').read_text(encoding='utf-8'))
        records = []
        for question, reference in transport.examples(questions):
            author = source.__new__(source)
            author.mode = config['mode']
            author.separator = config['separator']
            author.seed = 42
            author.aux_dataset = aux_dataset
            author.template_cit = author.TEMPLATE_CIT.replace('{', author.separator[0]).replace('}', author.separator[1])
            author.template_cvt = author.TEMPLATE_CVT.replace('{', author.separator[0]).replace('}', author.separator[1])
            prompt = author.generate_attack(
                prompt=question, goal=question, target=reference,
                attempt_index=config['attempt_index'],
            )
            records.append({
                'question': question,
                'mode': config['mode'],
                'prompt': prompt,
                'response': str(target.query(prompt)),
            })
        return records, {}
