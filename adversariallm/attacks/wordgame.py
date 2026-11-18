"""Hosted binding for Jailbreak Foundry's WordGame and WordGame+ implementation."""

from dataclasses import dataclass, field
import json
import logging
import os
import re
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

from .attack import Attack
from .utils.author_runtime import load_symbols, normalize_config, run_attack, validate_routes


@dataclass
class WordGameConfig:
    name: str = 'wordgame'
    type: str = 'discrete'
    version: str = '0.1.0'
    seed: None = None
    # The JBF helper's default source label; hosted routes are configured separately.
    source_profile: str = 'gpt-3.5-turbo'
    use_wordgame_plus: bool = True
    num_hints: int = 5
    attacker: dict = field(default_factory=lambda: {'id': None, 'provider': None})


class _BaseLLM:
    """Supply only the framework state used by the preserved LiteLLM wrapper."""

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.last_query = None


def _query_client(transport, role, source_profile, sleep):
    # Keep JBF message parsing, temperature=1, omitted token limit, and five retries.
    # Disable nested OpenAI SDK retries so each original completion attempt is one call.
    def completion(**kwargs):
        kwargs.pop('api_key', None)
        kwargs.pop('api_base', None)
        return transport.create_for(role, **kwargs)

    bindings = {
        'BaseLLM': _BaseLLM, 'CostTracker': Any,
        'Dict': Dict, 'List': List, 'Optional': Optional, 'Union': Union, 'Any': Any, 'Tuple': Tuple,
        'os': os, 'json': json, 'logger': logging.getLogger(__name__),
        'time': SimpleNamespace(time=time.time, monotonic=time.monotonic, sleep=sleep),
        'completion': completion,
    }
    source = load_symbols('wordgame/wordgame_llm.py', ['RichResponse', 'RichResponseList', 'LLMLiteLLM'], bindings)
    llm = source['LLMLiteLLM'](
        model_identifier=f'openai/{source_profile}', model_name=source_profile,
        provider='openai', real_provider='openai', log_dir=None)
    # Repository traces replace JBF cost tracking and approximate reasoning-token accounting.
    llm._resolve_cost_tracker = lambda _tracker: None
    llm.get_token_count = lambda text: int(len(text.split()) * 1.33)
    return llm


class WordGameAttack(Attack):
    """Run the preserved JBF prompt generator with explicit hosted model routes."""

    REQUIRES_HOSTED_TARGET = True

    def __init__(self, config):
        self.config = normalize_config(config)

    def run(self, target, dataset):
        return run_attack(self, target, dataset)

    @staticmethod
    def validate_config(model, config):
        validate_routes(model, config, ('attacker',))
        if config.get('source_profile') != 'gpt-3.5-turbo':
            raise ValueError('WordGame source_profile must retain gpt-3.5-turbo.')
        if type(config.get('num_hints')) is not int or config['num_hints'] < 1:
            raise ValueError('WordGame num_hints must be an integer >= 1.')
        if type(config.get('use_wordgame_plus')) is not bool:
            raise ValueError('WordGame use_wordgame_plus must be a boolean.')
        if 'generation_config' in config:
            raise ValueError('WordGame retains JBF generation settings.')

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
        helper = _query_client(transport, 'attacker', config['source_profile'], sleep or time.sleep)
        target = _query_client(transport, 'target', config['source_profile'], sleep or time.sleep)
        source = load_symbols('wordgame/wordgame_pipeline.py', ['WordGameGen'], {
            'ModernBaseAttack': object, 'AttackParameter': lambda **kwargs: kwargs,
            're': re, 'logger': logging.getLogger(__name__),
        })['WordGameGen']

        records = []
        for question, reference in transport.examples(questions):
            # Replace framework initialization/parameter lookup; execute every algorithm method unchanged.
            author = source.__new__(source)
            author.attacker_llm = helper
            author.get_parameter_value = config.__getitem__
            record = {'question': question}
            original_hints = author._generate_hints

            def hints(word, num_hints):
                value = original_hints(word, num_hints)
                record.update(malicious_word=word, hints=value)
                return value

            author._generate_hints = hints
            prompt = author.generate_attack(prompt=question, goal=question, target=reference)
            # Match the JBF runner's default single attempt and target LLMLiteLLM.query call.
            record.update(prompt=prompt, response=str(target.query(prompt)))
            records.append(record)
        return records, {}
