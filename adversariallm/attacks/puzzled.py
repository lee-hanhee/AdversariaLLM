"""Hosted binding for Jailbreak Foundry's PUZZLED implementation."""

from dataclasses import dataclass, field
import logging
import random
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from .attack import Attack
from .utils.author_runtime import load_symbols, normalize_config, run_attack, validate_routes
from .utils.jbf_runtime import make_jbf_query_client


@dataclass
class PUZZLEDConfig:
    name: str = 'puzzled'
    type: str = 'discrete'
    version: str = '0.1.0'
    seed: None = None
    # JBF clue helper default label; hosted OpenRouter routes are configured separately.
    source_profile: str = 'gpt-4o'
    puzzle_type: str = 'anagram'
    attacker: dict = field(default_factory=lambda: {'id': None, 'provider': None})


class PUZZLEDAttack(Attack):
    """Run the preserved JBF PUZZLED generator with explicit hosted model routes."""

    REQUIRES_HOSTED_TARGET = True

    def __init__(self, config):
        self.config = normalize_config(config)

    def run(self, target, dataset):
        return run_attack(self, target, dataset)

    @staticmethod
    def validate_config(model, config):
        validate_routes(model, config, ('attacker',))
        if config.get('source_profile') != 'gpt-4o':
            raise ValueError('PUZZLED source_profile must retain gpt-4o.')
        if config.get('puzzle_type') not in {'anagram', 'word_search', 'crossword'}:
            raise ValueError("PUZZLED puzzle_type must be 'anagram', 'word_search', or 'crossword'.")
        if 'generation_config' in config:
            raise ValueError('PUZZLED retains JBF generation settings.')

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
        clue_llm = make_jbf_query_client(transport, 'attacker', config['source_profile'], sleep or time.sleep)
        target = make_jbf_query_client(transport, 'target', config['source_profile'], sleep or time.sleep)
        source = load_symbols('puzzled/puzzled_pipeline.py', ['PuzzledGen'], {
            'ModernBaseAttack': object, 'AttackParameter': lambda **kwargs: kwargs,
            'LLMLiteLLM': SimpleNamespace(from_config=lambda **_kw: clue_llm),
            'Dict': Dict, 'Any': Any, 'Optional': Optional, 'List': List, 'Tuple': Tuple,
            'random': random, 'logger': logging.getLogger(__name__),
        })['PuzzledGen']

        nlp = None
        try:
            import spacy
            nlp = spacy.load('en_core_web_sm')
        except (ImportError, OSError):
            nlp = None

        records = []
        clue_cache = {}
        for question, reference in transport.examples(questions):
            author = source.__new__(source)
            author.puzzle_type = config['puzzle_type']
            author.clue_model = {'model': config['source_profile'], 'provider': 'openai'}
            author.clue_cache = clue_cache
            author.nlp = nlp
            prompt = author.generate_attack(prompt=question, goal=question, target=reference)
            records.append({
                'question': question,
                'puzzle_type': config['puzzle_type'],
                'prompt': prompt,
                'response': str(target.query(prompt)),
            })
        return records, {}
