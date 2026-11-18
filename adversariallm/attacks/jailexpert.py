"""Hosted binding for Jailbreak Foundry's JailExpert implementation."""

import base64
import codecs
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
import logging
import os
import pickle
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .attack import Attack
from .utils.author_runtime import load_symbols, normalize_config, run_attack, validate_routes
from .utils.jbf_runtime import make_jbf_query_client


@dataclass
class JailExpertConfig:
    name: str = 'jailexpert'
    type: str = 'discrete'
    version: str = '0.1.0'
    seed: None = None
    # JBF attacker default label; hosted OpenRouter routes are configured separately.
    source_profile: str = 'gpt-3.5-turbo'
    experience_path: str = 'attacks_paper_info/2508.19292/JailExpert/experience/experiences_examples.json'
    n_clusters: int = 10
    top_k: int = 1
    scalar_weight: float = 0.1
    semantic_weight: float = 0.9
    attacker: dict = field(default_factory=lambda: {'id': None, 'provider': None})
    embedding: dict = field(default_factory=lambda: {'id': None, 'provider': None})


class JailExpertAttack(Attack):
    """Run the preserved JBF JailExpert generator with explicit hosted model & embedding routes."""

    REQUIRES_HOSTED_TARGET = True

    def __init__(self, config):
        self.config = normalize_config(config)

    def run(self, target, dataset):
        return run_attack(self, target, dataset)

    @staticmethod
    def validate_config(model, config):
        validate_routes(model, config, ('attacker', 'embedding'))
        if config.get('source_profile') != 'gpt-3.5-turbo':
            raise ValueError('JailExpert source_profile must retain gpt-3.5-turbo.')
        for key in ('n_clusters', 'top_k'):
            if type(config.get(key)) is not int or config[key] < 1:
                raise ValueError(f'JailExpert {key} must be an integer >= 1.')
        for key in ('scalar_weight', 'semantic_weight'):
            val = config.get(key)
            if type(val) not in (int, float) or val < 0:
                raise ValueError(f'JailExpert {key} must be >= 0.')
        if 'generation_config' in config:
            raise ValueError('JailExpert retains JBF generation settings.')

    @staticmethod
    def _add_steps(run, record, trace, duration):
        request = next(item for item in reversed(trace) if item['role'] == 'target' and 'metadata' in item)
        run.add_step(request, messages=request['request']['messages'],
                     completions=[record['response']], raw_completions=[request['raw_text']],
                     time_taken=duration, author_result=record, author_requests=trace)

    @staticmethod
    def _execute(transport, model, config, questions, sleep=None):
        transport.routes['attacker'] = config['attacker']
        transport.routes['embedding'] = config['embedding']
        transport.client = transport.client.with_options(max_retries=0)
        attack_llm = make_jbf_query_client(transport, 'attacker', config['source_profile'], sleep or time.sleep)
        target = make_jbf_query_client(transport, 'target', config['source_profile'], sleep or time.sleep)
        emb_client = transport.for_role('embedding')

        symbols = ['JailbreakState', 'MutationEngine', 'ExperienceIndex', 'JailExpertAttack']
        ns = load_symbols('jailexpert/jailexpert_pipeline.py', symbols, {
            'ModernBaseAttack': object, 'AttackParameter': lambda **kwargs: kwargs, 'LLMLiteLLM': object,
            'os': os, 'json': json, 'logging': logging, 'pickle': pickle, 'np': np,
            'Dict': Dict, 'Any': Any, 'List': List, 'Optional': Optional, 'Tuple': Tuple,
            'defaultdict': defaultdict, 'Counter': Counter, 'dataclass': dataclass,
            'base64': base64, 'codecs': codecs, 'logger': logging.getLogger(__name__),
        })
        cls = ns['JailExpertAttack']
        mutation_cls = ns['MutationEngine']
        index_cls = ns['ExperienceIndex']

        def embedding_call(model=None, input=None, **_kwargs):
            resp = emb_client.embeddings.create(model=model, input=input)
            return {'data': [{'embedding': item.embedding} for item in resp.data]}

        prev_litellm = sys.modules.get('litellm')
        sys.modules['litellm'] = SimpleNamespace(embedding=embedding_call)
        try:
            author = cls.__new__(cls)
            author.experience_path = config['experience_path']
            author.embedding_model_name = {'model': config['embedding']['id'], 'provider': 'openai'}
            author.attack_model_name = {'model': config['source_profile'], 'provider': 'openai'}
            author.n_clusters = config['n_clusters']
            author.top_k = config['top_k']
            author.scalar_weight = config['scalar_weight']
            author.semantic_weight = config['semantic_weight']
            author.embedding_llm = SimpleNamespace(
                model_name=config['embedding']['id'], api_key='bound', api_base=None,
            )
            author.attack_llm = attack_llm
            author.mutation_engine = mutation_cls(attack_llm)
            author.experience_index = index_cls(
                scalar_weight=config['scalar_weight'],
                semantic_weight=config['semantic_weight'],
                max_clusters=config['n_clusters'],
            )
            author._load_and_index_experiences()
            author.current_prompt = None
            author.sorted_clusters = []
            author.query_embedding = None

            records = []
            for question, reference in transport.examples(questions):
                prompt = author.generate_attack(prompt=question, goal=question, target=reference)
                records.append({
                    'question': question,
                    'prompt': prompt,
                    'response': str(target.query(prompt)),
                })
            return records, {'setup_requests': transport.setup_requests}
        finally:
            if prev_litellm is None:
                sys.modules.pop('litellm', None)
            else:
                sys.modules['litellm'] = prev_litellm
