"""Hosted binding for Jailbreak Foundry's TOMBRAIDER implementation."""

from dataclasses import dataclass, field
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from .attack import Attack
from .utils.author_runtime import load_symbols, normalize_config, run_attack, validate_routes
from .utils.jbf_runtime import make_jbf_query_client


@dataclass
class TombraiderConfig:
    name: str = 'tombraider'
    type: str = 'discrete'
    version: str = '0.1.0'
    seed: None = None
    # JBF attacker/inspector default label; hosted OpenRouter routes are configured separately.
    source_profile: str = 'gpt-4o'
    max_rounds: int = 5
    keyword: Optional[str] = None
    attacker: dict = field(default_factory=lambda: {'id': None, 'provider': None})
    judge: dict = field(default_factory=lambda: {'id': None, 'provider': None})


class TombraiderAttack(Attack):
    """Run the preserved JBF TOMBRAIDER multi-turn loop with explicit hosted model routes."""

    REQUIRES_HOSTED_TARGET = True

    def __init__(self, config):
        self.config = normalize_config(config)

    def run(self, target, dataset):
        return run_attack(self, target, dataset)

    @staticmethod
    def validate_config(model, config):
        validate_routes(model, config, ('attacker', 'judge'))
        if config.get('source_profile') != 'gpt-4o':
            raise ValueError('TOMBRAIDER source_profile must retain gpt-4o.')
        if type(config.get('max_rounds')) is not int or config['max_rounds'] < 1:
            raise ValueError('TOMBRAIDER max_rounds must be an integer >= 1.')
        kw = config.get('keyword')
        if kw is not None and (not isinstance(kw, str) or not kw.strip()):
            raise ValueError('TOMBRAIDER keyword must be None or a non-empty string.')
        if 'generation_config' in config:
            raise ValueError('TOMBRAIDER retains JBF generation settings.')

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
        transport.routes['judge'] = config['judge']
        transport.client = transport.client.with_options(max_retries=0)
        attacker_llm = make_jbf_query_client(transport, 'attacker', config['source_profile'], sleep or time.sleep)
        inspector_llm = make_jbf_query_client(transport, 'judge', config['source_profile'], sleep or time.sleep)
        target_llm = make_jbf_query_client(transport, 'target', config['source_profile'], sleep or time.sleep)

        source = load_symbols('tombraider/tombraider_pipeline.py', ['TombraiderGenAttack'], {
            'ModernBaseAttack': object, 'AttackParameter': lambda **kwargs: kwargs, 'LLMLiteLLM': object,
            'Dict': Dict, 'Any': Any, 'Optional': Optional, 'List': List,
        })['TombraiderGenAttack']

        records = []
        for question, reference in transport.examples(questions):
            author = source.__new__(source)
            author.args = SimpleNamespace(model=config['source_profile'], provider='openai', api_key=None, api_base=None)
            author.kwargs = {}
            author.inspector_model = {'model': config['source_profile'], 'provider': 'openai'}
            author.attacker_model = {'model': config['source_profile'], 'provider': 'openai'}
            author.max_rounds = config['max_rounds']
            author.keyword = config.get('keyword')
            author.inspector_llm = inspector_llm
            author.attacker_llm = attacker_llm
            author.target_llm = target_llm
            author._initialize_llms = lambda **_kw: None
            prompt = author.generate_attack(prompt=question, goal=question, target=reference)
            records.append({
                'question': question,
                'prompt': prompt,
                'response': str(target_llm.query(prompt)),
            })
        return records, {}
