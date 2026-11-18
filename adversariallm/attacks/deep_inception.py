"""Hosted binding for Jailbreak Foundry's DeepInception implementation."""

from dataclasses import dataclass
import time
from typing import Any, Dict, Optional

from .attack import Attack
from .utils.author_runtime import load_symbols, normalize_config, run_attack, validate_routes
from .utils.jbf_runtime import make_jbf_query_client


@dataclass
class DeepInceptionConfig:
    name: str = 'deep_inception'
    type: str = 'discrete'
    version: str = '0.1.0'
    seed: None = None
    # JBF target default label; hosted OpenRouter routes are configured separately.
    source_profile: str = 'gpt-4'
    scene_type: str = 'science fiction'
    character_number: int = 5
    layer_number: int = 5


class DeepInceptionAttack(Attack):
    """Run the preserved JBF DeepInception generator with explicit hosted model routes."""

    REQUIRES_HOSTED_TARGET = True

    def __init__(self, config):
        self.config = normalize_config(config)

    def run(self, target, dataset):
        return run_attack(self, target, dataset)

    @staticmethod
    def validate_config(model, config):
        validate_routes(model, config, ())
        if config.get('source_profile') != 'gpt-4':
            raise ValueError('DeepInception source_profile must retain gpt-4.')
        scene = config.get('scene_type')
        if not isinstance(scene, str) or not scene.strip():
            raise ValueError('DeepInception scene_type must be a non-empty string.')
        if type(config.get('character_number')) is not int or config['character_number'] < 1:
            raise ValueError('DeepInception character_number must be an integer >= 1.')
        if type(config.get('layer_number')) is not int or config['layer_number'] < 1:
            raise ValueError('DeepInception layer_number must be an integer >= 1.')
        if 'generation_config' in config:
            raise ValueError('DeepInception retains JBF generation settings.')

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
        source = load_symbols('deep_inception/deep_inception_pipeline.py', ['DeepInceptionAttack'], {
            'ModernBaseAttack': object, 'AttackParameter': lambda **kwargs: kwargs,
            'Dict': Dict, 'Any': Any, 'Optional': Optional,
        })['DeepInceptionAttack']

        records = []
        for question, reference in transport.examples(questions):
            author = source.__new__(source)
            author.get_parameter_value = config.__getitem__
            prompt = author.generate_attack(prompt=question, goal=question, target=reference)
            records.append({
                'question': question,
                'prompt': prompt,
                'response': str(target.query(prompt)),
            })
        return records, {}
