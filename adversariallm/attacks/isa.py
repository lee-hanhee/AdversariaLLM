"""Hosted binding for Jailbreak Foundry's ISA (Intent Shift Attack) implementation."""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, Optional

from .attack import Attack
from .utils.author_runtime import load_symbols, normalize_config, run_attack, validate_routes
from .utils.jbf_runtime import make_jbf_query_client


@dataclass
class ISAConfig:
    name: str = 'isa'
    type: str = 'discrete'
    version: str = '0.1.0'
    seed: None = None
    # JBF helper default label; hosted OpenRouter routes are configured separately.
    source_profile: str = 'gpt-4o'
    shift_type: str = 'question'
    attacker: dict = field(default_factory=lambda: {'id': None, 'provider': None})


class ISAAttack(Attack):
    """Run the preserved JBF ISA generator with explicit hosted model routes."""

    REQUIRES_HOSTED_TARGET = True

    def __init__(self, config):
        self.config = normalize_config(config)

    def run(self, target, dataset):
        return run_attack(self, target, dataset)

    @staticmethod
    def validate_config(model, config):
        validate_routes(model, config, ('attacker',))
        if config.get('source_profile') != 'gpt-4o':
            raise ValueError('ISA source_profile must retain gpt-4o.')
        if config.get('shift_type') not in {'person', 'tense', 'voice', 'mood', 'question'}:
            raise ValueError("ISA shift_type must be one of 'person', 'tense', 'voice', 'mood', or 'question'.")
        if 'generation_config' in config:
            raise ValueError('ISA retains JBF generation settings.')

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
        helper = make_jbf_query_client(
            transport, 'attacker', config['source_profile'], sleep or time.sleep, temperature=0.0,
        )
        target = make_jbf_query_client(transport, 'target', config['source_profile'], sleep or time.sleep)
        source = load_symbols('isa/isa_pipeline.py', ['NORMALIZATION_PROMPT', 'SHIFT_PROMPTS', 'ISAGen'], {
            'ModernBaseAttack': object, 'AttackParameter': lambda **kwargs: kwargs,
            'Dict': Dict, 'Any': Any, 'Optional': Optional,
        })['ISAGen']

        records = []
        for question, reference in transport.examples(questions):
            author = source.__new__(source)
            author.helper_llm = helper
            author.get_parameter_value = config.__getitem__
            prompt = author.generate_attack(prompt=question, goal=question, target=reference)
            records.append({
                'question': question,
                'shift_type': config['shift_type'],
                'prompt': prompt,
                'response': str(target.query(prompt)),
            })
        return records, {}
