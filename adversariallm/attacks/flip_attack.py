"""Repository binding for the original FlipAttack driver, clients, and evaluator."""

from dataclasses import dataclass, field
from types import SimpleNamespace
import ast
import time

from .attack import Attack
from .utils.author_runtime import (
    AUTHOR_IMPLEMENTATIONS, load_symbols, normalize_config, run_attack, validate_routes,
)


@dataclass
class FlipAttackConfig:
    # Keep repository identification, but do not introduce an RNG seed.
    name: str = 'flip_attack'
    type: str = 'discrete'
    version: str = '0.2.0'
    seed: None = None

    # Select original model handling independently of the hosted model ID.
    source_profile: str = 'gpt-4-0613'

    # Preserve the original CLI and LLM constructor defaults.
    flip_mode: str = 'FCS'
    cot: bool = False
    lang_gpt: bool = False
    few_shot: bool = False
    temperature: float = 0
    max_token: int = -1
    retry_time: int = 1000
    failed_sleep_time: int = 1
    round_sleep_time: int = 1

    # Match the optional --eval switch and the original judge-model default.
    evaluate: bool = False
    judge_model: str = 'gpt-4-0613'

    # Require an explicit OpenRouter model and provider for the author judge.
    judge: dict = field(default_factory=lambda: {'id': None, 'provider': None})


FLIP_MODELS = {
    'gpt-3.5-turbo-0125': 'openai', 'gpt-4-turbo-2024-04-09': 'openai',
    'gpt-4-0613': 'openai', 'gpt-4o-2024-08-06': 'openai',
    'gpt-4o-mini-2024-07-18': 'openai', 'claude-3-5-sonnet-20240620': 'anthropic',
    'Meta-Llama-3.1-405B-Instruct': 'deepinfra', 'Mixtral-8x22B-Instruct-v0.1': 'deepinfra',
}


class FlipAttack(Attack):
    """Run the preserved flip_attack algorithm through the hosted target."""

    REQUIRES_HOSTED_TARGET = True

    def __init__(self, config):
        # Preserve the source algorithm's unseeded random stream.
        self.config = normalize_config(config)

    def run(self, target, dataset):
        return run_attack(self, target, dataset)

    @staticmethod
    def validate_config(model, config):
        validate_routes(model, config, ('judge',))
        if config.get('source_profile') not in FLIP_MODELS:
            raise ValueError('FlipAttack source_profile must be an original CLI model name.')

        # Validate the original flipping modes and guidance switches.
        if config['flip_mode'] not in {'FWO', 'FCW', 'FCS', 'FMM'}:
            raise ValueError('Unknown FlipAttack mode.')
        for option in ('cot', 'lang_gpt', 'few_shot', 'evaluate'):
            if not isinstance(config[option], bool):
                raise ValueError(f'{option} must be boolean.')

        # Keep numeric options consistent with the original CLI argument types.
        for option in ('retry_time', 'failed_sleep_time', 'round_sleep_time', 'max_token'):
            if isinstance(config[option], bool) or not isinstance(config[option], int):
                raise ValueError(f'{option} must match the author CLI integer argument.')

        # Limit the judge source profile to the default supported by this binding.
        if config['judge_model'] != 'gpt-4-0613':
            raise ValueError('FlipAttack judge_model must retain the author default gpt-4-0613.')

    @staticmethod
    def _add_steps(run, record, trace, duration):
        target_request = next(request for request in trace if request['role'] == 'target' and 'metadata' in request)
        scores = {}

        # Copy optional author evaluation fields without adding another evaluator.
        if 'judge_score_gpt4' in record:
            scores['flip_attack_author'] = {key: [float(record[key])] for key in (
                'judge_success_dict', 'judge_score_gpt4', 'judge_success_gpt4')}

        # Pair the processed answer with its raw API response and original call trace.
        run.add_step(target_request, messages=record['all_prompt'],
                     completions=[record['output']], raw_completions=[target_request['raw_text']],
                     scores=scores, time_taken=duration, author_result=record, author_requests=trace)

    @staticmethod
    def _execute(transport, model, config, questions, sleep=None):
        questions = [question for question, _ in questions]
        # Collect original example records and optional dataset-level evaluation counters.
        records, summary = [], {}
        # Bind the original imports and credentials to the existing transport.
        author_os = SimpleNamespace(environ=dict.fromkeys(
            ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'DEEPINFRA_API_KEY'), transport.client.api_key))
        llm = load_symbols('flip_attack/flip_attack_llm.py', ['LLM'], {
            'os': author_os, 'time': SimpleNamespace(sleep=sleep or time.sleep),
            'OpenAI': transport, 'Anthropic': transport, 'anthropic': SimpleNamespace(Anthropic=transport),
        })['LLM']
        evaluator = load_symbols('flip_attack/flip_attack_evaluator.py', ['Evaluator'], {'re': __import__('re'), 'LLM': llm})['Evaluator']
        builder = load_symbols('flip_attack/flip_attack.py', ['FlipAttack'], {'textwrap': __import__('textwrap')})['FlipAttack']

        # Translate repository settings into the original driver argument names.
        args = SimpleNamespace(**config, victim_llm=config['source_profile'], judge_llm=config['judge_model'],
                               eval=config['evaluate'], begin=0, end=len(questions))
        victim = llm(model_id=config['source_profile'], **{key: config[key] for key in (
            'temperature', 'max_token', 'retry_time', 'failed_sleep_time', 'round_sleep_time')})

        # Execute the unchanged per-example loop, binding only dataset/progress and result storage.
        source = ast.parse((AUTHOR_IMPLEMENTATIONS / 'flip_attack' / 'flip_attack_driver.py').read_bytes())
        main = next(node for node in source.body if isinstance(node, ast.If))
        loop = next(node for node in main.body if isinstance(node, ast.For))

        # Replace dataset iteration and output storage while retaining the original loop body.
        namespace = {'args': args, 'adv_bench': {'goal': questions}, 'FlipAttack': builder,
                     'victim_llm': victim, 'Evaluator': evaluator, 'result_dict_list': records,
                     'all_count': 0, 'dict_success_count': 0, 'gpt_success_count': 0,
                     'tqdm': transport.examples}
        exec(compile(ast.Module(body=[loop], type_ignores=[]), 'author-flip-loop', 'exec'), namespace)

        # Expose counters computed by the author only when evaluation is enabled.
        if config['evaluate']:
            summary = {key: namespace[key] for key in ('all_count', 'dict_success_count', 'gpt_success_count')}

        return records, summary
