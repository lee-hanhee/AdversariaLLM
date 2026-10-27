"""Bind the preserved FlipAttack driver to repository I/O."""

import ast
import time
from types import SimpleNamespace

from ...author_execution import AUTHOR_UTILS, load_symbols


def execute(transport, model, config, questions, sleep):
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
    args = SimpleNamespace(**config, victim_llm=model['author_model'], judge_llm=config['judge_model'],
                           eval=config['evaluate'], begin=0, end=len(questions))
    victim = llm(model_id=model['author_model'], **{key: config[key] for key in (
        'temperature', 'max_token', 'retry_time', 'failed_sleep_time', 'round_sleep_time')})

    # Execute the unchanged per-example loop, binding only dataset/progress and result storage.
    source = ast.parse((AUTHOR_UTILS / 'flip_attack' / 'flip_attack_driver.py').read_bytes())
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
