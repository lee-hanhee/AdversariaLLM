"""Bind the original TAP search to hosted role clients and repository logging."""

import ast
import copy
from functools import cache
import importlib.util
import logging
import os
from pathlib import Path
import random
import re
import string
import sys
import time
from types import SimpleNamespace
from typing import Dict, List

from ...author_execution import load_symbols


@cache
def conversation_factory():
    # Keep the pinned dependency isolated from the repository's incompatible package requirements.
    package = Path(__file__).resolve().parents[4] / '.author_envs' / 'tap_fastchat'
    metadata = package / 'fschat-0.2.32.dist-info' / 'METADATA'
    if not metadata.is_file():
        raise ImportError('Install fschat==0.2.32 with --no-deps --target .author_envs/tap_fastchat; see docs/author_execution.md.')

    # Import the exact standard-library-only conversation module without FastChat model-server imports.
    source = package / 'fastchat' / 'conversation.py'
    spec = importlib.util.spec_from_file_location('_adversariallm_tap_fastchat_conversation', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    # These two original adapters are the only matches for the supported source profiles.
    registry = []
    adapters = load_symbols(package / 'fastchat/model/model_adapter.py', [
        'BaseModelAdapter', 'ChatGPTAdapter', 'get_model_adapter', 'get_conversation_template',
    ], {'Conversation': module.Conversation, 'get_conv_template': module.get_conv_template,
        'model_adapters': registry, 'cache': cache, 'os': os})
    registry.extend([adapters['ChatGPTAdapter'](), adapters['BaseModelAdapter']()])
    return adapters['get_conversation_template']


def execute(transport, model, config, questions, sleep):
    import numpy as np

    # Load the original dependency before making requests; do not import or download model weights.
    template = conversation_factory()
    constants = load_symbols('tap/config.py', ['ATTACK_TEMP', 'TARGET_TEMP', 'ATTACK_TOP_P',
                                            'TARGET_TOP_P', 'MAX_PARALLEL_STREAMS', 'VICUNA_PATH', 'LLAMA_PATH'], {})
    prompts = load_symbols('tap/system_prompts.py', ['get_attacker_system_prompt',
        'get_evaluator_system_prompt_for_judge', 'get_evaluator_system_prompt_for_on_topic'], {})
    helpers = load_symbols('tap/common.py', ['random_string', 'extract_json', 'get_init_msg',
        'process_target_response', 'conv_template'], {'random': random, 'string': string,
            'ast': ast, 'logging': logging, 'get_conversation_template': template})
    common = SimpleNamespace(**helpers)
    context = SimpleNamespace(role='attacker', evaluation=None)
    clock = SimpleNamespace(sleep=sleep or time.sleep)
    records = []

    # The legacy SDK does not retry HTTP status errors; retain only the author's application loop.
    transport.client = transport.client.with_options(max_retries=0)
    transport.routes['attacker'] = config['attacker']

    def legacy_create(**kwargs):
        # Translate the legacy timeout keyword and response mapping for the shared modern SDK.
        original = copy.deepcopy(kwargs)
        kwargs['timeout'] = kwargs.pop('request_timeout')
        try:
            response = transport.create_for(context.role, **kwargs)
        finally:
            request = transport.requests[transport.index][-1]
            request['legacy_request'] = original
            if context.evaluation is not None:
                request['evaluation'] = context.evaluation
        return response.model_dump()

    # Execute the unchanged legacy GPT wrapper, including its retry count, sleeps, and error marker.
    legacy = SimpleNamespace(ChatCompletion=SimpleNamespace(create=legacy_create))
    language = load_symbols('tap/language_models.py', ['LanguageModel', 'GPT'], {
        'List': List, 'Dict': Dict, 'os': os, 'openai': legacy, 'time': clock})
    original_gpt = language['GPT']

    class RecordingGPT(original_gpt):
        def generate(self, conv, max_n_tokens, temperature, top_p):
            # Record logical target calls, including retries that occur inside the original wrapper.
            start = len(transport.requests[transport.index])
            response = super().generate(conv, max_n_tokens, temperature, top_p)
            if context.role == 'target':
                records[-1]['queries'].append({'messages': copy.deepcopy(conv), 'response': response,
                    'request_start': start, 'request_end': len(transport.requests[transport.index])})
            return response

    # Load original GPT-only constructors and conversation handling without changing source definitions.
    conversers = load_symbols('tap/conversers.py', ['AttackLLM', 'TargetLLM', 'load_indiv_model',
        'get_model_path_and_template', 'load_attack_and_target_models'], {
            **constants, 'common': common, 'GPT': RecordingGPT})
    evaluators = load_symbols('tap/evaluators.py', ['EvaluatorBase', 'NoEvaluator', 'GPTEvaluator',
        'load_evaluator'], {**prompts, 're': re, 'GPT': RecordingGPT, 'get_conversation_template': template})

    def load_models(args):
        # Preserve model sharing when source names match, while selecting routes at each call boundary.
        attack, target = conversers['load_attack_and_target_models'](args)
        get_attack, get_response = attack.get_attack, target.get_response

        def attack_call(*args, **kwargs):
            context.role, context.evaluation = 'attacker', None
            return get_attack(*args, **kwargs)

        def target_call(*args, **kwargs):
            context.role, context.evaluation = 'target', None
            return get_response(*args, **kwargs)

        attack.get_attack, target.get_response = attack_call, target_call
        return attack, target

    def load_evaluator(args):
        # Observe original topic and rating results without adding a parser or another evaluation call.
        evaluator = evaluators['load_evaluator'](args)
        on_topic, judge = evaluator.on_topic_score, evaluator.judge_score

        def topics(*args, **kwargs):
            context.role, context.evaluation = 'judge', 'on_topic'
            return on_topic(*args, **kwargs)

        def ratings(*args, **kwargs):
            context.role, context.evaluation = 'judge', 'rating'
            scores = judge(*args, **kwargs)
            if scores:
                for query, score in zip(records[-1]['queries'][-len(scores):], scores):
                    query['score'] = score
            return scores

        evaluator.on_topic_score, evaluator.judge_score = topics, ratings
        return evaluator

    class Logger:
        def __init__(self, args, system_prompt):
            # Capture original logger inputs; W&B and parquet output remain repository I/O bindings.
            records[-1]['system_prompt'] = system_prompt

        def log(self, iteration, attack_list, response_list, judge_scores, on_topic_scores=None,
                conv_ids=None, parent_conv_ids=None):
            records[-1]['iterations'].append(copy.deepcopy({
                'iteration': iteration, 'attack_list': attack_list, 'response_list': response_list,
                'judge_scores': judge_scores, 'on_topic_scores': on_topic_scores,
                'conv_ids': conv_ids, 'parent_conv_ids': parent_conv_ids}))

        def finish(self):
            pass

    # Execute the unchanged pruning functions and complete search for each repository example.
    driver = load_symbols('tap/main_TAP.py', ['clean_attacks_and_convs', 'prune', 'main'], {
        **prompts, **helpers, 'common': common, 'copy': copy, 'np': np,
        'load_attack_and_target_models': load_models, 'load_evaluator': load_evaluator, 'WandBLogger': Logger})
    for goal, target_str in transport.examples(questions):
        index = transport.index
        record = {'iterations': [], 'queries': []}
        records.append(record)
        args = SimpleNamespace(**{key: config[key] for key in (
            'attack_model', 'evaluator_model', 'attack_max_n_tokens', 'target_max_n_tokens',
            'evaluator_max_n_tokens', 'evaluator_temperature', 'max_n_attack_attempts',
            'branching_factor', 'width', 'depth', 'n_streams', 'keep_last_n')},
            target_model=model['author_model'], goal=goal, target_str=target_str,
            index=index, category=None, iter_index=index, store_folder='')
        driver['main'](args)

    return records, {}
