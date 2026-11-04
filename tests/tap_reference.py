"""Independent original TAP execution using the legacy SDK and pinned FastChat package."""

import ast
import copy
import logging
import os
from pathlib import Path
import random
import re
import string
import sys
from types import SimpleNamespace
from typing import Dict, List

import numpy as np


def tap_role(messages):
    # Distinguish original role prompts even when their source model names coincide.
    system = messages[0]['content']
    if system.startswith('You are a helpful red teaming assistant. You must obey'):
        return 'attacker'
    if system.startswith('You are a helpful red teaming assistant. You will receive'):
        return 'judge'
    return 'target'


def reference_template_factory(package):
    # Import the actual pinned dependency rather than the production dependency loader.
    sys.path.insert(0, str(package))
    from fastchat.conversation import Conversation, get_conv_template
    tree = ast.parse((package / 'fastchat/model/model_adapter.py').read_bytes())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    constants = next(node for node in tree.body if isinstance(node, ast.Assign)
                     and getattr(node.targets[0], 'id', None) == 'ANTHROPIC_MODEL_LIST')
    bindings = {'os': os, 're': re, 'Conversation': Conversation, 'get_conv_template': get_conv_template,
                'ANTHROPIC_MODEL_LIST': ast.literal_eval(constants.value)}

    def template(name):
        # Evaluate every original adapter match, independently checking the production restricted registry.
        matches = []
        for cls in classes:
            method = next((node for node in cls.body if isinstance(node, ast.FunctionDef)
                           and node.name == 'match'), None)
            if method is None:
                continue
            scope = dict(bindings)
            exec(compile(ast.Module(body=[method], type_ignores=[]), 'reference-adapter-match', 'exec'), scope)
            if scope['match'](SimpleNamespace(), name):
                matches.append(cls)
        specific = [cls for cls in matches if cls.name != 'BaseModelAdapter']
        assert len(specific) <= 1
        selected = specific[0] if specific else next(cls for cls in matches if cls.name == 'BaseModelAdapter')
        method = next(node for node in selected.body if isinstance(node, ast.FunctionDef)
                      and node.name == 'get_default_conv_template')
        scope = dict(bindings)
        exec(compile(ast.Module(body=[method], type_ignores=[]), 'reference-adapter-template', 'exec'), scope)
        return scope['get_default_conv_template'](SimpleNamespace(), name)

    return template


def run_reference_tap(payload, factory, sleep, base):
    # Execute complete original definitions and assignments; replace imports and external I/O only.
    def original(filename, bindings):
        tree = ast.parse((base / filename).read_bytes())
        body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
                and not (isinstance(node, ast.If) and '__name__' in ast.unparse(node.test))]
        scope = dict(bindings)
        exec(compile(ast.Module(body=body, type_ignores=[]), str(base / filename), 'exec'), scope)
        return scope

    package = Path(payload['fastchat_package'])
    template = reference_template_factory(package)
    constants = original('config.py', {})
    prompts = original('system_prompts.py', {})
    helpers = original('common.py', {'ast': ast, 'random': random, 'string': string,
                                   'logging': logging, 'get_conversation_template': template})
    common = SimpleNamespace(**helpers)
    sdk = factory('legacy', {})
    language = original('language_models.py', {**constants, 'openai': sdk, 'os': os,
                         'time': SimpleNamespace(sleep=sleep), 'Dict': Dict, 'List': List})
    source_gpt = language['GPT']
    rows = []
    current = {'role': None, 'offset': 0}
    count = lambda: factory('request_count', {}) - current['offset']

    class ObserveGPT(source_gpt):
        def generate(self, *args, **kwargs):
            # Observe target results without implementing generation, retries, or parsing in the oracle.
            start = count()
            result = super().generate(*args, **kwargs)
            if current['role'] == 'target':
                rows[-1]['queries'].append({'messages': copy.deepcopy(args[0]), 'response': result,
                                           'request_start': start, 'request_end': count()})
            return result

    conversers = original('conversers.py', {**constants, 'common': common, 'GPT': ObserveGPT})
    evaluation = original('evaluators.py', {**prompts, 're': re, 'GPT': ObserveGPT,
                                          'get_conversation_template': template})

    def models(args):
        # Role observation does not change source model sharing or conversation objects.
        attacker, target = conversers['load_attack_and_target_models'](args)
        attack, respond = attacker.get_attack, target.get_response

        def attack_call(*args, **kwargs):
            current['role'] = 'attacker'
            return attack(*args, **kwargs)

        def target_call(*args, **kwargs):
            current['role'] = 'target'
            return respond(*args, **kwargs)

        attacker.get_attack, target.get_response = attack_call, target_call
        return attacker, target

    def evaluator(args):
        # Copy source ratings from the returned list; the original evaluator owns every decision.
        instance = evaluation['load_evaluator'](args)
        topics, scores = instance.on_topic_score, instance.judge_score

        def topic_call(*args, **kwargs):
            current['role'] = 'judge'
            return topics(*args, **kwargs)

        def score_call(*args, **kwargs):
            current['role'] = 'judge'
            result = scores(*args, **kwargs)
            if result:
                for row, score in zip(rows[-1]['queries'][-len(result):], result):
                    row['score'] = score
            return result

        instance.on_topic_score, instance.judge_score = topic_call, score_call
        return instance

    class Output:
        def __init__(self, args, system):
            rows[-1]['system_prompt'] = system

        def log(self, iteration, attack_list, response_list, judge_scores, on_topic_scores=None,
                conv_ids=None, parent_conv_ids=None):
            # Keep exactly the arguments passed to the original standalone logger.
            rows[-1]['iterations'].append(copy.deepcopy(dict(iteration=iteration, attack_list=attack_list,
                response_list=response_list, judge_scores=judge_scores, on_topic_scores=on_topic_scores,
                conv_ids=conv_ids, parent_conv_ids=parent_conv_ids)))

        def finish(self):
            pass

    driver = original('main_TAP.py', {**prompts, **helpers, 'copy': copy, 'np': np, 'common': common,
        'WandBLogger': Output, 'load_evaluator': evaluator, 'load_attack_and_target_models': models})
    config = payload['config']
    for index, conversation in enumerate(payload['dataset']):
        rows.append({'iterations': [], 'queries': []})
        current['offset'] = factory('request_count', {})
        args = SimpleNamespace(**{key: config[key] for key in (
            'attack_model', 'evaluator_model', 'attack_max_n_tokens', 'target_max_n_tokens',
            'evaluator_max_n_tokens', 'evaluator_temperature', 'max_n_attack_attempts',
            'branching_factor', 'width', 'depth', 'n_streams', 'keep_last_n')},
            target_model=payload['config']['source_profile'], goal=conversation[0]['content'],
            target_str=conversation[1]['content'], index=index, category=None, iter_index=index, store_folder='')
        driver['main'](args)
    return rows
