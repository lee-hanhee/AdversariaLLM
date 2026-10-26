"""Execute unchanged author code using the shared runtime SDK."""

import ast
import copy
from importlib.metadata import version
import os
from pathlib import Path
import time
from types import SimpleNamespace

AUTHOR_METHODS = frozenset({'flip_attack'})
FLIP_MODELS = {
    'gpt-3.5-turbo-0125': 'openai', 'gpt-4-turbo-2024-04-09': 'openai',
    'gpt-4-0613': 'openai', 'gpt-4o-2024-08-06': 'openai',
    'gpt-4o-mini-2024-07-18': 'openai', 'claude-3-5-sonnet-20240620': 'anthropic',
    'Meta-Llama-3.1-405B-Instruct': 'deepinfra', 'Mixtral-8x22B-Instruct-v0.1': 'deepinfra',
}


def validate_author_config(method, model, config):
    # Match argparse's rejection of unknown switches instead of silently ignoring overrides.
    common = {'name', 'type', 'version', 'seed'}
    options = {'flip_mode', 'cot', 'lang_gpt', 'few_shot', 'temperature', 'max_token',
                'retry_time', 'failed_sleep_time', 'round_sleep_time', 'evaluate', 'judge_model', 'judge'}
    if set(config) - common - options:
        raise ValueError(f'Unsupported author settings: {set(config) - common - options}')

    # Routing IDs are explicit and independent of names used by the author logic.
    if model.get('backend') != 'openrouter':
        raise ValueError(f'{method} requires backend=openrouter.')
    for role, route in [('target', model), ('judge', config.get('judge') or {})]:
        for field in ('id', 'provider'):
            value = route.get(field)
            if not isinstance(value, str) or not value.strip() or value == '???':
                raise ValueError(f'{role} requires an explicit OpenRouter {field}.')
    if config.get('seed') is not None or 'generation_config' in config:
        raise ValueError('Author execution does not seed RNG or accept generic generation_config overrides.')
    if model.get('base_url') or 'judge_base_url' in config:
        raise ValueError('Author transport uses only https://openrouter.ai/api/v1.')

    # Preserve the original CLI model choices and request controls.
    if method == 'flip_attack':
        if model.get('author_model') not in FLIP_MODELS:
            raise ValueError('FlipAttack author_model must be an original CLI model name.')
        if config['flip_mode'] not in {'FWO', 'FCW', 'FCS', 'FMM'}:
            raise ValueError('Unknown FlipAttack mode.')
        for option in ('cot', 'lang_gpt', 'few_shot', 'evaluate'):
            if not isinstance(config[option], bool):
                raise ValueError(f'{option} must be boolean.')
        for option in ('retry_time', 'failed_sleep_time', 'round_sleep_time', 'max_token'):
            if isinstance(config[option], bool) or not isinstance(config[option], int):
                raise ValueError(f'{option} must match the author CLI integer argument.')
        if config['judge_model'] != 'gpt-4-0613':
            raise ValueError('FlipAttack judge_model must retain the author default gpt-4-0613.')

    else:
        raise ValueError(f'Unknown author method: {method}')


DATA = Path(__file__).resolve().parent / 'data'


def load_symbols(filename, names, bindings):
    # Execute only declared author definitions; bind imports and storage I/O explicitly.
    path = DATA / filename
    tree = ast.parse(path.read_bytes())
    nodes = [node for node in tree.body if (
        isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
        or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets))]
    namespace = dict(bindings)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return {name: namespace[name] for name in names}


def execute(payload, *, sdk_factory=None, sleep=None):
    method, model, config = payload['method'], payload['model'], payload['config']
    validate_author_config(method, model, config)

    environment = {'openai': version('openai')}
    key = os.environ['OPENROUTER_API_KEY']
    requests, durations = [], {}
    state = {'index': -1, 'phase': 'target'}

    def client(provider, **options):
        # Original constructors are I/O bindings only; no original credential or endpoint is used.
        options = {'api_key': key, 'base_url': 'https://openrouter.ai/api/v1'}
        if sdk_factory:
            native = sdk_factory('openai', options)
        else:
            from openai import OpenAI
            native = OpenAI(**options)

        def create(**kwargs):
            role = state['phase']
            route = model if role == 'target' else config['judge']
            record = {'example': state['index'], 'role': role,
                      'endpoint': str(native.base_url), 'request': copy.deepcopy(kwargs)}
            requests.append(record)

            # Translate Anthropic's system field to a chat message without changing its text.
            routed = copy.deepcopy(kwargs)
            routed['model'] = route['id']
            if provider == 'anthropic':
                routed['messages'] = [{'role': 'system', 'content': routed.pop('system')}, *routed['messages']]
            routed['extra_body'] = {'provider': {'only': [route['provider']],
                                                 'allow_fallbacks': False, 'require_parameters': True}}
            record['routed_request'] = copy.deepcopy(routed)
            try:
                # The author owns application retries; the SDK retains its own normal retry behavior.
                response = native.chat.completions.create(**routed)
                usage = getattr(response, 'usage', None)
                record['metadata'] = {'id': response.id, 'model': response.model,
                                      'usage': usage.model_dump() if usage else None,
                                      'finish_reasons': [choice.finish_reason for choice in response.choices],
                                      'provider': getattr(response, 'provider', None)}
                record['raw_text'] = response.choices[0].message.content
                state['phase'] = 'judge'
                if provider == 'anthropic':
                    # Return the shape expected by the untouched Claude response processing.
                    return SimpleNamespace(content=[SimpleNamespace(text=record['raw_text'])])
                return response
            except Exception as error:
                record['error_type'] = type(error).__name__
                raise

        return (SimpleNamespace(messages=SimpleNamespace(create=create)) if provider == 'anthropic'
                else SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))

    # Iteration preserves dataset order and measures time without touching the author's RNG.
    def examples(values):
        for index, value in enumerate(values):
            state.update(index=index, phase='target')
            started = time.perf_counter()
            yield value
            durations[index] = time.perf_counter() - started

    openai_factory = lambda **kwargs: client('openai', **kwargs)
    anthropic_factory = lambda **kwargs: client('anthropic', **kwargs)
    clock = SimpleNamespace(sleep=sleep or time.sleep)
    # Bind original environment lookups locally; never mutate process credentials.
    author_os = SimpleNamespace(environ={'OPENAI_API_KEY': key, 'ANTHROPIC_API_KEY': key, 'DEEPINFRA_API_KEY': key}, path=os.path)
    questions = [conv[0]['content'] for conv in payload['dataset']]
    records, summary = [], {}

    try:
        if method == 'flip_attack':
            # Reuse the original client and evaluator classes, including retries and response processing.
            llm = load_symbols('flip_attack_llm.py', ['LLM'], {
                'os': author_os, 'time': clock, 'OpenAI': openai_factory, 'Anthropic': anthropic_factory,
                'anthropic': SimpleNamespace(Anthropic=anthropic_factory),
            })['LLM']
            import re
            evaluator = load_symbols('flip_attack_evaluator.py', ['Evaluator'], {'re': re, 'LLM': llm})['Evaluator']
            builder = load_symbols('flip_attack.py', ['FlipAttack'], {'textwrap': __import__('textwrap')})['FlipAttack']
            args = SimpleNamespace(**config, victim_llm=model['author_model'], judge_llm=config['judge_model'],
                                   eval=config['evaluate'], begin=0, end=len(questions))
            victim = llm(model_id=model['author_model'], **{key: config[key] for key in (
                'temperature', 'max_token', 'retry_time', 'failed_sleep_time', 'round_sleep_time')})

            # Execute the original per-example loop; replace only dataset/progress and result bindings.
            source = ast.parse((DATA / 'flip_attack_driver.py').read_bytes())
            main = next(node for node in source.body if isinstance(node, ast.If))
            loop = next(node for node in main.body if isinstance(node, ast.For))
            namespace = {'args': args, 'adv_bench': {'goal': questions}, 'FlipAttack': builder,
                         'victim_llm': victim, 'Evaluator': evaluator, 'result_dict_list': records,
                         'all_count': 0, 'dict_success_count': 0, 'gpt_success_count': 0,
                         'tqdm': lambda values: examples(values)}
            exec(compile(ast.Module(body=[loop], type_ignores=[]), 'author-flip-loop', 'exec'), namespace)
            if config['evaluate']:
                summary = {key: namespace[key] for key in ('all_count', 'dict_success_count', 'gpt_success_count')}
        # Map artifacts only after the unchanged author logic finishes successfully.
        runs = []
        for index, record in enumerate(records):
            trace = [request for request in requests if request['example'] == index]
            target = next(request for request in trace if request['role'] == 'target' and 'metadata' in request)
            runs.append({'author_result': record, 'requests': trace, 'messages': record['all_prompt'],
                         'response': record['output'],
                         'raw_response': target['raw_text'], 'api_metadata': target['metadata'],
                         'time_taken': durations[index]})
        return {'runs': runs, 'author_summary': summary, 'environment': environment}
    except Exception as error:
        error.author_requests = requests
        raise
