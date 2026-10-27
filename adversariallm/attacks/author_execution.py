"""Execute unchanged author code using the shared runtime SDK."""

import ast
import copy
from importlib.metadata import version
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
    if config.get('seed') is not None:
        raise ValueError('Author execution does not seed RNG.')
    if model.get('base_url'):
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


AUTHOR_UTILS = Path(__file__).resolve().parent / 'utils'


def load_symbols(filename, names, bindings):
    # Execute only declared author definitions; bind imports and storage I/O explicitly.
    path = AUTHOR_UTILS / filename
    tree = ast.parse(path.read_bytes())
    nodes = [node for node in tree.body if (
        isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
        or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets))]
    namespace = dict(bindings)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return {name: namespace[name] for name in names}


class OpenRouterAuthorClient:
    """Bind author client interfaces to one SDK client and retain per-example traces."""

    def __init__(self, client, model, judge):
        self.client, self.model, self.judge = client, model, judge
        self.requests, self.durations = [], []
        self.index, self.phase = -1, 'target'

        # These are the only OpenAI/Anthropic client interfaces used by the author code.
        self.chat = SimpleNamespace(completions=self)
        self.messages = self

    def __call__(self, **_options):
        # Original constructor credentials/endpoints are replaced by the existing hosted client.
        return self

    def create(self, **kwargs):
        route = self.model if self.phase == 'target' else self.judge
        record = {'example': self.index, 'role': self.phase, 'endpoint': str(self.client.base_url),
                  'request': copy.deepcopy(kwargs)}
        self.requests[self.index].append(record)

        # Claude has a separate system field and expects a different response shape.
        routed = copy.deepcopy(kwargs)
        anthropic = 'system' in routed
        if anthropic:
            routed['messages'] = [{'role': 'system', 'content': routed.pop('system')}, *routed['messages']]
        routed['model'] = route['id']
        routed['extra_body'] = {'provider': {'only': [route['provider']],
                                             'allow_fallbacks': False, 'require_parameters': True}}
        record['routed_request'] = copy.deepcopy(routed)
        try:
            # Keep SDK retries; all application retries remain in the original author methods.
            response = self.client.chat.completions.create(**routed)
            usage = getattr(response, 'usage', None)
            record['metadata'] = {'id': response.id, 'model': response.model,
                                  'usage': usage.model_dump() if usage else None,
                                  'finish_reasons': [choice.finish_reason for choice in response.choices],
                                  'provider': getattr(response, 'provider', None)}
            record['raw_text'] = response.choices[0].message.content if response.choices else None
            self.phase = 'judge'
            if anthropic:
                content = [SimpleNamespace(text=record['raw_text'])] if response.choices else []
                return SimpleNamespace(content=content)
            return response
        except Exception as error:
            record['error_type'] = type(error).__name__
            raise

    def examples(self, values):
        # Group traces as requests happen, avoiding a later scan of the full request history.
        for self.index, value in enumerate(values):
            self.phase = 'target'
            self.requests.append([])
            started = time.perf_counter()
            yield value
            self.durations.append(time.perf_counter() - started)


def execute(method, model, config, dataset, *, client, sleep=None):
    validate_author_config(method, model, config)
    transport = OpenRouterAuthorClient(client, model, config['judge'])
    questions = [conv[0]['content'] for conv in dataset]
    records = []

    # Reuse original definitions with local I/O bindings; never mutate process credentials.
    author_os = SimpleNamespace(environ=dict.fromkeys(
        ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'DEEPINFRA_API_KEY'), client.api_key))
    llm = load_symbols('flip_attack/flip_attack_llm.py', ['LLM'], {
        'os': author_os, 'time': SimpleNamespace(sleep=sleep or time.sleep),
        'OpenAI': transport, 'Anthropic': transport, 'anthropic': SimpleNamespace(Anthropic=transport),
    })['LLM']
    evaluator = load_symbols('flip_attack/flip_attack_evaluator.py', ['Evaluator'], {'re': __import__('re'), 'LLM': llm})['Evaluator']
    builder = load_symbols('flip_attack/flip_attack.py', ['FlipAttack'], {'textwrap': __import__('textwrap')})['FlipAttack']
    args = SimpleNamespace(**config, victim_llm=model['author_model'], judge_llm=config['judge_model'],
                           eval=config['evaluate'], begin=0, end=len(questions))
    victim = llm(model_id=model['author_model'], **{key: config[key] for key in (
        'temperature', 'max_token', 'retry_time', 'failed_sleep_time', 'round_sleep_time')})

    # Execute the unchanged per-example loop, binding only dataset/progress and result storage.
    source = ast.parse((AUTHOR_UTILS / 'flip_attack' / 'flip_attack_driver.py').read_bytes())
    main = next(node for node in source.body if isinstance(node, ast.If))
    loop = next(node for node in main.body if isinstance(node, ast.For))
    namespace = {'args': args, 'adv_bench': {'goal': questions}, 'FlipAttack': builder,
                 'victim_llm': victim, 'Evaluator': evaluator, 'result_dict_list': records,
                 'all_count': 0, 'dict_success_count': 0, 'gpt_success_count': 0,
                 'tqdm': transport.examples}
    try:
        exec(compile(ast.Module(body=[loop], type_ignores=[]), 'author-flip-loop', 'exec'), namespace)
    except Exception as error:
        error.author_requests = [request for trace in transport.requests for request in trace]
        raise
    summary = ({key: namespace[key] for key in ('all_count', 'dict_success_count', 'gpt_success_count')}
               if config['evaluate'] else {})
    return {'records': records, 'requests': transport.requests, 'durations': transport.durations,
            'author_summary': summary, 'environment': {'openai': version('openai')}}
