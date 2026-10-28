"""Execute unchanged author code using the shared runtime SDK."""

import ast
import copy
from importlib.metadata import version
from pathlib import Path
import time
from types import SimpleNamespace

# List the original-driver integrations available to the experiment runner.
AUTHOR_METHODS = frozenset({'flip_attack', 'equacode', 'gta', 'mousetrap', 'query_attack'})

# Retain the source model names that select FlipAttack's original client branches.
FLIP_MODELS = {
    'gpt-3.5-turbo-0125': 'openai', 'gpt-4-turbo-2024-04-09': 'openai',
    'gpt-4-0613': 'openai', 'gpt-4o-2024-08-06': 'openai',
    'gpt-4o-mini-2024-07-18': 'openai', 'claude-3-5-sonnet-20240620': 'anthropic',
    'Meta-Llama-3.1-405B-Instruct': 'deepinfra', 'Mixtral-8x22B-Instruct-v0.1': 'deepinfra',
}


def validate_author_config(method, model, config):
    # Match argparse's rejection of unknown switches instead of silently ignoring overrides.
    common = {'name', 'type', 'version', 'seed'}
    if method == 'flip_attack':
        options = {'flip_mode', 'cot', 'lang_gpt', 'few_shot', 'temperature', 'max_token',
                   'retry_time', 'failed_sleep_time', 'round_sleep_time', 'evaluate', 'judge_model', 'judge'}

    elif method == 'equacode':
        options = {'variant', 'judge_model', 'judge'}

    elif method == 'gta':
        options = {'rounds', 'attacker', 'judge'}

    elif method == 'mousetrap':
        options = {'judge'}

    elif method == 'query_attack':
        options = {'translation_language', 'target_language', 'trans_verify', 'temperate',
                   'top_p', 'max_new_tokens', 'request_seed', 'translator'}

    else:
        raise ValueError(f'Unknown author method: {method}')

    if set(config) - common - options:
        raise ValueError(f'Unsupported author settings: {set(config) - common - options}')

    # Routing IDs are explicit and independent of names used by the author logic.
    if model.get('backend') != 'openrouter':
        raise ValueError(f'{method} requires backend=openrouter.')
    routes = [('target', model)]
    if method == 'query_attack':
        routes.append(('translator', config.get('translator') or {}))
    else:
        routes.append(('judge', config.get('judge') or {}))
    if method == 'gta':
        routes.append(('attacker', config.get('attacker') or {}))
    for role, route in routes:
        for field in ('id', 'provider'):
            value = route.get(field)
            if not isinstance(value, str) or not value.strip() or value == '???':
                raise ValueError(f'{role} requires an explicit OpenRouter {field}.')

    # Reject configuration changes to the author RNG stream or fixed transport endpoint.
    if config.get('seed') is not None:
        raise ValueError('Author execution does not seed RNG.')
    if model.get('base_url'):
        raise ValueError('Author transport uses only https://openrouter.ai/api/v1.')

    # Preserve the original CLI model choices and request controls.
    if method == 'flip_attack':
        if model.get('author_model') not in FLIP_MODELS:
            raise ValueError('FlipAttack author_model must be an original CLI model name.')

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

    elif method == 'equacode':
        if model.get('author_model') != 'gpt-3.5-turbo':
            raise ValueError('EquaCode author_model must retain gpt-3.5-turbo.')
        if config.get('judge_model') != 'GPT-4':
            raise ValueError('EquaCode judge_model must retain the source label GPT-4.')
        if config['variant'] not in {'equacode', 'equation', 'code', 'origin'}:
            raise ValueError('Unknown EquaCode variant.')

    elif method == 'gta':
        if model.get('author_model') != 'deepseek_r1':
            raise ValueError('GTA initially supports the published deepseek_r1 target profile only.')
        if isinstance(config['rounds'], bool) or not isinstance(config['rounds'], int):
            raise ValueError('GTA rounds must match the author CLI integer argument.')

    elif method == 'mousetrap':
        if model.get('author_model') == 'gemini-2.5-pro-exp-03-25':
            raise ValueError('Mousetrap Google execution is unavailable until OpenRouter safety settings are verified.')
        if model.get('author_model') not in {'o1-2024-12-17', 'claude-3-7-sonnet-20250219'}:
            raise ValueError('Mousetrap author_model must select a supported notebook target.')

    elif method == 'query_attack':
        # Keep the original translation and target CLI selections independent.
        languages = {'C++', 'C', 'C#', 'Python', 'Go', 'SQL', 'Java', 'JavaScript', 'URL'}
        if config['target_language'] not in languages:
            raise ValueError('QueryAttack target_language must select an original target language.')
        if config['translation_language'] not in languages | {'ALL'}:
            raise ValueError('Unknown QueryAttack translation_language.')
        if model.get('author_model') != 'gpt-4-1106-preview':
            raise ValueError('QueryAttack retains the original gpt-4-1106-preview source label.')
        if not isinstance(config['trans_verify'], bool):
            raise ValueError('QueryAttack trans_verify must be boolean.')

        # The source seed controls the API request only, without seeding local RNG.
        for option in ('max_new_tokens', 'request_seed'):
            if isinstance(config[option], bool) or not isinstance(config[option], int):
                raise ValueError(f'{option} must match the author CLI integer argument.')
        for option in ('temperate', 'top_p'):
            if isinstance(config[option], bool) or not isinstance(config[option], (int, float)):
                raise ValueError(f'{option} must match the author CLI float argument.')


AUTHOR_UTILS = Path(__file__).resolve().parent / 'utils'


def load_symbols(filename, names, bindings):
    # Execute only declared author definitions; bind imports and storage I/O explicitly.
    path = AUTHOR_UTILS / filename
    tree = ast.parse(path.read_bytes())
    nodes = [node for node in tree.body if (
        isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
        or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets))]

    # Execute the selected source definitions with explicitly supplied I/O globals.
    namespace = dict(bindings)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return {name: namespace[name] for name in names}


class OpenRouterAuthorClient:
    """Bind author client interfaces to one SDK client and retain per-example traces."""

    def __init__(self, client, model, judge):
        # Share the existing SDK client and keep separate routes and traces for each role.
        self.client = client
        self.routes = {'target': model, 'judge': judge}
        self.requests, self.durations = [], []
        self.index, self.phase = -1, 'target'

        # These are the only OpenAI/Anthropic client interfaces used by the author code.
        self.chat = SimpleNamespace(completions=self)
        self.messages = SimpleNamespace(create=lambda **kwargs: self.create_for(self.phase, _anthropic=True, **kwargs))

    def __call__(self, **_options):
        # Original constructor credentials/endpoints are replaced by the existing hosted client.
        return self

    def create(self, **kwargs):
        # Follow the target/judge phase expected by sequential author clients.
        return self.create_for(self.phase, **kwargs)

    def for_role(self, role):
        # Bind either author SDK interface to a fixed role across repeated attempts.
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: self.create_for(role, **kwargs))),
            messages=SimpleNamespace(create=lambda **kwargs: self.create_for(role, _anthropic=True, **kwargs)))

    def create_for(self, role, _anthropic=False, **kwargs):
        # Explicit roles allow multi-agent attacks to interleave target/attacker/judge calls.
        route = self.routes[role]
        record = {'example': self.index, 'role': role, 'endpoint': str(self.client.base_url),
                  'request': copy.deepcopy(kwargs)}
        self.requests[self.index].append(record)

        # Claude has a separate system field and expects a different response shape.
        routed = copy.deepcopy(kwargs)
        anthropic = _anthropic or 'system' in routed
        if 'system' in routed:
            routed['messages'] = [{'role': 'system', 'content': routed.pop('system')}, *routed['messages']]

        # Pin the configured OpenRouter model and provider while retaining source request controls.
        routed['model'] = route['id']
        routed['extra_body'] = {'provider': {'only': [route['provider']],
                                             'allow_fallbacks': False, 'require_parameters': True}}
        record['routed_request'] = copy.deepcopy(routed)
        try:
            # Keep SDK retries; all application retries remain in the original author methods.
            response = self.client.chat.completions.create(**routed)

            # Capture text and API metadata for the repository result adapter.
            usage = getattr(response, 'usage', None)
            record['metadata'] = {'id': response.id, 'model': response.model,
                                  'usage': usage.model_dump() if usage else None,
                                  'finish_reasons': [choice.finish_reason for choice in response.choices],
                                  'provider': getattr(response, 'provider', None)}
            record['raw_text'] = response.choices[0].message.content if response.choices else None
            self.phase = 'judge'

            # Return the response shape expected by the original author SDK wrapper.
            if anthropic:
                content = [SimpleNamespace(text=record['raw_text'])] if response.choices else []
                return SimpleNamespace(content=content)
            return response
        except Exception as error:
            # Retain the failed call in the trace and propagate the original SDK error.
            record['error_type'] = type(error).__name__
            raise

    def examples(self, values):
        # Group traces as requests happen, avoiding a later scan of the full request history.
        for self.index, value in enumerate(values):
            self.phase = 'target'
            self.requests.append([])
            started = time.perf_counter()
            yield value

            # Finish timing when the author driver completes this example.
            self.durations.append(time.perf_counter() - started)


def execute(method, model, config, dataset, *, client, sleep=None):
    # Validate before preparing a shared transport and dataset question sequence.
    validate_author_config(method, model, config)
    transport = OpenRouterAuthorClient(client, model, config.get('judge', {}))
    questions = [conv[0]['content'] for conv in dataset]
    try:
        # Dispatch to each attack's I/O binding through the same execution interface.
        if method == 'flip_attack':
            from .utils.flip_attack import binding as flip_attack

            records, summary = flip_attack.execute(transport, model, config, questions, sleep)

        elif method == 'equacode':
            from .utils.equacode import binding as equacode

            records, summary = equacode.execute(transport, model, config, questions, sleep)

        elif method == 'gta':
            from .utils.gta import binding as gta

            records, summary = gta.execute(transport, model, config, questions, sleep)

        elif method == 'mousetrap':
            from .utils.mousetrap import binding as mousetrap

            records, summary = mousetrap.execute(transport, model, config, questions, sleep)

        elif method == 'query_attack':
            from .utils.query_attack import binding as query_attack

            records, summary = query_attack.execute(transport, model, config, questions, sleep)
    except Exception as error:
        # Expose the attempted calls when the original driver aborts.
        error.author_requests = [request for trace in transport.requests for request in trace]
        raise

    # Return original records and counters with request groups and measured durations.
    return {'records': records, 'requests': transport.requests, 'durations': transport.durations,
            'author_summary': summary, 'environment': {'openai': version('openai')}}
