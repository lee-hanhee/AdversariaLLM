"""Execute unchanged author code using the shared runtime SDK."""

import ast
import copy
from importlib.metadata import version
from pathlib import Path
import time
from types import SimpleNamespace

AUTHOR_METHODS = frozenset({'flip_attack', 'equacode', 'gta'})
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

    else:
        raise ValueError(f'Unknown author method: {method}')

    if set(config) - common - options:
        raise ValueError(f'Unsupported author settings: {set(config) - common - options}')

    # Routing IDs are explicit and independent of names used by the author logic.
    if model.get('backend') != 'openrouter':
        raise ValueError(f'{method} requires backend=openrouter.')
    routes = [('target', model), ('judge', config.get('judge') or {})]
    if method == 'gta':
        routes.append(('attacker', config.get('attacker') or {}))
    for role, route in routes:
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
        self.client = client
        self.routes = {'target': model, 'judge': judge}
        self.requests, self.durations = [], []
        self.index, self.phase = -1, 'target'

        # These are the only OpenAI/Anthropic client interfaces used by the author code.
        self.chat = SimpleNamespace(completions=self)
        self.messages = self

    def __call__(self, **_options):
        # Original constructor credentials/endpoints are replaced by the existing hosted client.
        return self

    def create(self, **kwargs):
        return self.create_for(self.phase, **kwargs)

    def create_for(self, role, **kwargs):
        # Explicit roles allow multi-agent attacks to interleave target/attacker/judge calls.
        route = self.routes[role]
        record = {'example': self.index, 'role': role, 'endpoint': str(self.client.base_url),
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



def _execute_gta(transport, config, questions, sleep):
    import re

    prompts = load_symbols('gta/gta_prompts.py', ['model_a_background_base', 'model_b_background_base',
                                               'model_c_background_base', 'judge_system_prompt'], {})
    transport.routes.update(Police=config['attacker'], Thomas=transport.routes['target'],
                            Marcos=transport.routes['target'], Judge=config['judge'])
    clock = SimpleNamespace(sleep=sleep or time.sleep)
    records = []
    for goal in transport.examples(questions):
        agents, calls, judgements = {}, [], []

        def agent_class(role, filename, name):
            # Bind the unchanged models.chat per role, including DeepSeek processing and invalid fallback.
            sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
                create=lambda **kwargs: transport.create_for(role, **kwargs))))
            original_chat = load_symbols('gta/gta_models.py', ['chat'], {
                'client_gpt': sdk, 'client_azure': sdk,
            })['chat']

            def chat(model, messages, temperature=0.1):
                call = {'role': role, 'model': model, 'messages': copy.deepcopy(messages), 'temperature': temperature}
                calls.append(call)
                try:
                    call['response'] = original_chat(model, messages, temperature)
                    return call['response']
                except Exception as error:
                    call['error_type'] = type(error).__name__
                    raise

            base = load_symbols('gta/base_agent.py', ['BaseAgent'], {'chat': chat})['BaseAgent']
            return load_symbols('gta/' + filename, [name], {
                'BaseAgent': base, 're': re, 'time': clock, 'chat': chat,
            })[name]

        def capture_agent(cls, **kwargs):
            instance = cls(**kwargs)
            agents[kwargs['role_name']] = instance
            return instance

        police = agent_class('Police', 'attack_agent.py', 'AttackAgent')
        thomas = agent_class('Thomas', 'target_agent.py', 'TargetAgent')
        marcos = agent_class('Marcos', 'target_agent.py', 'TargetAgent')
        judge = agent_class('Judge', 'judge_agent.py', 'JudgeAgent')

        class RecordingJudge(judge):
            def judge(self, response):
                # Capture the author's returned rating; never parse or recompute it here.
                rating, analysis = super().judge(response)
                judgements.append({'role': 'Thomas' if len(judgements) % 2 == 0 else 'Marcos',
                                   'round': len(judgements) // 2 + 1, 'response': response,
                                   'rating': rating, 'analysis': analysis,
                                   'request_end': len(transport.requests[transport.index])})
                return rating, analysis

        def save(conversation, num, output_dir):
            records.append({'id': num, 'conversation': copy.deepcopy(conversation),
                            'judgements': judgements, 'calls': calls,
                            'histories': {role: copy.deepcopy(agent.message_memory) for role, agent in agents.items()}})

        driver = load_symbols('gta/gta_driver.py', ['ours_attack'], {
            **prompts, 'os': SimpleNamespace(makedirs=lambda *args, **kwargs: None), 'save_conversation': save,
            'AttackAgent': lambda **kwargs: capture_agent(police, **kwargs),
            'TargetAgent': lambda **kwargs: capture_agent(thomas if kwargs['role_name'] == 'Thomas' else marcos, **kwargs),
            'JudgeAgent': lambda **kwargs: capture_agent(RecordingJudge, **kwargs),
        })['ours_attack']
        driver(rounds=config['rounds'], prompt=goal, num=transport.index,
               target_model_1='deepseek_r1', target_model_2='deepseek_r1', attack_model='gpt-4o-mini', output_dir='')
    return records


def execute(method, model, config, dataset, *, client, sleep=None):
    validate_author_config(method, model, config)
    transport = OpenRouterAuthorClient(client, model, config['judge'])
    questions = [conv[0]['content'] for conv in dataset]
    records = []

    summary = {}
    try:
        if method == 'flip_attack':
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
            exec(compile(ast.Module(body=[loop], type_ignores=[]), 'author-flip-loop', 'exec'), namespace)
            if config['evaluate']:
                summary = {key: namespace[key] for key in ('all_count', 'dict_success_count', 'gpt_success_count')}

        elif method == 'equacode':
            # Reuse the original EquaCode class; replace spreadsheet I/O with record capture.
            templates = load_symbols('equacode/equacode_prompts.py', ['equacoder', 'equa', 'coder', 'origin'], {})
            helper = load_symbols('equacode/equacode_utils.py', ['synthesis_message'], {})

            def save(row, fields, filename):
                records.append({'question': fields[0], 'prompt': fields[1], 'response': fields[2],
                                'evaluation': fields[3], 'row': row})

            driver = load_symbols('equacode/equacode_driver.py', ['targetLLM'], {
                **templates, **helper, 'os': __import__('os'), 'OpenAI': transport,
                'openai_key': client.api_key, 'base_url': str(client.base_url),
                'assessmodel': config['judge_model'], 'assess_base_url': str(client.base_url), 'save_data': save,
            })['targetLLM']
            author = driver(questions=transport.examples(questions), target_model=model['author_model'], text=config['variant'])
            author.filecheck = lambda: None
            author.gernerate()

        elif method == 'gta':
            records = _execute_gta(transport, config, questions, sleep)
    except Exception as error:
        error.author_requests = [request for trace in transport.requests for request in trace]
        raise
    return {'records': records, 'requests': transport.requests, 'durations': transport.durations,
            'author_summary': summary, 'environment': {'openai': version('openai')}}
