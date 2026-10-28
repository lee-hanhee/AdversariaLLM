"""Offline subprocess harness: original reference versus integrated author execution."""

import ast
from contextlib import redirect_stdout
from importlib.metadata import version
import json
import os
import random
import time
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import httpx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# SDK baselines omit the ML stack imported by attacks/__init__.py. Expose only
# its package path here; public entry-point tests still use normal package imports.
package = ModuleType('adversariallm.attacks')
package.__path__ = [str(ROOT / 'adversariallm/attacks')]
sys.modules[package.__name__] = package
from adversariallm.attacks import author_execution


def reference_symbols(path, names, bindings):
    # The expected implementation comes from independent official reference files.
    tree = ast.parse(path.read_bytes())
    chosen = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            chosen.append(node)
        elif isinstance(node, ast.Assign) and any(getattr(t, 'id', None) in names for t in node.targets):
            chosen.append(node)

    # Execute the independently selected source with controlled API and storage globals.
    namespace = dict(bindings)
    exec(compile(ast.Module(body=chosen, type_ignores=[]), str(path), 'exec'), namespace)
    return {name: namespace[name] for name in names}


def run_reference(payload, factory, sleep):
    # Resolve source-level model labels, questions, and the pinned reference package.
    method, config, model = payload['method'], payload['config'], payload['model']
    model = {**model, 'id': model['author_model']}
    questions = [conv[0]['content'] for conv in payload['dataset']]
    base = ROOT / 'verification' / method / 'reference'

    # Delegate the larger original-module and notebook oracles to their reference runners.
    if method == 'gta':
        from gta_reference import run_reference_gta
        return run_reference_gta(payload, factory, sleep, base)

    if method == 'mousetrap':
        from mousetrap_reference import run_reference_mousetrap
        return run_reference_mousetrap(payload, factory, base)

    if method == 'query_attack':
        from query_attack_reference import run_reference_query_attack
        return run_reference_query_attack(payload, factory, base)

    # Construct native SDK clients for the independent original-source path.
    openai = lambda **kw: factory('openai', kw)
    anthropic = lambda **kw: factory('anthropic', kw)
    records = []

    if method == 'flip_attack':
        llm = reference_symbols(base / 'src/llm.py', ['LLM'], {
            'os': os, 'time': SimpleNamespace(sleep=sleep), 'OpenAI': openai,
            'Anthropic': anthropic, 'anthropic': SimpleNamespace(Anthropic=anthropic),
        })['LLM']
        evaluator = reference_symbols(base / 'src/eval_util.py', ['Evaluator'], {
            're': __import__('re'), 'LLM': llm,
        })['Evaluator']
        builder = reference_symbols(base / 'src/flip_attack.py', ['FlipAttack'], {
            'textwrap': __import__('textwrap'),
        })['FlipAttack']

        # Extract the author's original per-example loop rather than recreating its control flow.
        original = ast.parse((base / 'src/main.py').read_bytes())
        main = next(n for n in original.body if isinstance(n, ast.If))
        loop = next(n for n in main.body if isinstance(n, ast.For))

        # Supply the original CLI argument names and the scripted dataset to that loop.
        args = SimpleNamespace(**config, victim_llm=model['id'], judge_llm=config['judge_model'],
                               eval=config['evaluate'], begin=0, end=len(questions))
        victim = llm(model_id=model['id'], **{k: config[k] for k in (
            'temperature', 'max_token', 'retry_time', 'failed_sleep_time', 'round_sleep_time')})
        scope = {'args': args, 'adv_bench': {'goal': questions}, 'FlipAttack': builder,
                 'Evaluator': evaluator, 'victim_llm': victim, 'tqdm': lambda values: values,
                 'result_dict_list': records, 'all_count': 0, 'dict_success_count': 0, 'gpt_success_count': 0}
        exec(compile(ast.Module(body=[loop], type_ignores=[]), 'reference-loop', 'exec'), scope)
        return records

    # Load EquaCode templates and capture the rows its original spreadsheet callback emits.
    templates = reference_symbols(base / 'scripts/attack.py', ['equacoder', 'equa', 'coder', 'origin'], {})
    helper = reference_symbols(base / 'utils.py', ['synthesis_message'], {})
    driver = reference_symbols(base / 'main.py', ['targetLLM'], {
        **templates, **helper, 'os': os, 'OpenAI': openai, 'openai_key': 'test-key',
        'base_url': 'https://original-target.example/v1', 'assessmodel': config['judge_model'],
        'assess_base_url': 'https://original-judge.example/v1',
        'save_data': lambda row, fields, filename: records.append({
            'question': fields[0], 'prompt': fields[1], 'response': fields[2],
            'evaluation': fields[3], 'row': row}),
    })['targetLLM']

    # Run the original mandatory target/judge flow with filesystem setup disabled.
    instance = driver(questions=questions, target_model=model['id'], text=config['variant'])
    instance.filecheck = lambda: None
    instance.gernerate()
    return records


def main():
    # Read one benign test scenario and collect requests plus application/SDK sleeps.
    payload = json.load(sys.stdin)
    wire, sleeps, sdk_sleeps = [], [], []
    # Fix only the offline SDK retry jitter; production author execution is unseeded.
    random.seed(payload.get('rng_seed', 2026))
    time.sleep = sdk_sleeps.append
    scripted = iter(payload['replies'])

    # Use fake native credentials and remove external endpoint overrides from the test process.
    for key in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'DEEPINFRA_API_KEY'):
        os.environ[key] = 'test-key'
    os.environ['OPENROUTER_API_KEY'] = 'test-openrouter-key'
    os.environ.pop('OPENAI_BASE_URL', None)
    os.environ.pop('ANTHROPIC_BASE_URL', None)

    # Real pinned SDKs serialize requests and parse replies, with HTTP kept entirely in memory.
    def http(request):
        # Capture the real SDK request body and check hosted authentication.
        body = json.loads(request.content)
        wire.append({'url': str(request.url), 'body': body})
        if payload['method'] == 'gta':
            from gta_reference import gta_role
            wire[-1]['role'] = gta_role(body['messages'])
        if request.url.host == 'openrouter.ai':
            assert request.headers['authorization'] == 'Bearer test-openrouter-key'

        # Return the next scripted failure or model reply entirely in memory.
        item = next(scripted)
        if item.get('status', 200) != 200:
            return httpx.Response(item['status'], json={'error': {
                'message': 'Unavailable original model', 'type': 'invalid_request_error', 'code': 'model_not_found',
            }}, request=request)

        # Build the native provider response shape consumed by each SDK parser.
        text = item['text']
        if request.url.host == 'api.anthropic.com':
            data = {'id': 'msg-test', 'type': 'message', 'role': 'assistant', 'model': body['model'],
                    'content': [{'type': 'text', 'text': text}], 'stop_reason': 'end_turn',
                    'usage': {'input_tokens': 2, 'output_tokens': 3}}
        else:
            data = {'id': 'chatcmpl-test', 'object': 'chat.completion', 'created': 0, 'model': body['model'],
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5}}

        # Let empty-choice cases reach the author's original response-processing code.
        if item.get('empty_choices'):
            data['content' if request.url.host == 'api.anthropic.com' else 'choices'] = []
        return httpx.Response(200, json=data, request=request)

    def factory(provider, options):
        # Attach the scripted HTTP transport to each native SDK constructor.
        transport = httpx.Client(transport=httpx.MockTransport(http))
        if provider == 'anthropic':
            from anthropic import Anthropic
            return Anthropic(**options, http_client=transport)
        from openai import OpenAI, AzureOpenAI
        client = AzureOpenAI if provider == 'azure' else OpenAI
        return client(**options, http_client=transport)

    try:
        # Keep author printouts separate from the JSON comparison result.
        with redirect_stdout(sys.stderr):
            if payload.pop('reference', False):
                base = ROOT / 'verification' / payload['method'] / 'reference'
                # Mousetrap publishes no dependency pins; use the declared shared test baseline.
                pins = {}
                if payload['method'] != 'mousetrap':
                    requirements = base / ('requirement.txt' if payload['method'] == 'equacode' else 'requirements.txt')
                    pins = dict(line.replace(' ', '').split('==') for line in requirements.read_text().splitlines() if '==' in line)

                if payload['method'] == 'query_attack':
                    assert version('openai') == '1.54.3'
                    assert version('nltk') == '3.9.1'

                # Enforce source SDK pins when the original release declares them.
                for name in ('openai', 'anthropic'):
                    if name in pins:
                        assert version(name) == pins[name]
                reference_result = run_reference(payload, factory, sleeps.append)
                if payload['method'] == 'query_attack':
                    result = reference_result
                else:
                    result = {'records': reference_result}
            else:
                for key in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'DEEPINFRA_API_KEY'):
                    os.environ.pop(key, None)

                # Run the integrated path using only the shared OpenRouter client.
                client = factory('openai', {'api_key': 'test-openrouter-key', 'base_url': 'https://openrouter.ai/api/v1'})
                result = author_execution.execute(payload['method'], payload['model'], payload['config'],
                                                  payload['dataset'], client=client, sleep=sleeps.append)

        # Expose wire requests and both retry layers for comparison.
        result.update(wire=wire, sleeps=sleeps, sdk_sleeps=sdk_sleeps)
    except Exception as error:
        result = {'error': {'type': type(error).__name__, 'message': str(error)}, 'wire': wire, 'sleeps': sleeps, 'sdk_sleeps': sdk_sleeps}

    # Include the final random state when checking the notebook's shared RNG stream.
    if payload['method'] == 'mousetrap':
        result['rng_state'] = random.getstate()

    # Emit a machine-readable result for the parent test process.
    json.dump(result, sys.stdout, ensure_ascii=False)


if __name__ == '__main__':
    main()
