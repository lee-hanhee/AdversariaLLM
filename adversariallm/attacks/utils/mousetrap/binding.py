"""Bind the preserved mousetrap driver to repository I/O."""

import ast
from contextlib import contextmanager
import io
import json
import random
from types import SimpleNamespace

from ...author_execution import AUTHOR_UTILS, load_symbols


def execute(transport, model, config, questions, sleep):
    import re

    # Keep original SDK wrappers, binding constructors to the existing role-specific client.
    def client_bindings(role):
        client = transport.for_role(role)
        factory = lambda **kwargs: client
        return {'openai': SimpleNamespace(OpenAI=factory), 'anthropic': SimpleNamespace(Anthropic=factory)}

    # Instantiate the original judge and the explicitly selected supported target wrapper.
    clients = 'mousetrap/mousetrap_clients.py'
    judger = load_symbols(clients, ['Judger'], client_bindings('judge'))['Judger']()
    if model['author_model'] == 'o1-2024-12-17':
        target_name = 'CheckerTarget_OpenaAI'
    elif model['author_model'] == 'claude-3-7-sonnet-20250219':
        target_name = 'CheckerTarget_Anthropic'
    target = load_symbols(clients, [target_name], client_bindings('target'))[target_name]()

    # Load the original transformation, attack, and judge functions into one namespace.
    pipeline = 'mousetrap/mousetrap_pipeline.py'
    names = [node.name for node in ast.parse((AUTHOR_UTILS / pipeline).read_bytes()).body
             if isinstance(node, ast.FunctionDef)]
    namespace = load_symbols(pipeline, names, {'random': random, 're': re, 'judger': judger, 'target': target})
    records = []

    @contextmanager
    def capture_output(path, mode, encoding):
        # Capture unchanged author JSONL writes; preserve every field, including unavailable dim.
        output = io.StringIO()
        yield output
        row = json.loads(output.getvalue())

        # Separate per-attempt records from the author's final minimum-success-length log.
        if path == namespace['output_file_path']:
            records[-1]['attempts'].append(row)
        elif path == namespace['output_log_path']:
            records[-1]['log'] = row
        else:
            raise ValueError(f'Unexpected author output path: {path}')

    # Execute original setup and per-example body, replacing only input/output file I/O.
    source = ast.parse((AUTHOR_UTILS / 'mousetrap/mousetrap_driver.py').read_bytes())
    setup = [node for node in source.body if isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in {'input_file_path', 'output_file_path', 'output_log_path', 'params'}
        for t in node.targets)]
    loop = next(node for node in source.body if isinstance(node, ast.For))

    # Supply repository questions and unavailable category labels through the original data shape.
    namespace.update(json=json, random=random, open=capture_output,
                     data=[{'ptq': question, 'dim': None} for question in questions])

    # Sample the original class permutation once before any examples run.
    exec(compile(ast.Module(body=setup, type_ignores=[]), 'author-mousetrap-setup', 'exec'), namespace)
    body = compile(ast.Module(body=loop.body, type_ignores=[]), 'author-mousetrap-example', 'exec')

    # Reuse that permutation and RNG stream while executing the original example body.
    for i in transport.examples(range(len(questions))):
        records.append({'params': list(namespace['params']), 'attempts': []})
        namespace['i'] = i
        exec(body, namespace)

    return records, {}
