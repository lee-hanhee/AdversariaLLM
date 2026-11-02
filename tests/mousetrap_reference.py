"""Independent execution of the original notebook with native SDKs and in-memory files."""

import ast
from contextlib import contextmanager
import io
import json
import random
import re
from types import SimpleNamespace


def run_reference_mousetrap(payload, factory, base):
    # Read original notebook cells and prepare a fresh execution namespace.
    notebook = json.loads((base / 'Evaluation.ipynb').read_bytes())
    scope = {'random': random, 're': re, 'json': json}
    source = lambda index: ''.join(notebook['cells'][index]['source'])

    # Compile original classes directly, independently of the production source loader.
    for index in (2, 3):
        tree = ast.parse(source(index))
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        exec(compile(ast.Module(body=classes, type_ignores=[]), 'reference-clients', 'exec'), scope)

    # Instantiate the original judge and target classes with native test SDK clients.
    scope['openai'] = SimpleNamespace(OpenAI=lambda **kwargs: factory('openai', kwargs))
    scope['anthropic'] = SimpleNamespace(Anthropic=lambda **kwargs: factory('anthropic', kwargs))
    scope['judger'] = scope['Judger']()
    classes = {'o1-2024-12-17': 'CheckerTarget_OpenaAI',
               'claude-3-7-sonnet-20250219': 'CheckerTarget_Anthropic'}
    scope['target'] = scope[classes[payload['model']['author_model']]]()

    # Execute complete original function cells without using the production extraction loader.
    for index in (5, 7, 9, 10):
        exec(source(index), scope)

    # Prepare the notebook input shape and separate capture buffers for each example.
    inputs = [{'ptq': conv[0]['content'], 'dim': None} for conv in payload['dataset']]
    attempts = [[] for _ in inputs]
    logs = [{} for _ in inputs]

    @contextmanager
    def memory_file(path, mode, encoding):
        # Run the whole original driver, replacing only JSONL file access.
        if mode == 'r':
            with io.StringIO(''.join(json.dumps(row) + '\n' for row in inputs)) as stream:
                yield stream
        else:
            with io.StringIO() as stream:
                yield stream
                row = json.loads(stream.getvalue())

            # Keep the original attempt rows and final iteration logs in their source order.
            if path == scope['output_file_path']:
                attempts[scope['i']].append(row)
            elif path == scope['output_log_path']:
                logs[scope['i']] = row
            else:
                raise ValueError(path)

    # Run the entire original driver cell with only file access replaced.
    scope['open'] = memory_file
    exec(source(11), scope)

    # Expose the original shared permutation alongside captured rows and stop logs.
    return [{'params': scope['params'], 'attempts': rows, 'log': log} for rows, log in zip(attempts, logs)]
