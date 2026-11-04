"""Repository entry point for the preserved Mousetrap notebook protocol."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
import ast
import io
import json
import random

from .attack import Attack
from .utils.author_runtime import (
    AUTHOR_IMPLEMENTATIONS, load_symbols, normalize_config, run_attack, validate_routes,
)


@dataclass
class MousetrapConfig:
    # The notebook uses one unseeded random stream and a fixed three-by-three protocol.
    name: str = 'mousetrap'
    type: str = 'discrete'
    version: str = '0.1.0'
    seed: None = None

    # Select original model handling independently of the hosted model ID.
    source_profile: str | None = None

    # Require an explicit OpenRouter route for the original author judge.
    judge: dict = field(default_factory=lambda: {'id': None, 'provider': None})


class MousetrapAttack(Attack):
    """Run the preserved mousetrap algorithm through the hosted target."""

    REQUIRES_HOSTED_TARGET = True

    def __init__(self, config):
        # Preserve the source algorithm's unseeded random stream.
        self.config = normalize_config(config)

    def run(self, target, dataset):
        return run_attack(self, target, dataset)

    @staticmethod
    def validate_config(model, config):
        validate_routes(model, config, ('judge',))
        if config.get('source_profile') == 'gemini-2.5-pro-exp-03-25':
            raise ValueError('Mousetrap Google execution is unavailable until OpenRouter safety settings are verified.')
        if config.get('source_profile') not in {'o1-2024-12-17', 'claude-3-7-sonnet-20250219'}:
            raise ValueError('Mousetrap source_profile must select a supported notebook target.')

    @staticmethod
    def _add_steps(run, record, trace, duration):
        # Each original attempt makes one target call followed by one judge call.
        for index, attempt in enumerate(record['attempts']):
            turn = trace[2 * index:2 * index + 2]
            target_request = turn[0]
            response = attempt['response']
            raw = target_request.get('raw_text')
            run.add_step(target_request, messages=target_request['request']['messages'],
                         completions=[] if response is None else [response],
                         raw_completions=[] if raw is None else [raw],
                         scores={'mousetrap_author': {'ats': [float(attempt['ats'])], 'aru': [float(attempt['aru'])]}},
                         author_result=attempt, author_requests=turn)

    @staticmethod
    def _execute(transport, model, config, questions, sleep=None):
        questions = [question for question, _ in questions]
        import re

        # Keep original SDK wrappers, binding constructors to the existing role-specific client.
        def client_bindings(role):
            client = transport.for_role(role)
            factory = lambda **kwargs: client
            return {'openai': SimpleNamespace(OpenAI=factory), 'anthropic': SimpleNamespace(Anthropic=factory)}

        # Instantiate the original judge and the explicitly selected supported target wrapper.
        clients = 'mousetrap/mousetrap_clients.py'
        judger = load_symbols(clients, ['Judger'], client_bindings('judge'))['Judger']()
        if config['source_profile'] == 'o1-2024-12-17':
            target_name = 'CheckerTarget_OpenaAI'
        elif config['source_profile'] == 'claude-3-7-sonnet-20250219':
            target_name = 'CheckerTarget_Anthropic'
        target = load_symbols(clients, [target_name], client_bindings('target'))[target_name]()

        # Load the original transformation, attack, and judge functions into one namespace.
        pipeline = 'mousetrap/mousetrap_pipeline.py'
        names = [node.name for node in ast.parse((AUTHOR_IMPLEMENTATIONS / pipeline).read_bytes()).body
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
        source = ast.parse((AUTHOR_IMPLEMENTATIONS / 'mousetrap/mousetrap_driver.py').read_bytes())
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
