"""Repository entry point for the original QueryAttack translation and target stages."""

from bisect import bisect_right
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import ast
import csv
import io
import json
import os
import re
import time

from .attack import Attack
from .utils.author_runtime import (
    AUTHOR_IMPLEMENTATIONS, load_symbols, normalize_config, run_attack, validate_routes,
)


@dataclass
class QueryAttackConfig:
    # Keep repository identification separate from the author's API request seed.
    name: str = 'query_attack'
    type: str = 'discrete'
    version: str = '0.1.0'
    seed: None = None

    # Select original model handling independently of the hosted model ID.
    source_profile: str = 'gpt-4-1106-preview'

    # Preserve the independent defaults of translator.py and test.py.
    translation_language: str = 'ALL'
    target_language: str = 'SQL'
    trans_verify: bool = False

    # Retain the original target CLI controls, including its temperature spelling.
    temperate: float = 0
    top_p: float = 1
    max_new_tokens: int = 512
    request_seed: int = 0

    # Require one explicit route for splitting, extraction, and translation verification.
    translator: dict = field(default_factory=lambda: {'id': None, 'provider': None})


class QueryAttack(Attack):
    """Run the preserved query_attack algorithm through the hosted target."""

    REQUIRES_HOSTED_TARGET = True

    def __init__(self, config):
        # Preserve the source algorithm's unseeded random stream.
        self.config = normalize_config(config)

    def run(self, target, dataset):
        return run_attack(self, target, dataset)

    @staticmethod
    def validate_config(model, config):
        validate_routes(model, config, ('translator',))
        # Keep the original translation and target CLI selections independent.
        languages = {'C++', 'C', 'C#', 'Python', 'Go', 'SQL', 'Java', 'JavaScript', 'URL'}
        if config['target_language'] not in languages:
            raise ValueError('QueryAttack target_language must select an original target language.')
        if config['translation_language'] not in languages | {'ALL'}:
            raise ValueError('Unknown QueryAttack translation_language.')
        if config.get('source_profile') != 'gpt-4-1106-preview':
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

    @staticmethod
    def _add_steps(run, record, trace, duration):
        # Physical input lines remain separate target calls within their originating example.
        targets = [request for request in trace if request['role'] == 'target']
        if len(targets) != len(record['lines']):
            raise ValueError('QueryAttack target trace does not match its source-line records.')
        for line, request in zip(record['lines'], targets):
            run.add_step(request, messages=request['request']['messages'],
                         completions=[line['response']], raw_completions=[request['raw_text']],
                         author_result=line, author_requests=[request])

    @staticmethod
    def _execute(transport, model, config, questions, sleep=None):
        questions = [question for question, _ in questions]
        import nltk

        # Resolve NLTK resources before any hosted request; never download during an attack.
        nltk.word_tokenize('Resource check.')
        nltk.pos_tag(['Resource', 'check'])

        # Keep helper calls distinct from target calls while reusing the same SDK client.
        transport.routes['translator'] = config['translator']
        helper_client = transport.for_role('translator')
        target_client = transport.for_role('target')
        helper = load_symbols('query_attack/openai_log.py', ['ask_response'], {
            'OpenAI': lambda **kwargs: helper_client,
            'your_api_key': 'bound-by-repository', 'your_base_url': None,
            'your_model': config['source_profile'],
        })['ask_response']

        # Execute the original parser and verification definitions without importing standalone modules.
        split = load_symbols('query_attack/split_sentence_llms.py', [
            'split_system_template', 'split_user_template', 'split_sentence_llms', 'validate_response',
        ], {'ask_response': helper, 'json': json, 're': re})
        translation = load_symbols('query_attack/trans_llms.py', [
            'trans_system_template', 'trans_user_template', 'test_safety_system_content',
            'test_safety_user_content', 'trans_sentence_llms', 'validate_response', 'test_safety',
        ], {'ask_response': helper, 're': re, 'word_tokenize': nltk.word_tokenize, 'pos_tag': nltk.pos_tag})

        # Load unchanged template constants and the complete original translation stage.
        template_tree = ast.parse((AUTHOR_IMPLEMENTATIONS / 'query_attack/all_template.py').read_bytes())
        names = [target.id for node in template_tree.body if isinstance(node, ast.Assign)
                 for target in node.targets if isinstance(target, ast.Name)]
        templates = load_symbols('query_attack/all_template.py', names, {})
        translator_tree = ast.parse((AUTHOR_IMPLEMENTATIONS / 'query_attack/translator.py').read_bytes())
        names = [node.name for node in translator_tree.body if isinstance(node, ast.FunctionDef)]

        # Mirror the original physical-line input file and retain each line's repository owner.
        line_owners = [index for index, question in enumerate(questions)
                       for _ in io.StringIO(question + '\n', newline=None)]
        records = [{'lines': []} for _ in questions]
        transport.requests = [[] for _ in questions]
        transport.durations = [0.0 for _ in questions]

        with TemporaryDirectory(prefix='queryattack-') as directory:
            root = Path(directory)
            (root / 'input.csv').write_text(''.join(question + '\n' for question in questions), encoding='utf-8')

            # Redirect file locations and observe the original per-input language-file writes.
            written_chunks = []

            @contextmanager
            def open_file(path, *args, **kwargs):
                mode = args[0] if args else kwargs.get('mode', 'r')
                with open(root / path, *args, **kwargs) as stream:
                    if str(path) == f'data/llms/{config["target_language"]}_translated.csv' and mode == 'w':
                        def write(text):
                            # Match the universal-newline view used by the original reader.
                            written_chunks.append(text.replace('\r\n', '\n').replace('\r', '\n'))
                            return stream.write(text)

                        yield SimpleNamespace(write=write)
                    else:
                        yield stream

            filesystem = SimpleNamespace(
                path=SimpleNamespace(exists=lambda path: (root / path).exists(), dirname=os.path.dirname),
                makedirs=lambda path: (root / path).mkdir(parents=True))
            args = SimpleNamespace(path='input.csv', store_path='data',
                                   target_language=config['translation_language'], trans_verify=config['trans_verify'])
            definitions = load_symbols('query_attack/translator.py', names, {
                **templates, 'csv': csv, 'os': filesystem, 'open': open_file,
                'word_tokenize': nltk.word_tokenize, 'trans_sentence_llms': translation['trans_sentence_llms'],
                'split_sentence_using_llms': split['split_sentence_llms'],
            })
            scope = definitions['main'].__globals__
            scope['parse_args'] = lambda: args
            original_translate = scope['translate']
            source_lines = []

            def timed_lines(data):
                # Attribute helper calls and elapsed time without changing original batch iteration.
                source_lines.extend(data)
                if len(source_lines) != len(line_owners):
                    raise ValueError('QueryAttack input lines do not match repository examples.')
                for index, line in enumerate(data):
                    transport.index = line_owners[index]
                    started = time.perf_counter()
                    yield line
                    transport.durations[transport.index] += time.perf_counter() - started

            # Keep translation before target generation and retain the author's phrase CSV round trip.
            scope['translate'] = lambda data, options: original_translate(timed_lines(data), options)
            scope['main']()
            phrases = scope['load_phrases'](args)

            # Bind original target definitions; the separate post-run evaluator is never invoked.
            driver = load_symbols('query_attack/queryattack_driver.py', [
                'ask_response_context', 'load_trans', 'process_sentence',
            ], {'OpenAI': lambda **kwargs: target_client, 'open': open_file,
                'your_api_key': 'bound-by-repository', 'your_base_url': None,
                'your_model': config['source_profile']})
            original_process = driver['process_sentence']
            target_sources = []

            def load_trans(path):
                # Preserve every original parsed fragment, including extra rows created by field delimiters.
                translated = driver['load_trans'](path)
                with open_file(path, 'r', encoding='utf-8') as stream:
                    serialized = stream.read()

                # Locate fragments in actual file writes, so extra target calls retain their input owner.
                boundaries = []
                position = 0
                for chunk in written_chunks:
                    position += len(chunk)
                    boundaries.append(position)
                cursor = 0
                for sentence in translated:
                    start = serialized.index(sentence, cursor)
                    target_sources.append(bisect_right(boundaries, start))
                    cursor = start + len(sentence)
                return translated

            def process_sentence(index, sentence, options, first_user, assistant):
                # Run the unchanged target call and newline escaping under the source line's owner.
                source_index = target_sources[index]
                transport.index = line_owners[source_index]
                started = time.perf_counter()
                response = original_process(index, sentence, options, first_user, assistant)
                transport.durations[transport.index] += time.perf_counter() - started

                # Retain original phrase rows, loaded prompt text, and file-formatted response.
                records[transport.index]['lines'].append({
                    'sentence_index': source_index, 'target_index': index, 'question': source_lines[source_index],
                    'phrases': [row for row in phrases if int(row['sentence_index']) == source_index],
                    'translation': sentence, 'response': response,
                })
                return response

            # Execute the original target main body after its argparse block, with explicit CLI values.
            tree = ast.parse((AUTHOR_IMPLEMENTATIONS / 'query_attack/queryattack_driver.py').read_bytes())
            main = next(node for node in tree.body if isinstance(node, ast.If))
            start = next(index for index, node in enumerate(main.body) if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Name) and target.id == 'start_str_map' for target in node.targets))
            target_args = SimpleNamespace(target_language=config['target_language'], temperate=config['temperate'],
                                          top_p=config['top_p'], max_new_tokens=config['max_new_tokens'],
                                          seed=config['request_seed'])
            namespace = {**templates, **driver, 'args': target_args, 'os': filesystem, 'open': open_file,
                         'your_model': config['source_profile'], 'load_trans': load_trans,
                         'process_sentence': process_sentence}
            exec(compile(ast.Module(body=main.body[start:], type_ignores=[]),
                         'author-queryattack-target', 'exec'), namespace)

            # Capture author output files before temporary storage is removed.
            files = {str(path.relative_to(root)): path.read_text(encoding='utf-8')
                     for path in sorted(root.rglob('*')) if path.is_file() and path.name != 'input.csv'}

        return records, {'files': files}
