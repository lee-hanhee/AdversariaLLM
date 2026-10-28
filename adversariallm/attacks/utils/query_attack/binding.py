"""Execute the original QueryAttack file stages with temporary storage and hosted clients."""

import ast
from bisect import bisect_right
from contextlib import contextmanager
import csv
import io
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace

from ...author_execution import AUTHOR_UTILS, load_symbols


def execute(transport, model, config, questions, sleep):
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
        'your_model': model['author_model'],
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
    template_tree = ast.parse((AUTHOR_UTILS / 'query_attack/all_template.py').read_bytes())
    names = [target.id for node in template_tree.body if isinstance(node, ast.Assign)
             for target in node.targets if isinstance(target, ast.Name)]
    templates = load_symbols('query_attack/all_template.py', names, {})
    translator_tree = ast.parse((AUTHOR_UTILS / 'query_attack/translator.py').read_bytes())
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
            'your_model': model['author_model']})
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
        tree = ast.parse((AUTHOR_UTILS / 'query_attack/queryattack_driver.py').read_bytes())
        main = next(node for node in tree.body if isinstance(node, ast.If))
        start = next(index for index, node in enumerate(main.body) if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == 'start_str_map' for target in node.targets))
        target_args = SimpleNamespace(target_language=config['target_language'], temperate=config['temperate'],
                                      top_p=config['top_p'], max_new_tokens=config['max_new_tokens'],
                                      seed=config['request_seed'])
        namespace = {**templates, **driver, 'args': target_args, 'os': filesystem, 'open': open_file,
                     'your_model': model['author_model'], 'load_trans': load_trans,
                     'process_sentence': process_sentence}
        exec(compile(ast.Module(body=main.body[start:], type_ignores=[]),
                     'author-queryattack-target', 'exec'), namespace)

        # Capture author output files before temporary storage is removed.
        files = {str(path.relative_to(root)): path.read_text(encoding='utf-8')
                 for path in sorted(root.rglob('*')) if path.is_file() and path.name != 'input.csv'}

    return records, {'files': files}
