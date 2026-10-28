"""Run complete official QueryAttack scripts independently of the production binding."""

import builtins
import csv
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace


def run_reference_query_attack(payload, factory, base):
    import openai

    # Import complete original modules, replacing only SDK construction and placeholder endpoints.
    original_import = builtins.__import__
    sdk = ModuleType('openai')
    sdk.__dict__.update(vars(openai))
    sdk.OpenAI = lambda **kwargs: factory('openai', {
        **kwargs, 'api_key': 'test-key', 'base_url': 'https://original-queryattack.example/v1'})

    def import_source(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'openai':
            return sdk
        return original_import(name, globals, locals, fromlist, level)

    source_builtins = {**vars(builtins), '__import__': import_source}
    modules = ['all_template', 'openai_log', 'gpt_judge', 'split_sentence_llms', 'trans_llms']
    for name in modules:
        module = ModuleType(name)
        module.__dict__['__builtins__'] = source_builtins
        sys.modules[name] = module
        exec(compile((base / f'{name}.py').read_bytes(), str(base / f'{name}.py'), 'exec'), vars(module))

    # Serialize repository messages into the same physical-line file consumed by the author CLI.
    config = payload['config']
    questions = [conversation[0]['content'] for conversation in payload['dataset']]
    owners = []
    for index, question in enumerate(questions):
        normalized = (question + '\n').replace('\r\n', '\n').replace('\r', '\n')
        owners.extend([index] * (len(normalized.split('\n')) - 1))
    records = [{'lines': []} for _ in questions]
    previous = Path.cwd()

    with TemporaryDirectory(prefix='reference-queryattack-') as directory:
        try:
            # The oracle runs in its own subprocess, so the original relative paths can be used directly.
            os.chdir(directory)
            Path('input.csv').write_text(''.join(question + '\n' for question in questions), encoding='utf-8')
            with open('input.csv', encoding='utf-8') as stream:
                source_lines = [line.strip() for line in stream]
            sys.argv = ['translator.py', '--path', 'input.csv', '--store_path', 'data',
                        '--target_language', config['translation_language']]
            if config['trans_verify']:
                sys.argv.extend(['--trans_verify', 'True'])
            scope = {'__name__': '__main__', '__builtins__': source_builtins}
            exec(compile((base / 'translator.py').read_bytes(), str(base / 'translator.py'), 'exec'), scope)

            # Obtain per-input assembled strings from the original pure assembly function for attribution.
            with open('data/llms/phrases.csv', encoding='utf-8') as stream:
                phrases = list(csv.DictReader(stream))
            generated = {}
            scope['store_data'] = lambda rows, options: generated.update({options.target_language: list(rows)})
            scope['assemble_code'](phrases, SimpleNamespace(target_language=config['translation_language']))

            # Run the original target CLI, including its file parser and output serialization.
            sys.argv = ['test.py', '--target_language', config['target_language'],
                        '--temperate', str(config['temperate']), '--top_p', str(config['top_p']),
                        '--max_new_tokens', str(config['max_new_tokens']), '--seed', str(config['request_seed'])]
            responses = []

            def target_print(text):
                # Capture the exact printed response, including carriage returns that text-file reads normalize.
                responses.append(text.partition(':  ')[2])
                print(text)

            scope = {'__name__': '__main__', '__builtins__': source_builtins, 'print': target_print}
            exec(compile((base / 'test.py').read_bytes(), str(base / 'test.py'), 'exec'), scope)

            # Locate the original assembled inputs in the actual file, independently of production write tracking.
            translated_file = Path(f'data/llms/{config["target_language"]}_translated.csv')
            serialized = translated_file.read_text(encoding='utf-8')
            starts = []
            cursor = 0
            for sentence in generated[config['target_language']]:
                sentence = sentence.replace('\r\n', '\n').replace('\r', '\n')
                start = serialized.index(sentence, cursor)
                starts.append(start)
                cursor = start + len(sentence)

            # Map every original target row, including delimiter-created fragments, to its physical input.
            cursor, source_index = 0, 0
            for index, (sentence, response) in enumerate(zip(scope['data'], responses)):
                start = serialized.index(sentence, cursor)
                while source_index + 1 < len(starts) and starts[source_index + 1] <= start:
                    source_index += 1
                cursor = start + len(sentence)
                records[owners[source_index]]['lines'].append({
                    'sentence_index': source_index, 'target_index': index, 'question': source_lines[source_index],
                    'phrases': [row for row in phrases if int(row['sentence_index']) == source_index],
                    'translation': sentence, 'response': response,
                })
            files = {str(path): path.read_text(encoding='utf-8') for path in sorted(Path('.').rglob('*'))
                     if path.is_file() and path.name != 'input.csv'}
        finally:
            os.chdir(previous)

    return {'records': records, 'author_summary': {'files': files}}
