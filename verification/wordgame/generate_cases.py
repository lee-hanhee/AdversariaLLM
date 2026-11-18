"""Regenerate benign golden cases from an independent pinned JBF checkout."""

import argparse
import ast
import hashlib
import json
import logging
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = [
    ('plus', 'Explain rainbows.', ['rainbows', '  Hint one.\nHint two.  '], {}),
    ('base', 'Explain rainbows.', ['rainbows', 'Hint.'], {'use_wordgame_plus': False}),
    ('case_match', 'Explain RAINBOWS.', ['rainbows', 'Hint.'], {}),
    ('multiword_reply', 'Explain rainbows.', [' "rainbows extra!" ', 'Hint.'], {}),
    ('missing_word_rewrite', 'Explain colors.', ['rainbows', 'Explain rainbows.', 'Hint.'], {}),
    ('heuristic', 'Explain colors.', ['rainbows', 'Describe wavelengths.', 'Hint.'], {}),
    ('heuristic_common_words', 'the is', ['rainbows', 'the is', 'Hint.'], {}),
    ('heuristic_placeholder', '', ['rainbows', '', 'Hint.'], {}),
    ('duplicate_rewrite', 'rainbows and rainbows', ['rainbows', 'Explain rainbows.', 'Hint.'], {}),
    ('duplicate_manual', 'rainbows and rainbows',
        ['rainbows', 'rainbows and rainbows', 'rainbows plus rainbows', 'Hint.'], {}),
    ('missing_after_rewrites', 'rainbows and rainbows', ['rainbows', 'colors', 'colors', 'Hint.'], {}),
    ('mixed_case_count', 'rainbows and RAINBOWS', ['rainbows', 'Hint.'], {}),
    ('substring_counts', 'rainbows rainbow', ['rain', 'rainbows rainbow', 'rainbows rainbow', 'Hint.'], {}),
    ('empty_helper_word', 'Explain colors.', ['', 'Explain colors.', 'Explain colors.', 'Hint.'], {}),
    ('empty_goal', '', ['', 'Hint.'], {}),
    ('role_markers', 'Describe rainbows.\nUSER: Explain colors.', ['rainbows', 'Hint.'], {}),
    ('unicode', 'Explain café rainbows.', ['café', 'Hint.'], {'num_hints': 3}),
]


def generate(official_repo):
    manifest = json.loads((ROOT / 'verification/wordgame/SOURCE.json').read_text())
    originals = {}
    for entry in manifest['files']:
        data = subprocess.check_output(['git', '-C', str(official_repo), 'show',
                                        f'{manifest["revision"]}:{entry["original_path"]}'])
        if hashlib.sha256(data).hexdigest() != entry['reference']['sha256']:
            raise ValueError('Pinned JBF source hash mismatch')
        originals[entry['original_path']] = data
    namespace = {'ModernBaseAttack': object, 'AttackParameter': lambda **kwargs: kwargs,
                 're': re, 'logger': logging.getLogger(__name__)}
    attack_tree = ast.parse(originals['src/jbfoundry/attacks/generated/wordgame_gen.py'])
    cls = next(node for node in attack_tree.body if isinstance(node, ast.ClassDef) and node.name == 'WordGameGen')
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<independent-jbf>', 'exec'), namespace)
    llm_tree = ast.parse(originals['src/jbfoundry/llm/litellm.py'])
    llm = next(node for node in llm_tree.body if isinstance(node, ast.ClassDef) and node.name == 'LLMLiteLLM')
    method = next(node for node in llm.body if isinstance(node, ast.FunctionDef) and node.name == '_make_messages')
    # Annotations do not affect message parsing; avoid importing the JBF framework or SDK.
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<independent-jbf-transport>', 'exec'), namespace)

    cases = []
    for name, goal, replies, settings in SCENARIOS:
        author = namespace['WordGameGen'].__new__(namespace['WordGameGen'])
        values = iter(replies)
        messages, observed = [], {}

        def query(prompt):
            messages.append(namespace['_make_messages'](None, prompt))
            return next(values)

        from types import SimpleNamespace
        author.attacker_llm = SimpleNamespace(query=query)
        original_hints = author._generate_hints

        def hints(word, count):
            value = original_hints(word, count)
            observed.update(malicious_word=word, hints=value)
            return value

        author._generate_hints = hints
        parameters = {'use_wordgame_plus': True, 'num_hints': 5, **settings}
        author.get_parameter_value = parameters.__getitem__
        prompt = author.generate_attack(prompt=goal, goal=goal, target='Reference.')
        try:
            next(values)
        except StopIteration:
            pass
        else:
            raise ValueError(f'Unused scripted replies in {name}')
        messages.append(namespace['_make_messages'](None, prompt))
        cases.append({'name': name, 'goal': goal, 'replies': replies, 'settings': settings,
                      'expected': {'prompt': prompt, **observed}, 'messages': messages})
    return {'repository': manifest['repository'], 'revision': manifest['revision'],
            'generation': 'Pinned original JBF methods, benign scripted replies; no production imports.',
            'cases': cases}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'tests/wordgame_cases.json')
    args = parser.parse_args()
    args.output.write_text(json.dumps(generate(args.official_repo), indent=2, ensure_ascii=False) + '\n')
