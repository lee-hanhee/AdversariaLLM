"""Fixtures for original-versus-integrated author behavior comparisons."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

from adversariallm.attacks.flip_attack import FlipAttackConfig


ROOT = Path(__file__).resolve().parents[1]


def author_payload(method, **overrides):
    # Environment paths are explicit so missing author environments fail instead of skipping evidence.
    env_name = f'{method.upper()}_AUTHOR_PYTHON'
    python = os.environ.get(env_name, str(ROOT / '.author_envs' / method / 'bin/python'))
    config = asdict(FlipAttackConfig())
    config.update(overrides)
    model = {'backend': 'openrouter', 'id': 'test/target', 'provider': 'target-provider',
             'author_model': 'gpt-4-0613'}
    config['judge'] = {'id': 'test/judge', 'provider': 'judge-provider'}
    return {'method': method, 'model': model, 'config': config,
            'reference_python': python, 'shared_python': os.environ.get('AUTHOR_SHARED_PYTHON', sys.executable),
            'dataset': [[{'role': 'user', 'content': 'Explain rainbows through café windows.\nKeep {braces}.'},
                         {'role': 'assistant', 'content': 'Reference answer.'}]],
            'replies': [{'text': '  A benign answer.\n'}]}


def compare_author(payload):
    results = []
    reference_python = payload['reference_python']
    assert Path(reference_python).is_file(), f'Set {payload["method"].upper()}_AUTHOR_PYTHON to its pinned SDK baseline.'
    for reference, python in ((True, reference_python), (False, payload['shared_python'])):
        completed = subprocess.run([python, str(ROOT / 'tests/author_execution_harness.py')],
                                   input=json.dumps({**payload, 'reference': reference}),
                                   capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        results.append(json.loads(completed.stdout))
    original, adapted = results
    # Compare all author request fields after accounting only for declared transport changes.
    expected = []
    for request in original['wire']:
        body = dict(request['body'])
        # The original judge prompt identifies its role even when source model names coincide.
        messages = body['messages']
        judge = (messages[0]['role'] == 'system' and isinstance(messages[0]['content'], str)
                 and messages[0]['content'].startswith('You are a helpful red teaming assistant.'))
        route = payload['config']['judge'] if judge else payload['model']
        body['model'] = route['id']
        if 'system' in body:
            body['messages'] = [{'role': 'system', 'content': body.pop('system')}, *body['messages']]
        body['provider'] = {'only': [route['provider']], 'allow_fallbacks': False, 'require_parameters': True}
        expected.append({'url': 'https://openrouter.ai/api/v1/chat/completions', 'body': body})
    assert adapted['wire'] == expected
    assert adapted['sleeps'] == original['sleeps']
    assert adapted['sdk_sleeps'] == original['sdk_sleeps']
    assert adapted.get('error', {}).get('type') == original.get('error', {}).get('type')
    if original.get('error', {}).get('type') in {'UnboundLocalError', 'IndexError', 'ValueError'}:
        assert adapted['error'] == original['error']
    if 'error' not in original:
        assert [run['author_result'] for run in adapted['runs']] == original['records']
    return original, adapted
