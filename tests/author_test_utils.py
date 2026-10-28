"""Fixtures for original-versus-integrated author behavior comparisons."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

from adversariallm.attacks.flip_attack import FlipAttackConfig
from adversariallm.attacks.equacode import EquaCodeConfig
from adversariallm.attacks.gta import GTAConfig
from adversariallm.attacks.mousetrap import MousetrapConfig
from adversariallm.attacks.query_attack import QueryAttackConfig


ROOT = Path(__file__).resolve().parents[1]


def author_payload(method, **overrides):
    # Environment paths are explicit so missing author environments fail instead of skipping evidence.
    env_name = f'{method.upper()}_AUTHOR_PYTHON'
    python = os.environ.get(env_name, str(ROOT / '.author_envs' / method / 'bin/python'))

    # Build attack defaults with fake, explicit routes for each required role.
    configs = {'flip_attack': FlipAttackConfig, 'equacode': EquaCodeConfig,
               'gta': GTAConfig, 'mousetrap': MousetrapConfig, 'query_attack': QueryAttackConfig}
    config = asdict(configs[method]())
    config.update(overrides)
    model = {'backend': 'openrouter', 'id': 'test/target', 'provider': 'target-provider',
             'author_model': {'flip_attack': 'gpt-4-0613', 'equacode': 'gpt-3.5-turbo',
                              'gta': 'deepseek_r1', 'mousetrap': 'o1-2024-12-17',
                              'query_attack': 'gpt-4-1106-preview'}[method]}
    if method == 'query_attack':
        config['translator'] = {'id': 'test/translator', 'provider': 'translator-provider'}
    else:
        config['judge'] = {'id': 'test/judge', 'provider': 'judge-provider'}
    if method == 'gta':
        config['attacker'] = {'id': 'test/attacker', 'provider': 'attacker-provider'}

    # Use the declared shared comparison environment for releases without exact SDK pins.
    if method in ('gta', 'mousetrap'):
        python = os.environ.get('AUTHOR_SHARED_PYTHON', sys.executable)

    if method == 'query_attack':
        shared_python = os.environ.get('QUERY_ATTACK_SHARED_PYTHON', sys.executable)
    else:
        shared_python = os.environ.get('AUTHOR_SHARED_PYTHON', sys.executable)

    # Keep successful Mousetrap fixtures ASCII; dedicated cases check original Unicode quirks.
    question = 'Explain rainbows through café windows.\nKeep {braces}.'
    if method == 'mousetrap':
        question = 'Explain rainbows through cafe windows.\nKeep {braces}.'

    # Package the benign dataset, scripted replies, and interpreter choices for the harness.
    return {'method': method, 'model': model, 'config': config,
            'reference_python': python, 'shared_python': shared_python,
            'dataset': [[{'role': 'user', 'content': question},
                         {'role': 'assistant', 'content': 'Reference answer.'}]],
            'replies': [{'text': '  A benign answer.\n'}]}


def compare_author(payload):
    # Run the reference and integrated paths in separate interpreters with identical inputs.
    results = []
    reference_python = payload['reference_python']
    assert Path(reference_python).is_file(), f'Set {payload["method"].upper()}_AUTHOR_PYTHON to its pinned SDK baseline.'
    for reference, python in ((True, reference_python), (False, payload['shared_python'])):
        completed = subprocess.run([python, str(ROOT / 'tests/author_execution_harness.py')],
                                   input=json.dumps({**payload, 'reference': reference}),
                                   capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        results.append(json.loads(completed.stdout))

    # Unpack both results before applying only the declared transport translations.
    original, adapted = results
    # Compare all author request fields after accounting only for declared transport changes.
    expected = []
    for request in original['wire']:
        body = dict(request['body'])
        # The original judge prompt identifies its role even when source model names coincide.
        messages = body['messages']
        judge = (messages[0]['role'] == 'system' and isinstance(messages[0]['content'], str)
                 and messages[0]['content'].startswith('You are a helpful red teaming assistant.'))
        if payload['method'] == 'gta':
            role = request['role']
            route = {'Police': payload['config']['attacker'], 'Judge': payload['config']['judge'],
                     'Thomas': payload['model'], 'Marcos': payload['model']}[role]
        elif payload['method'] == 'query_attack':
            route = payload['model'] if len(messages) == 4 else payload['config']['translator']
        elif payload['method'] == 'mousetrap':
            route = payload['config']['judge'] if messages[0]['role'] == 'system' else payload['model']
        else:
            route = payload['config']['judge'] if judge else payload['model']

        # Translate endpoint/model routing and Claude system layout without changing prompt content.
        body['model'] = route['id']
        if 'system' in body:
            body['messages'] = [{'role': 'system', 'content': body.pop('system')}, *body['messages']]
        body['provider'] = {'only': [route['provider']], 'allow_fallbacks': False, 'require_parameters': True}
        expected.append({'url': 'https://openrouter.ai/api/v1/chat/completions', 'body': body})
        if payload['method'] == 'gta':
            expected[-1]['role'] = request['role']

    # Compare request controls, retries, failures, and original result records.
    assert adapted['wire'] == expected
    assert adapted['sleeps'] == original['sleeps']
    assert adapted['sdk_sleeps'] == original['sdk_sleeps']
    assert adapted.get('error', {}).get('type') == original.get('error', {}).get('type')
    if original.get('error', {}).get('type') in {'UnboundLocalError', 'IndexError', 'ValueError'}:
        assert adapted['error'] == original['error']
    if payload['method'] == 'mousetrap':
        assert adapted['rng_state'] == original['rng_state']
    if 'error' not in original:
        assert adapted['records'] == original['records']
        if payload['method'] == 'query_attack':
            assert adapted['author_summary'] == original['author_summary']
    return original, adapted
