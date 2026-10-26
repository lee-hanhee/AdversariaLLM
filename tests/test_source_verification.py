import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'verification/verify_source.py'
spec = importlib.util.spec_from_file_location('shared_source_verification', SCRIPT)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


@pytest.fixture
def packages(tmp_path, monkeypatch):
    monkeypatch.setattr(verifier, 'ROOT', tmp_path)

    def make(name, prefix=False):
        base = tmp_path / 'verification' / name
        base.mkdir(parents=True)
        body = b'VALUE = "rainbows"\n'
        original = body + b'\nif __name__ == "__main__":\n    print(VALUE)\n' if prefix else body
        marker = 'if __name__ == "__main__":'
        expected = original.split(marker.encode(), 1)[0] if prefix else original
        runtime = b'"""Attribution."""\n\n' + expected if prefix else original
        reference_path = f'verification/{name}/original.py'
        local_path = f'{name}.py'
        diff_path = f'verification/{name}/adaptation.diff'
        (tmp_path / reference_path).write_bytes(original)
        (tmp_path / local_path).write_bytes(runtime)
        (tmp_path / diff_path).write_text(''.join(difflib.unified_diff(
            original.decode().splitlines(keepends=True), runtime.decode().splitlines(keepends=True),
            fromfile=reference_path, tofile=local_path,
        )))
        manifest = {
            'method': name, 'revision': 'pinned-revision',
            'files': [
                {'original_path': 'src/original.py', 'local_path': reference_path,
                 'sha256': hashlib.sha256(original).hexdigest()},
                {'local_path': local_path, 'derived_from': reference_path,
                 'sha256': hashlib.sha256(runtime).hexdigest()},
            ],
            'adaptations': [{'local_path': local_path, 'diff_path': diff_path}],
        }
        if prefix:
            manifest['adaptations'][0].update(stop_before=marker, prepend_docstring=True)
        path = base / 'SOURCE.json'
        path.write_text(json.dumps(manifest))
        return path, manifest

    return tmp_path, make


def save(path, manifest):
    path.write_text(json.dumps(manifest))


def replace_runtime(root, manifest, data):
    # Update the declared hash too: extraction checks must catch changes independently.
    entry = manifest['files'][1]
    (root / entry['local_path']).write_bytes(data)
    entry['sha256'] = hashlib.sha256(data).hexdigest()


def test_two_different_methods_share_the_same_verifier(packages):
    _, make = packages
    for name, prefix in [('alpha', False), ('beta', True)]:
        path, _ = make(name, prefix)
        verifier.verify(path)


def test_changed_reference_is_rejected(packages):
    root, make = packages
    path, manifest = make('alpha')
    (root / manifest['files'][0]['local_path']).write_text('Changed reference')
    with pytest.raises(ValueError, match='Hash mismatch'):
        verifier.verify(path)


@pytest.mark.parametrize('tampering', ['body', 'whitespace', 'extra_code', 'missing_header'])
def test_extraction_rejects_tampering_even_with_updated_hash(packages, tampering):
    root, make = packages
    path, manifest = make('beta', prefix=True)
    runtime = (root / manifest['files'][1]['local_path']).read_bytes()
    if tampering == 'body':
        runtime = runtime.replace(b'rainbows', b'clouds')
    elif tampering == 'whitespace':
        runtime = runtime.replace(b'VALUE =', b'VALUE  =')
    elif tampering == 'extra_code':
        runtime = runtime.replace(b'VALUE =', b'ADDED = True\nVALUE =')
    else:
        runtime = runtime.split(b'\n\n', 1)[1]
    replace_runtime(root, manifest, runtime)
    save(path, manifest)
    with pytest.raises(ValueError, match='Original bytes changed|Extra syntax|Expected an attribution'):
        verifier.verify(path)


def test_recorded_diff_cannot_become_stale(packages):
    root, make = packages
    path, manifest = make('beta', prefix=True)
    (root / manifest['adaptations'][0]['diff_path']).write_text('Outdated diff')
    with pytest.raises(ValueError, match='Outdated adaptation diff'):
        verifier.verify(path)


@pytest.mark.parametrize('issue', ['missing_rule', 'unknown_rule', 'missing_marker', 'wrong_reference'])
def test_incomplete_or_unsupported_extraction_rules_fail(packages, issue):
    _, make = packages
    path, manifest = make('beta', prefix=True)
    if issue == 'missing_rule':
        manifest['adaptations'] = []
    elif issue == 'unknown_rule':
        manifest['adaptations'][0]['rewrite_imports'] = True
    elif issue == 'missing_marker':
        manifest['adaptations'][0]['stop_before'] = 'missing marker'
    else:
        manifest['files'][1]['derived_from'] = 'missing-reference.py'
    save(path, manifest)
    with pytest.raises(ValueError):
        verifier.verify(path)


def test_official_comparison_uses_the_pin_and_original_path(packages, monkeypatch):
    root, make = packages
    path, manifest = make('alpha')
    original = (root / manifest['files'][0]['local_path']).read_bytes()
    calls = []

    def show(command):
        calls.append(command)
        return original

    monkeypatch.setattr(verifier.subprocess, 'check_output', show)
    verifier.verify(path, official_repo='/independent/clone')
    assert calls == [['git', '-C', '/independent/clone', 'show', 'pinned-revision:src/original.py']]
    monkeypatch.setattr(verifier.subprocess, 'check_output', lambda command: b'Different official bytes')
    with pytest.raises(ValueError, match='Official source mismatch'):
        verifier.verify(path, official_repo='/independent/clone')


def test_checks_remain_active_under_python_optimization(packages):
    root, make = packages
    path, manifest = make('alpha')
    (root / manifest['files'][0]['local_path']).write_text('Changed reference')
    code = (
        'import importlib.util\n'
        f'spec = importlib.util.spec_from_file_location("verifier", {str(SCRIPT)!r})\n'
        'module = importlib.util.module_from_spec(spec)\n'
        'spec.loader.exec_module(module)\n'
        f'module.ROOT = module.Path({str(root)!r})\n'
        f'module.verify({str(path)!r})\n'
    )
    result = subprocess.run([sys.executable, '-O', '-c', code], capture_output=True, text=True)
    assert result.returncode != 0 and 'Hash mismatch' in result.stderr
