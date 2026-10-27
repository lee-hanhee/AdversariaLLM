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
    # Keep all source-verifier fixture reads and writes inside the temporary package.
    monkeypatch.setattr(verifier, 'ROOT', tmp_path)

    def make(name, prefix=False):
        # Create original and runtime artifacts for exact-copy or prefix-extraction cases.
        base = tmp_path / 'verification' / name
        base.mkdir(parents=True)
        body = b'VALUE = "rainbows"\n'
        original = body + b'\nif __name__ == "__main__":\n    print(VALUE)\n' if prefix else body
        marker = 'if __name__ == "__main__":'
        expected = original.split(marker.encode(), 1)[0] if prefix else original
        runtime = b'"""Attribution."""\n\n' + expected if prefix else original

        # Give each artifact a manifest path and an independently recorded hash.
        reference_path = f'verification/{name}/original.py'
        local_path = f'{name}.py'
        diff_path = f'verification/{name}/adaptation.diff'
        (tmp_path / reference_path).write_bytes(original)
        (tmp_path / local_path).write_bytes(runtime)
        (tmp_path / diff_path).write_text(''.join(difflib.unified_diff(
            original.decode().splitlines(keepends=True), runtime.decode().splitlines(keepends=True),
            fromfile=reference_path, tofile=local_path,
        )))

        # Describe original-to-local correspondence and the recorded extraction diff.
        manifest = {
            'method': name, 'revision': 'pinned-revision',
            'files': [{
                'original_path': 'src/original.py',
                'reference': {'path': reference_path, 'sha256': hashlib.sha256(original).hexdigest()},
                'implementations': [{
                    'path': local_path, 'sha256': hashlib.sha256(runtime).hexdigest(),
                    'adaptation': {'diff_path': diff_path},
                }],
            }],
        }

        # Declare the allowed attribution/header extraction for prefix fixtures.
        if prefix:
            manifest['files'][0]['implementations'][0]['adaptation'].update(
                stop_before=marker, prepend_docstring=True)

        # Write a complete manifest that the shared verifier can consume.
        path = base / 'SOURCE.json'

        # Save fixture metadata independently of the artifact files under test.
        path.write_text(json.dumps(manifest))
        return path, manifest

    return tmp_path, make


def save(path, manifest):
    path.write_text(json.dumps(manifest))


def replace_runtime(root, manifest, data):
    # Update the declared hash too: extraction checks must catch changes independently.
    entry = manifest['files'][0]['implementations'][0]
    (root / entry['path']).write_bytes(data)
    entry['sha256'] = hashlib.sha256(data).hexdigest()


def test_two_different_methods_share_the_same_verifier(packages):
    # Build a temporary source package with explicit hashes and extraction evidence.
    _, make = packages
    for name, prefix in [('alpha', False), ('beta', True)]:
        path, _ = make(name, prefix)
        verifier.verify(path)


def test_changed_reference_is_rejected(packages):
    # Build a temporary source package with explicit hashes and extraction evidence.
    root, make = packages
    path, manifest = make('alpha')
    (root / manifest['files'][0]['reference']['path']).write_text('Changed reference')

    # Reject a reference whose bytes no longer match its declared hash.
    with pytest.raises(ValueError, match='Hash mismatch'):
        verifier.verify(path)


@pytest.mark.parametrize('tampering', ['body', 'whitespace', 'extra_code', 'missing_header'])
def test_extraction_rejects_tampering_even_with_updated_hash(packages, tampering):
    # Build a temporary source package with explicit hashes and extraction evidence.
    root, make = packages
    path, manifest = make('beta', prefix=True)
    runtime = (root / manifest['files'][0]['implementations'][0]['path']).read_bytes()
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

    # Detect changed author bytes independently of an updated runtime hash.
    with pytest.raises(ValueError, match='Original bytes changed|Extra syntax|Expected an attribution'):
        verifier.verify(path)


def test_recorded_diff_cannot_become_stale(packages):
    # Build a temporary source package with explicit hashes and extraction evidence.
    root, make = packages
    path, manifest = make('beta', prefix=True)
    (root / manifest['files'][0]['implementations'][0]['adaptation']['diff_path']).write_text('Outdated diff')

    # Reject a recorded extraction diff that no longer matches the artifacts.
    with pytest.raises(ValueError, match='Outdated adaptation diff'):
        verifier.verify(path)


@pytest.mark.parametrize('issue', ['missing_rule', 'unknown_rule', 'missing_marker', 'missing_reference'])
def test_incomplete_or_unsupported_extraction_rules_fail(packages, issue):
    # Build a temporary source package with explicit hashes and extraction evidence.
    _, make = packages
    path, manifest = make('beta', prefix=True)
    if issue == 'missing_rule':
        del manifest['files'][0]['implementations'][0]['adaptation']
    elif issue == 'unknown_rule':
        manifest['files'][0]['implementations'][0]['adaptation']['rewrite_imports'] = True
    elif issue == 'missing_marker':
        manifest['files'][0]['implementations'][0]['adaptation']['stop_before'] = 'missing marker'
    else:
        del manifest['files'][0]['reference']
    save(path, manifest)

    # Require a complete, supported extraction declaration.
    with pytest.raises(ValueError):
        verifier.verify(path)


def test_official_comparison_uses_the_pin_and_original_path(packages, monkeypatch):
    # Build a temporary source package with explicit hashes and extraction evidence.
    root, make = packages
    path, manifest = make('alpha')
    original = (root / manifest['files'][0]['reference']['path']).read_bytes()
    calls = []

    def show(command):
        calls.append(command)
        return original

    monkeypatch.setattr(verifier.subprocess, 'check_output', show)
    verifier.verify(path, official_repo='/independent/clone')

    # Read the declared pinned Git object and reject mismatched official bytes.
    assert calls == [['git', '-C', '/independent/clone', 'show', 'pinned-revision:src/original.py']]
    monkeypatch.setattr(verifier.subprocess, 'check_output', lambda command: b'Different official bytes')
    with pytest.raises(ValueError, match='Official source mismatch'):
        verifier.verify(path, official_repo='/independent/clone')


def test_checks_remain_active_under_python_optimization(packages):
    # Build a temporary source package with explicit hashes and extraction evidence.
    root, make = packages
    path, manifest = make('alpha')
    (root / manifest['files'][0]['reference']['path']).write_text('Changed reference')
    code = (
        'import importlib.util\n'
        f'spec = importlib.util.spec_from_file_location("verifier", {str(SCRIPT)!r})\n'
        'module = importlib.util.module_from_spec(spec)\n'
        'spec.loader.exec_module(module)\n'
        f'module.ROOT = module.Path({str(root)!r})\n'
        f'module.verify({str(path)!r})\n'
    )
    result = subprocess.run([sys.executable, '-O', '-c', code], capture_output=True, text=True)

    # Keep source checks active when Python assertions are optimized away.
    assert result.returncode != 0 and 'Hash mismatch' in result.stderr


def test_recorded_diff_preserves_crlf_source_lines(packages):
    # Build a temporary source package with explicit hashes and extraction evidence.
    root, make = packages
    path, manifest = make('crlf', prefix=True)
    artifacts = [manifest['files'][0]['reference'], *manifest['files'][0]['implementations']]
    for entry in artifacts:
        artifact = root / entry['path']
        data = artifact.read_bytes().replace(b'\n', b'\r\n')
        artifact.write_bytes(data)
        entry['sha256'] = hashlib.sha256(data).hexdigest()
    original, runtime = [(root / entry['path']).read_bytes().decode() for entry in artifacts]
    rule = manifest['files'][0]['implementations'][0]['adaptation']
    difference = ''.join(difflib.unified_diff(original.splitlines(keepends=True), runtime.splitlines(keepends=True),
                                            fromfile=artifacts[0]['path'], tofile=artifacts[1]['path']))
    (root / rule['diff_path']).write_bytes(difference.encode())
    save(path, manifest)
    verifier.verify(path)
    (root / rule['diff_path']).write_bytes(difference.replace('\r\n', '\n').encode())

    # Verify recorded diffs against the exact original line endings.
    with pytest.raises(ValueError, match='Outdated adaptation diff'):
        verifier.verify(path)


def test_reference_only_entry_and_multiple_implementations(packages):
    # Build a temporary source package with explicit hashes and extraction evidence.
    root, make = packages
    path, manifest = make('alpha')
    entry = manifest['files'][0]
    original = (root / entry['reference']['path']).read_bytes()
    # A second implementation stays attached to the same original without repeating its reference.
    (root / 'second.py').write_bytes(original)
    (root / 'second.diff').write_bytes(b'')
    entry['implementations'].append({'path': 'second.py', 'sha256': hashlib.sha256(original).hexdigest(),
                                     'adaptation': {'diff_path': 'second.diff'}})
    (root / 'LICENSE').write_bytes(b'License notice')
    manifest['files'].append({'original_path': 'LICENSE', 'reference': {
        'path': 'LICENSE', 'sha256': hashlib.sha256(b'License notice').hexdigest()}})
    save(path, manifest)
    verifier.verify(path)


@pytest.mark.parametrize('duplicate', ['original', 'reference', 'implementation'])
def test_duplicate_correspondence_is_rejected(packages, duplicate):
    # Build a temporary source package with explicit hashes and extraction evidence.
    _, make = packages
    path, manifest = make('alpha')
    entry = manifest['files'][0]
    if duplicate == 'original':
        manifest['files'].append(entry.copy())
    elif duplicate == 'reference':
        entry['implementations'][0]['path'] = entry['reference']['path']
    else:
        entry['implementations'].append(entry['implementations'][0].copy())
    save(path, manifest)

    # Reject duplicate original or local artifact identities.
    with pytest.raises(ValueError, match='Duplicate'):
        verifier.verify(path)


@pytest.fixture
def notebook_package(packages):
    root, make = packages
    path, manifest = make('notebook')
    entry = manifest['files'][0]

    # Mix code and markdown cells to exercise ordered, code-only notebook extraction.
    notebook = {'cells': [{'cell_type': 'code', 'source': ['VALUE = "rainbows"\n']},
                          {'cell_type': 'markdown', 'source': ['Description']},
                          {'cell_type': 'code', 'source': 'SECOND = 2'}]}
    original = json.dumps(notebook).encode()
    (root / entry['reference']['path']).write_bytes(original)
    entry['reference']['sha256'] = hashlib.sha256(original).hexdigest()

    # Build the exact declared cell concatenation with a separate attribution header.
    runtime = b'"""Attribution."""\n\nVALUE = "rainbows"\n\n\nSECOND = 2'
    replace_runtime(root, manifest, runtime)
    rule = entry['implementations'][0]['adaptation']
    rule.update(notebook_cells=[0, 2], prepend_docstring=True, reference_label='notebook:cells=0,2')
    source = 'VALUE = "rainbows"\n\n\nSECOND = 2'

    # Record a diff against extracted code rather than the complete notebook JSON.
    difference = ''.join(difflib.unified_diff(source.splitlines(keepends=True), runtime.decode().splitlines(keepends=True),
                                            fromfile=rule['reference_label'], tofile=entry['implementations'][0]['path']))
    (root / rule['diff_path']).write_bytes(difference.encode())
    save(path, manifest)
    return root, path, manifest


def test_exact_notebook_extraction_and_cell_diff_are_verified(notebook_package):
    # Build a temporary source package with explicit hashes and extraction evidence.
    _, path, _ = notebook_package
    verifier.verify(path)


@pytest.mark.parametrize('indices', [[], [True], [-1], [3], [1], [2, 0], [0, 0], '0'])
def test_invalid_notebook_cell_selection_is_rejected(notebook_package, indices):
    # Build a temporary source package with explicit hashes and extraction evidence.
    _, path, manifest = notebook_package
    manifest['files'][0]['implementations'][0]['adaptation']['notebook_cells'] = indices
    save(path, manifest)

    # Reject invalid, duplicate, reordered, or non-code cell selections.
    with pytest.raises(ValueError):
        verifier.verify(path)


def test_changed_cell_text_is_rejected_even_when_reference_hash_is_updated(notebook_package):
    # Build a temporary source package with explicit hashes and extraction evidence.
    root, path, manifest = notebook_package
    reference = manifest['files'][0]['reference']
    original = (root / reference['path']).read_bytes().replace(b'rainbows', b'clouds')
    (root / reference['path']).write_bytes(original)
    reference['sha256'] = hashlib.sha256(original).hexdigest()
    save(path, manifest)

    # Detect altered cell content even after the reference hash is refreshed.
    with pytest.raises(ValueError, match='Original bytes changed'):
        verifier.verify(path)
