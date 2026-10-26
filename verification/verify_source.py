import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess


# All manifest paths are relative to the repository, regardless of the caller's cwd.
BASE = Path(__file__).resolve().parent
ROOT = BASE.parent


def require(condition, message):
    """Keep verification active even when Python runs with optimization enabled."""
    if not condition:
        raise ValueError(message)


def verify(manifest_path, official_repo=None):
    manifest = json.loads(Path(manifest_path).read_text())
    paths, originals = set(), set()

    def read_artifact(artifact):
        # Check hashes and unique local paths before trusting comparison evidence.
        path = artifact['path']
        require(path not in paths, f'Duplicate artifact path: {path}')
        paths.add(path)
        data = (ROOT / path).read_bytes()
        require(hashlib.sha256(data).hexdigest() == artifact['sha256'], f'Hash mismatch: {path}')
        return data

    for entry in manifest['files']:
        # Each original owns its bundled reference and all derived implementations.
        original_path = entry['original_path']
        require(original_path not in originals, f'Duplicate original path: {original_path}')
        originals.add(original_path)
        reference = entry.get('reference')
        require(isinstance(reference, dict), f'Missing reference: {original_path}')
        original = read_artifact(reference)
        if official_repo:
            official = subprocess.check_output([
                'git', '-C', str(official_repo), 'show',
                f'{manifest["revision"]}:{original_path}',
            ])
            require(original == official, f'Official source mismatch: {original_path}')

        for implementation in entry.get('implementations', []):
            local_path = implementation['path']
            runtime = read_artifact(implementation)
            rule = implementation.get('adaptation')
            require(isinstance(rule, dict) and 'diff_path' in rule,
                    f'Missing adaptation/diff: {local_path}')
            allowed = {'diff_path', 'stop_before', 'prepend_docstring', 'reference_label'}
            require(not set(rule) - allowed, f'Unsupported adaptation fields: {set(rule) - allowed}')

            # Support unchanged copies or exact prefixes ending at one unique marker.
            expected = original
            if 'stop_before' in rule:
                marker = rule['stop_before'].encode()
                require(bool(marker) and original.count(marker) == 1,
                        f'Extraction marker must occur exactly once: {reference["path"]}')
                expected = original.split(marker, 1)[0]

            # An optional Python attribution docstring may precede otherwise unchanged author bytes.
            if rule.get('prepend_docstring', False):
                require(runtime.endswith(expected), f'Original bytes changed: {local_path}')
                tree = ast.parse(runtime)
                require(bool(tree.body) and isinstance(tree.body[0], ast.Expr)
                        and isinstance(tree.body[0].value, ast.Constant)
                        and isinstance(tree.body[0].value.value, str),
                        f'Expected an attribution docstring: {local_path}')
                tree.body.pop(0)
                require(ast.dump(tree) == ast.dump(ast.parse(expected)), f'Extra syntax: {local_path}')
            else:
                require(runtime == expected, f'Original bytes changed: {local_path}')

            # Regenerate each recorded diff so it stays consistent with the files under review.
            difference = ''.join(difflib.unified_diff(
                original.decode().splitlines(keepends=True), runtime.decode().splitlines(keepends=True),
                fromfile=rule.get('reference_label', reference['path']),
                tofile=local_path,
            ))
            require((ROOT / rule['diff_path']).read_bytes().decode() == difference,
                    f'Outdated adaptation diff: {rule["diff_path"]}')

    # Report bundled consistency separately from comparison against an independent official clone.
    print(f'{manifest["method"]}: hashes and declared extraction/diff checks passed.')
    if official_repo:
        print('Reference bytes also match the pinned revision in the supplied official clone.')
    else:
        print('Bundled evidence only; official source origin was not independently checked.')


if __name__ == '__main__':
    # Select one method, or check every source package currently present under verification/.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attack', nargs='?', help='Method directory; omit to check all bundled manifests')
    parser.add_argument('--official-repo', type=Path, help='Independent official clone for one selected method')
    args = parser.parse_args()
    if args.official_repo and not args.attack:
        parser.error('--official-repo requires a selected attack')
    manifests = [BASE / args.attack / 'SOURCE.json'] if args.attack else sorted(BASE.glob('*/SOURCE.json'))
    if not manifests:
        parser.error('No source manifests found')
    try:
        for manifest_path in manifests:
            verify(manifest_path, args.official_repo)
    except (ValueError, KeyError, OSError, SyntaxError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Verification failed: {error}\n')
