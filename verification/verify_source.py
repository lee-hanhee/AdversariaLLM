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
    # Check every artifact's hash before trusting it as comparison evidence.
    manifest = json.loads(Path(manifest_path).read_text())
    files = {entry['local_path']: entry for entry in manifest['files']}
    require(len(files) == len(manifest['files']), 'Duplicate artifact paths in manifest')
    contents = {}
    for path, entry in files.items():
        data = (ROOT / path).read_bytes()
        contents[path] = data
        require(hashlib.sha256(data).hexdigest() == entry['sha256'], f'Hash mismatch: {path}')

        # Derived runtime files are compared below; original references can be checked against Git.
        if official_repo and 'derived_from' not in entry:
            official = subprocess.check_output([
                'git', '-C', str(official_repo), 'show',
                f'{manifest["revision"]}:{entry["original_path"]}',
            ])
            require(data == official, f'Official source mismatch: {entry["original_path"]}')

    # Every derived artifact must have an explicit, supported extraction comparison.
    adaptations = manifest.get('adaptations', [])
    derived = {path for path, entry in files.items() if 'derived_from' in entry}
    declared = [rule['local_path'] for rule in adaptations]
    require(len(declared) == len(set(declared)) and set(declared) == derived,
            'Derived artifacts and adaptation rules must match exactly')

    for rule in adaptations:
        # Support unchanged copies or exact prefixes; other extractions need explicit new rules.
        allowed = {'local_path', 'diff_path', 'stop_before',
                   'prepend_docstring', 'reference_label'}
        require(not set(rule) - allowed, f'Unsupported adaptation fields: {set(rule) - allowed}')
        local_path = rule['local_path']
        reference_path = files[local_path]['derived_from']
        require(reference_path in files and 'derived_from' not in files[reference_path],
                f'Extraction needs an original reference: {reference_path}')
        original, runtime = contents[reference_path], contents[local_path]
        expected = original
        if 'stop_before' in rule:
            marker = rule['stop_before'].encode()
            require(bool(marker) and original.count(marker) == 1,
                    f'Extraction marker must occur exactly once: {reference_path}')
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
            fromfile=rule.get('reference_label', reference_path),
            tofile=local_path,
        ))
        require((ROOT / rule['diff_path']).read_text() == difference,
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
