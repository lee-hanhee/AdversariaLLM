# Verifying attack implementations

Use this process for every attack added to the repository. Verification answers
two separate questions: **does the source match the pinned author release?** and
**does our adapter preserve the behavior we claim?** Neither establishes that we
have reproduced the paper's benchmark results.

## 1. Identify the reference

Each attack has `verification/<attack>/SOURCE.json`. Review its official
repository, full commit SHA, source paths or notebook cells, hashes, attribution,
and license. Paths inside the manifest are relative to the repository root.

Clone the official repository separately and fetch the pinned commit. The
verifier reads that clone; it does not download or update it.

## 2. Verify source files

Run commands from this repository's root. Replace `ATTACK_NAME` with the directory
name under `verification/`, and `/path/to/official-clone` with your independent clone.

```bash
# Check one attack against its pinned official source.
python verification/verify_source.py ATTACK_NAME --official-repo /path/to/official-clone

# Check one attack's bundled files without an official clone.
python verification/verify_source.py ATTACK_NAME

# Check every bundled attack manifest without official clones.
python verification/verify_source.py
```

The checks are driven by each manifest:

| Field | What it describes |
| --- | --- |
| `files[].original_path` | Path in the pinned author repository. |
| `files[].reference` | Bundled original path and SHA-256 hash. |
| `files[].implementations[]` | Local implementation path and hash, grouped under its original. |
| `implementations[].adaptation` | Diff path and extraction rules for that implementation. |

A reference can have several implementations, or none for reference-only files
such as the license and dependency list. All correspondence is kept in one entry.

With `--official-repo`, original reference files are also compared with the
pinned Git revision. Without it, passing checks establish bundled consistency
only, not independent source origin.

Currently supported extractions are exact copies or prefixes ending before a
unique `stop_before` marker, or selected notebook code cells. Any extraction may
be preceded by a Python attribution docstring. `adaptation.notebook_cells`
lists zero-based cell indices in original order; their exact source strings are
joined with two newline characters. Each notebook diff compares that extracted
code with the local file, while the complete notebook is hashed and checked
against the official revision. Rewritten code and other transformations are not
automatically verified by a matching hash. Recorded diffs must match the files.

## 3. Review the adapter and compare behavior

Inspect author code and our wrapper separately. Preserve author algorithms,
prompts, whitespace, and redundant operations. Explain every extraction, changed
import, callback, or I/O replacement. Use concise comments and blank lines in
our code without editing the author reference.

Use separate SDK environments as [verification baselines](../docs/author_execution.md)
when needed. The runtime may use a shared SDK; record that difference explicitly. Run its behavioral tests and prerequisite
regressions using its recorded
commands and environment. Expected results must come from independent original
code or reference fixtures. Compare messages and request settings, including
omitted defaults. Where relevant, use identical scripted replies and fixed RNG
state to compare histories, retries, scoring, branching, stopping, and outputs.

## 4. Record what passed and what differs

Use the same metadata structure for every attack:

| Section | What to record |
| --- | --- |
| `repository_history` | Local project revisions used for the integration. Keep these separate from the top-level official author `revision`. |
| `integration.files[]` | Our integration files and each file's purpose. Author source correspondence stays in `files[]`. |
| `integration.preserved_behavior[]` | Author behavior that the integration retains. |
| `integration.changes[]` | One entry per changed area: `area`, `author`, `repository`, and `reason`. Use areas such as models/providers, dependencies, transport, data/storage, configuration, and reporting. |
| `verification_results.source_identity` | Source-check command, status, and result. |
| `verification_results.behavior` | Reference execution, compared behavior, allowed comparison differences, test runs, and how overlapping counts are handled. |
| `verification_results.environments` | Test-runner and baseline environments; link dependency locks where available. |
| `verification_results.execution_requirements` | Required setup and links to runnable instructions. |
| `verification_results.not_tested` / `limitations` | Remaining gaps and claims the checks do not establish. |

Record test runs separately rather than adding their pass counts together when
tests overlap. Keep source identity separate from behavioral evidence. Source
checks and mocked comparisons do not prove equivalent live model behavior or
paper-result reproduction.
