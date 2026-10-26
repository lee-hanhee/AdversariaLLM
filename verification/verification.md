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
| `files` | Original/local paths and SHA-256 hashes of the recorded artifacts. |
| `derived_from` | The bundled original reference used to produce a runtime file. |
| `adaptations` | Extraction rules and the recorded diff for each runtime file; its reference comes from `derived_from`. |

With `--official-repo`, original reference files are also compared with the
pinned Git revision. Without it, passing checks establish bundled consistency
only, not independent source origin.

Currently supported extractions are exact copies or prefixes ending before a
unique `stop_before` marker, optionally preceded by a Python attribution
docstring. Recorded diffs must match the files. Notebook-cell extraction,
rewritten code, or other transformations require explicit additional support;
they are not automatically verified by a matching hash.

## 3. Review the adapter and compare behavior

Inspect author code and our wrapper separately. Preserve author algorithms,
prompts, whitespace, and redundant operations. Explain every extraction, changed
import, callback, or I/O replacement. Use concise comments and blank lines in
our code without editing the author reference.

Run the method's behavioral tests and prerequisite regressions using its recorded
commands and environment. Expected results must come from independent original
code or reference fixtures. Compare messages and request settings, including
omitted defaults. Where relevant, use identical scripted replies and fixed RNG
state to compare histories, retries, scoring, branching, stopping, and outputs.

## 4. Record what passed and what differs

Keep commands, environment, outcomes, departures, and remaining gaps in the
method's verification record or manifest. Distinguish source checks from
behavioral tests. State model/provider, dataset, judge, seed, generation, and
control-flow changes explicitly. Source identity and mocked comparisons do not
prove equivalent live model behavior or paper-result reproduction.
