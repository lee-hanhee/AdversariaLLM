# Preserved author implementations

Each attack directory contains author source preserved according to its
`verification/<attack>/SOURCE.json` manifest. Attribution headers and declared
extractions, such as notebook cells or removing a standalone demo, are recorded
there. Preserve the remaining source bytes, including whitespace and quirks.

Our executable integration code lives separately in
`adversariallm/attacks/<attack>.py`. Shared loading, routing, and result bookkeeping live in `adversariallm/attacks/utils/author_runtime.py`.

`verification/<attack>/reference/` contains the original files used for independent
comparisons. Run `python verification/verify_source.py` to check all manifests;
see the [verification guide](../../../verification/verification.md) for official-source checks.
