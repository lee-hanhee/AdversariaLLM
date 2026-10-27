# OpenRouter targets

The Direct baseline can generate through OpenRouter without loading local model
weights or a tokenizer. FlipAttack, EquaCode, GTA, and Mousetrap also route their targets and
author judges through OpenRouter; see [author attack execution](author_execution.md)
for their required routes and preserved control flow.
Existing local model configurations continue to use the local loader. Hosted
targets do not expose local losses or input token IDs. Runtime defenses and
unsupported attack/backend combinations are rejected before dataset loading.

## Configuration

Use the repository's existing installation and MongoDB configuration. Export
`OPENROUTER_API_KEY` in the process environment, or put it in the repository-root
`.env`; existing environment variables take precedence. `OPENROUTER_BASE_URL`
optionally overrides the default `https://openrouter.ai/api/v1` endpoint.

Select the hosted configuration explicitly and supply an exact model ID and one
provider slug. Neither has a default. This command previews the configuration
without executing requests:

```bash
python run_attacks.py --cfg job \
  hydra/launcher=basic models=openrouter model=openrouter attack=direct \
  models.openrouter.id=MODEL_ID models.openrouter.provider=PROVIDER_SLUG \
  classifiers=null
```

Replace the placeholders, select the dataset and indices using the normal runner
options, and remove `--cfg job` to execute. `classifiers=null` disables automatic
judging; omit that override to use the existing judge configuration. Results use
the existing JSON-file and MongoDB logging path, including its database
requirements. Configure MongoDB environment variables before starting the runner.

Provider routing uses `only: [PROVIDER_SLUG]`, `allow_fallbacks: false`, and
`require_parameters: true`. Per-call request bodies cannot replace the configured
provider pin. An optional `reasoning` mapping in the model configuration is
forwarded in the request body.

## Request and result behavior

- Explicit `openrouter/` routing removes only the client prefix from the model
  name; the remaining model ID is preserved. Legacy unprefixed client selection
  continues to work.
- Common generation controls are mapped to API parameters. OpenRouter receives
  `max_tokens` for `max_new_tokens`, and nonzero `top_k` in `extra_body`.
  Unspecified settings remain omitted. Direct supplies its existing generation
  defaults.
- Multiple return sequences use separate requests per input, in input/sample
  order. Each OpenRouter completion has corresponding request metadata.
- Empty trailing assistant placeholders and local batching options are removed.
  Nonempty assistant prefixes require `supports_assistant_prefill: true` in the
  model configuration; enable this only for a target supporting that capability.
- Hosted request failures raise after application retries. Empty or non-text
  completions raise immediately, retaining usage and finish reasons on the
  generator for diagnosis. Legacy API generators retain their default behavior
  of returning an empty response after exhausted retries.
- `GenerationResult.api_metadata` is grouped by input and API request; each
  record contains response ID, model, provider when supplied, usage, and finish
  reasons. Direct copies each input's records to `AttackStepResult.api_metadata`.
  For legacy APIs returning multiple choices in one request, one metadata record
  describes that request and contains all its finish reasons.

## Extraction and verification

This slice extracts hosted support from preserved revision
`08484c461824be87a6b63bed24ff1d546ed49732`, onto upstream base
`215c17df719ed8a3a7bc058f3beff5c151bc7963`. It introduces no author attack
algorithm, prompt, template, or reference fixture. Direct receives interface
adaptations for optional local measurements and API metadata.

The extraction excludes native-chat support, run guards, optional JSON storage,
checkpoint callbacks, campaign tooling, and later attack registrations. The
existing duplicate retry declarations remain; the effective declaration uses
default factories so it can be imported on Python 3.11.

Focused offline regression suite:

```bash
python -m pytest -q \
  tests/test_api_clients.py tests/test_api_target.py \
  tests/test_run_attacks_defenses.py tests/test_defense_compat.py \
  tests/test_defenses.py tests/test_compatibility_contract.py \
  tests/test_attacks/test_direct_transfer_runtime.py \
  tests/test_attacks/test_bon_runtime.py
```

Recorded validation: **52 passed**, no skips, on Python 3.11.13, Torch 2.2.2,
Transformers 5.9.0, OpenAI SDK 1.109.1, pytest 8.4.2, Hydra 1.3.7, and
OmegaConf 2.3.1. Tests mock API requests, model loading, and database access.
The Hydra composition test explicitly selects the basic launcher.

The available Intel macOS environment differs from the repository's supported
Linux environment. Transformers disables model loading with this older Torch,
and upstream's `recompile_limit` setting is unavailable. For this recorded run,
the following test-process-only compatibility setup was used, with
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
`PYTHONDONTWRITEBYTECODE=1`, and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python - <<'PY'
import torch
import pytest

torch._dynamo.config._allowed_keys.add("recompile_limit")
torch._dynamo.config._config.setdefault("recompile_limit", 512)
raise SystemExit(pytest.main([
    "-q", "tests/test_api_clients.py", "tests/test_api_target.py",
    "tests/test_run_attacks_defenses.py", "tests/test_defense_compat.py",
    "tests/test_defenses.py", "tests/test_compatibility_contract.py",
    "tests/test_attacks/test_direct_transfer_runtime.py",
    "tests/test_attacks/test_bon_runtime.py",
]))
PY
```

This validates mocked transport and runner contracts. Live provider execution,
MongoDB integration, GPU inference, and automatic judges were not exercised.
