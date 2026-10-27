# Running author attacks through OpenRouter

FlipAttack and EquaCode execute the pinned author attack logic with an OpenRouter
transport binding. Targets and author judges use `openai==1.109.1` as a client
for `https://openrouter.ai/api/v1`. Only `OPENROUTER_API_KEY` is needed; no direct
OpenAI, Anthropic, or DeepInfra credentials are used.

## Configuration

Use the normal repository dependency setup and MongoDB configuration. The
repository lock already selects OpenAI 1.109.1; no extra runtime environment or
requirements file is needed. Both target and judge
require an explicit OpenRouter model ID and one provider slug. There are no
routing defaults, including for FlipAttack's disabled-by-default judge.

Preview a configuration without making requests:

```bash
python run_attacks.py --cfg job hydra/launcher=basic \
  models=equacode_author model=equacode_author attack=equacode \
  models.equacode_author.id=TARGET_MODEL_ID \
  models.equacode_author.provider=TARGET_PROVIDER \
  attacks.equacode.judge.id=JUDGE_MODEL_ID \
  attacks.equacode.judge.provider=JUDGE_PROVIDER \
  classifiers=null
```

For FlipAttack, replace `equacode` with `flip_attack` throughout. Set
`attacks.flip_attack.evaluate=true` to enable its author judge. Remove `--cfg job`
to run after selecting the dataset and indices normally. `classifiers=null`
disables the runner's separate post-run classifiers; it does not disable
EquaCode's mandatory author judge or FlipAttack's enabled author judge.

The model configuration's `author_model` is the source-level name used by the
unchanged author code. It is **not** sent to OpenRouter. FlipAttack uses it to
choose GPT/Claude/Llama/Mixtral handling; EquaCode uses its case-sensitive name
check when constructing the system message. `judge_model` likewise retains the
source label, while `judge.id` determines actual routing. Selecting a different
hosted model does not make it equivalent to the author's original model.

Each request pins `provider.only`, disables fallback, and requires parameter
support. Unsupported models or parameters surface as errors through the author's
retry behavior. No additional application retry layer is added.

## Preserved behavior and transport differences

| Area | Behavior |
| --- | --- |
| Source | Builder, driver, client logic, and evaluator definitions retain author bytes after attribution headers. Only required definitions and the per-example driver are executed. |
| FlipAttack | Original guidance, temperature, ignored GPT output-cap option, application retries and sleeps, GPT/Llama/Mixtral stripping, and unstripped Claude replies. |
| FlipAttack judging | Optional original keyword check and GPT rating parser; original judge client defaults remain independent of target overrides. |
| EquaCode | Four original templates, literal substitution, raw target text, mandatory judging, and up to six judge calls until a reply contains case-sensitive `Rating`. |
| Transport | OpenRouter replaces original endpoints and model IDs. Claude's separate system field becomes a system chat message; its 3,584-token cap is retained. Native Anthropic serialization is not reproduced. |
| SDK | Shared OpenAI 1.109.1 replaces the authors' pinned SDKs; SDK retries may differ for failures outside the tested cases. |
| Integration | Repository datasets and results replace CSV loading, spreadsheet output, and CLI reporting. No RNG seed or generic generation overrides are introduced. |

The adapter runs directly in the runner and reuses the existing `APITarget` SDK
client. `OpenRouterAuthorClient` keeps routing and request traces together; the
adapter converts original author records into repository results once. No extra
client is constructed during author execution. Original
post-loop reporting is outside the binding: FlipAttack's standalone reporting
uses `if eval` rather than `args.eval`; EquaCode's separate Excel keyword-score
export is not called by its original main execution either.

## Verification baselines

Older SDK environments are used only to run independent original-source
comparisons. Create separate virtual environments and install each method's
`verification/<attack>/environment.lock.txt`. Those locks describe the tested
SDK baselines, not a reconstruction of every historical dependency. FlipAttack's
HTTPX 0.27.2 pin is a compatibility addition for its older SDKs.

Set `FLIP_ATTACK_AUTHOR_PYTHON` and `EQUACODE_AUTHOR_PYTHON` to the respective
baseline Python executables. Tests run the integrated path with the shared
runtime; `AUTHOR_SHARED_PYTHON` optionally selects that test interpreter.

```bash
python -m pytest -q tests/test_author_execution.py \
  tests/test_attacks/test_flip_attack.py tests/test_attacks/test_equacode.py
python verification/verify_source.py
```

Tests use benign scripted replies and in-memory HTTP transports. They compare
author prompts, request controls, retries, sleeps, evaluation, and records,
accounting explicitly for routing and Claude serialization changes. They do not
establish live model equivalence or benchmark results. See each
`verification/<attack>/SOURCE.json` and the [verification guide](../verification/verification.md).
