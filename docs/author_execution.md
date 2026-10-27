# Running author attacks through OpenRouter

FlipAttack, EquaCode, and GTA execute the pinned author attack logic with an OpenRouter
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

For GTA, select `models=gta_author model=gta_author attack=gta` and supply
`models.gta_author.id`, `models.gta_author.provider`, `attacks.gta.attacker.id`,
`attacks.gta.attacker.provider`, `attacks.gta.judge.id`, and
`attacks.gta.judge.provider`. Every route is required. `attacks.gta.rounds`
defaults to the author's five rounds. Thomas and Marcos share the target route
but keep separate histories. In-attack judging always runs; keep
`classifiers=null` to disable separate repository post-run classifiers.

The model configuration's `author_model` is the source-level name used by the
unchanged author code. It is **not** sent to OpenRouter. FlipAttack uses it to
choose GPT/Claude/Llama/Mixtral handling; EquaCode uses its case-sensitive name
check when constructing the system message. GTA initially supports only the
published run-script source profiles: DeepSeek-R1 targets, GPT-4o-mini attacker,
and GPT-4o judge. Other author model paths are not exposed in this PR. `judge_model` likewise retains the
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
| GTA | Original Police/Thomas/Marcos/Judge agents and `ours_attack` orchestration, including format retries, distinct histories, source temperatures, DeepSeek think removal/error handling, and no success-based early stopping. |
| GTA fallback | Preserve the unsupported source `gpt-4o_u` fallback: three model-selection failures with two-second sleeps and a final `-1` rating. No invented fallback API route. |
| Transport | OpenRouter replaces original endpoints and model IDs. Claude's separate system field becomes a system chat message; its 3,584-token cap is retained. Native Anthropic serialization is not reproduced. |
| SDK | Shared OpenAI 1.109.1 replaces the authors' pinned SDKs; SDK retries may differ for failures outside the tested cases. |
| Integration | Repository datasets and results replace CSV loading, spreadsheet output, and CLI reporting. No RNG seed or generic generation overrides are introduced. |

The adapter runs directly in the runner and reuses the existing `APITarget` SDK
client. All author entry points inherit `AuthorAttack`, which handles configuration
and hosted-target checks without adding RNG seeding. Results share one hierarchy:
`AuthorAttackResult` contains an `AuthorRunResult` per dataset example, with
`AuthorStepResult` entries for individual target turns. Both runs and steps retain
`author_result` and `author_requests`; a run keeps the complete record even when
it has no steps. Explicit attack branches select the values, and shared step
construction handles repository metadata and indexing.

Our per-attack I/O binding lives in `adversariallm/attacks/utils/<attack>/binding.py`,
beside that attack's preserved author files. `author_execution.py` owns the shared
client, source loader, validation, and explicit dispatch; `author_adapter.py`
converts outputs into repository results. All bindings expose the same
`execute(transport, model, config, questions, sleep)` interface. Manifests list
our binding under `integration.files`; preserved author correspondence remains
under `files`.

`OpenRouterAuthorClient` keeps routing and request traces together; the
adapter converts original author records into repository results once. No extra
client is constructed during author execution. Original
post-loop reporting is outside the binding: FlipAttack's standalone reporting
uses `if eval` rather than `args.eval`; EquaCode's separate Excel keyword-score
export is not called by its original main execution either. GTA's separate
`eval.py` post-run evaluator (including its local guard model), additional
scenarios, and score-only ablation are outside this hosted attack slice. The
published GTA shell command also supplies `--attack`, which its parser does not
define; we bind the original `ours_attack` function rather than reproduce that
standalone CLI defect.

GTA preserves caught DeepSeek API failures as `None` in the original conversation
and judgement records. Repository steps expose no completion for those turns,
rather than manufacturing a refusal. Its original `-1` judge rating is retained.
Zero/negative integer rounds retain the original background-only result.

## Verification baselines

Older SDK environments are used only to run independent original-source
comparisons. Create separate virtual environments and install each method's
`verification/<attack>/environment.lock.txt`. Those locks describe the tested
SDK baselines, not a reconstruction of every historical dependency. FlipAttack's
HTTPX 0.27.2 pin is a compatibility addition for its older SDKs.

Set `FLIP_ATTACK_AUTHOR_PYTHON` and `EQUACODE_AUTHOR_PYTHON` to the respective
baseline Python executables. GTA has no pinned SDK version; its independent
reference execution uses the shared SDK (which satisfies the author requirement)
with native Azure/OpenAI serializers. Tests run the integrated path with the shared
runtime; `AUTHOR_SHARED_PYTHON` optionally selects that test interpreter.

```bash
python -m pytest -q tests/test_author_execution.py \
  tests/test_attacks/test_flip_attack.py tests/test_attacks/test_equacode.py \
  tests/test_attacks/test_gta.py
python verification/verify_source.py
```

Tests use benign scripted replies and in-memory HTTP transports. They compare
author prompts, request controls, retries, sleeps, evaluation, and records,
accounting explicitly for routing and Claude serialization changes. They do not
establish live model equivalence or benchmark results. See each
`verification/<attack>/SOURCE.json` and the [verification guide](../verification/verification.md).
