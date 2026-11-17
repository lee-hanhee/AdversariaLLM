# Running author attacks through OpenRouter

FlipAttack, EquaCode, GTA, Mousetrap, QueryAttack, and TAP execute pinned author
attack logic with an OpenRouter transport binding. Target, helper, and judge
calls use `openai==1.109.1` as a client for `https://openrouter.ai/api/v1`. Only `OPENROUTER_API_KEY` is needed; no direct
OpenAI, Anthropic, or DeepInfra credentials are used.

## Configuration

All hosted attacks share the `openrouter` entry in `conf/models/models.yaml`.
Select `model=openrouter` and supply `models.openrouter.id` and
`models.openrouter.provider`. Select the algorithm separately with `attack=...`.
Legacy source profiles belong to `attacks.<attack>.source_profile`; they select
original model handling independently of the hosted model ID. Mousetrap and TAP
require an explicit source profile; the other four attacks retain their source defaults.
`model=null` selects the entire catalog, including the unconfigured hosted entry;
use an explicit model selection for these runs.

Use the normal repository dependency setup and MongoDB configuration. The
repository lock already selects OpenAI 1.109.1; no extra runtime environment or
requirements file is needed. Each required target, helper, or judge route
needs an explicit OpenRouter model ID and one provider slug. There are no
routing defaults, including for FlipAttack's disabled-by-default judge.

Preview a configuration without making requests:

```bash
python run_attacks.py --cfg job hydra/launcher=basic \
  model=openrouter attack=equacode \
  models.openrouter.id=TARGET_MODEL_ID \
  models.openrouter.provider=TARGET_PROVIDER \
  attacks.equacode.judge.id=JUDGE_MODEL_ID \
  attacks.equacode.judge.provider=JUDGE_PROVIDER \
  classifiers=null
```

For FlipAttack, select `attack=flip_attack` and replace the `attacks.equacode`
overrides with `attacks.flip_attack`. Keep `model=openrouter`. Set
`attacks.flip_attack.evaluate=true` to enable its author judge. Remove `--cfg job`
to run after selecting the dataset and indices normally. `classifiers=null`
disables the runner's separate post-run classifiers; it does not disable
EquaCode's mandatory author judge or FlipAttack's enabled author judge.

For GTA, select `model=openrouter attack=gta` and supply
`models.openrouter.id`, `models.openrouter.provider`, `attacks.gta.attacker.id`,
`attacks.gta.attacker.provider`, `attacks.gta.judge.id`, and
`attacks.gta.judge.provider`. Every route is required. `attacks.gta.rounds`
defaults to the author's five rounds. Thomas and Marcos share the target route
but keep separate histories. In-attack judging always runs; keep
`classifiers=null` to disable separate repository post-run classifiers.

For Mousetrap, select `model=openrouter attack=mousetrap`.
Supply `attacks.mousetrap.source_profile` as `o1-2024-12-17` or
`claude-3-7-sonnet-20250219`, plus explicit target and judge model IDs/providers.
For example, preview the configuration without inference:

```bash
python run_attacks.py --cfg job hydra/launcher=basic \
  model=openrouter attack=mousetrap \
  attacks.mousetrap.source_profile=o1-2024-12-17 \
  models.openrouter.id=TARGET_MODEL_ID \
  models.openrouter.provider=TARGET_PROVIDER \
  attacks.mousetrap.judge.id=JUDGE_MODEL_ID \
  attacks.mousetrap.judge.provider=JUDGE_PROVIDER classifiers=null
```

Mousetrap's Google source wrapper is preserved, but the runtime rejects its
`gemini-2.5-pro-exp-03-25` profile until OpenRouter's handling of the four original
safety settings can be verified. No settings are silently dropped. The original
judge label is recorded in `verification/mousetrap/SOURCE.json` and the preserved
author class; configure the actual route using `attacks.mousetrap.judge.id` and
`attacks.mousetrap.judge.provider`.

For QueryAttack, select `model=openrouter attack=query_attack`.
Supply `models.openrouter.id/provider` and
`attacks.query_attack.translator.id/provider`. The helper route handles sentence
splitting, extraction, and optional translation verification; no post-run judge
route is used. Preview without inference:

```bash
python run_attacks.py --cfg job hydra/launcher=basic \
  model=openrouter attack=query_attack \
  models.openrouter.id=TARGET_MODEL_ID \
  models.openrouter.provider=TARGET_PROVIDER \
  attacks.query_attack.translator.id=HELPER_MODEL_ID \
  attacks.query_attack.translator.provider=HELPER_PROVIDER classifiers=null
```

Install the declared `nltk~=3.9.1` dependency and provision its original English
resources before running: `python -m nltk.downloader punkt_tab averaged_perceptron_tagger_eng`.
Use `NLTK_DATA` if they are installed outside NLTK's default search paths. The
attack checks for resources before requests and never downloads them at runtime.

QueryAttack retains the two scripts' independent defaults:
`translation_language=ALL`, `target_language=SQL`, `trans_verify=false`,
`temperate=0`, `top_p=1`, `max_new_tokens=512`, and `request_seed=0`. The last
setting is sent as the original API `seed`; the repository `seed` stays `null`
and Python RNG is not seeded. Select any of the nine original target languages.
A single translation language must produce the file required by the selected
target language. The original single-language C++ path does not write a file,
so it fails; the default `ALL` translation does produce its C++ file.

Each attack configuration's `source_profile` is the source-level name used by the
unchanged author code. It is **not** sent to OpenRouter. FlipAttack uses it to
choose GPT/Claude/Llama/Mixtral handling; EquaCode uses its case-sensitive name
check when constructing the system message. GTA initially supports only the
published run-script source profiles: DeepSeek-R1 targets, GPT-4o-mini attacker,
and GPT-4o judge. Other GTA source model paths are unsupported. FlipAttack and
EquaCode use `judge_model` for their source labels, while `judge.id` determines
actual routing. Selecting a different hosted model does not make it equivalent
to the author's original model.

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
| Mousetrap | Original notebook SDK wrappers, chaos mappings, overwritten prompt assignment, parser, and three attempts at each of three chain lengths; no checker call. Claude retains `max_tokens=2048`; OpenAI target and judge controls remain omitted. |
| QueryAttack | Original helper request defaults, >13-token splitting gate, extraction retries, optional verification/reversal, all nine templates, CSV round trip, target context, generation controls, and newline escaping. |
| GTA fallback | Preserve the unsupported source `gpt-4o_u` fallback: three model-selection failures with two-second sleeps and a final `-1` rating. No invented fallback API route. |
| Transport | OpenRouter replaces original endpoints and model IDs. FlipAttack's Claude system field becomes a system chat message and its 3,584-token cap remains. Mousetrap's Claude request has no system field and retains its 2,048-token cap. Native Anthropic serialization is not reproduced. |
| SDK | Shared OpenAI 1.109.1 replaces the original SDK clients; SDK retries may differ for failures outside the tested cases. |
| Integration | Repository datasets and results replace standalone input selection and durable output storage. QueryAttack retains temporary CSV/text file stages. No RNG seed or generic generation overrides are introduced. |

The attacks run directly in the runner and share one `OpenRouterTarget` SDK
client. Each attack validates its own configuration without adding RNG seeding.
Results share one hierarchy:
`AuthorAttackResult` contains an `AuthorRunResult` per dataset example, with
`AuthorStepResult` entries for individual target turns. Both runs and steps retain
`author_result` and `author_requests`; a run keeps the complete record even when
it has no steps. Explicit attack branches select the values, and shared step
construction handles repository metadata and indexing.

Each attack in `adversariallm/attacks/<attack>.py` defines its own `run`,
configuration validation, `_execute` driver, and `_add_steps` result conversion.
Preserved source remains in `adversariallm/attacks/author_implementations/<attack>/`.
Shared SDK routing, source loading, and result bookkeeping live in
`adversariallm/attacks/utils/author_runtime.py`; there is no central attack dispatcher.
The drivers accept `(transport, model, config, questions, sleep)`, where questions
are `(goal, reference_answer)` pairs. Manifests list the attack module under
`integration.files`; preserved author correspondence remains under `files`.

Each attack checks its supported source profiles and required options. The
previous exhaustive unknown-setting allowlist has been removed. Use the declared
attack defaults when constructing configurations.

`OpenRouterAuthorClient` keeps routing and request traces together; the
attack converts original author records into repository results once. No extra
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

Mousetrap samples one class permutation for the whole dataset and keeps the
original unseeded random stream across attempts and examples. Repository result
construction restores that stream after its bookkeeping. Each length starts from
the original question; all three attempts must succeed before stopping. The
notebook's `score >= 4` rule is retained, although the [paper, §3.4](https://arxiv.org/html/2502.15806v2#S3.SS4) says `score > 4`.
Missing or non-integer bracketed scores still raise, and original Unicode mapping
errors remain unchanged. No format repair or application retry is added.

Mousetrap results retain original attempt rows and the notebook's `iter` log,
with the sampled `params` on each run. `dim` is `null` because repository message
inputs have no category field. Original JSONL writes are captured in memory and
stored through the repository logger; per-attempt file durability is not retained.
Only per-example duration is measured. Original datasets and paper-only variants
are outside this slice.

QueryAttack writes each repository message followed by a newline into the
original input-file shape. The original reader strips each physical line;
multiline messages therefore produce multiple source requests, grouped under
their originating `AuthorRunResult`. Translation for the complete batch finishes
before target generation. Output files are captured in `author_summary.files`
before temporary storage is removed; phrase rows and escaped responses remain
in each run's source-line records. Raw API text stays in request traces and step
raw completions. Timing covers helper and target execution, excluding file-stage
bookkeeping.

The original writer uses `----` separators while the reader splits on `---`,
leaving one trailing hyphen after non-final records and four after the final
record (its final newline is stripped before splitting). Separator text inside
extracted fields can create extra target requests; these calls are preserved,
and each fragment records its source-line index and batch target index. Other
source quirks remain, including the ineffective temporary-list assignment in
NLTK filtering and the original verification loops without added retry limits.
`evaluate_responses.py` and its harmfulness scoring are reference-only and are
not executed. The pinned tree does not contain the README's `test_gemini.py`;
this integration uses the published `test.py` target path.

## Verification baselines

Older SDK environments are used only to run independent original-source
comparisons. FlipAttack and EquaCode provide
`verification/<attack>/environment.lock.txt` for separate environments. Those locks describe the tested
SDK baselines, not a reconstruction of every historical dependency. FlipAttack's
HTTPX 0.27.2 pin is a compatibility addition for its older SDKs.

Set `FLIP_ATTACK_AUTHOR_PYTHON` and `EQUACODE_AUTHOR_PYTHON` to the respective
baseline Python executables. GTA has no pinned SDK version; its independent
reference execution uses the shared SDK (which satisfies the author requirement)
with native Azure/OpenAI serializers. Mousetrap publishes no SDK pins; its
reference uses shared OpenAI 1.109.1 and Anthropic 0.36.0 for native serialization
comparisons, without claiming those are historical author versions. For that
shared comparison baseline, use an isolated interpreter with
`openai==1.109.1 anthropic==0.36.0 httpx==0.27.2` and set `AUTHOR_SHARED_PYTHON`
to its Python executable. These are test-only dependencies; the runtime still
uses the existing OpenAI client without importing native Anthropic/Google SDKs.

```bash
python -m pytest -q tests/test_author_execution.py \
  tests/test_attacks/test_flip_attack.py tests/test_attacks/test_equacode.py \
  tests/test_attacks/test_gta.py tests/test_attacks/test_mousetrap.py
python verification/verify_source.py
```

Tests use benign scripted replies and in-memory HTTP transports. They compare
author prompts, request controls, retries, sleeps, evaluation, and records,
accounting explicitly for routing and Claude serialization changes. They do not
establish live model equivalence or benchmark results. See each
`verification/<attack>/SOURCE.json` and the [verification guide](../verification/verification.md).

For QueryAttack, create a test-only baseline with
`python -m pip install -r verification/query_attack/environment.lock.txt` and set
`QUERY_ATTACK_AUTHOR_PYTHON` to its interpreter. It uses OpenAI 1.54.3 and NLTK
3.9.1, satisfying the author's compatible-version requirements; HTTPX 0.27.2 is
an added compatibility pin. Set `QUERY_ATTACK_SHARED_PYTHON` to another isolated
interpreter using `openai==1.109.1 nltk==3.9.1 httpx==0.27.2`, and provision the
same NLTK data for both. These interpreters compare complete official scripts
against the binding without importing the ML stack.

```bash
python -m pytest -q tests/test_attacks/test_query_attack.py
python verification/verify_source.py query_attack --official-repo /path/to/official-clone
```

## TAP

TAP runs the original `main_TAP.py` search, GPT wrapper, parser, pruning, and
evaluators. Only the original working GPT attacker/target source profiles are
supported: `gpt-3.5-turbo`, `gpt-4`, and `gpt-4-1106-preview`. Both source profiles
must be explicitly selected; native Vicuna/Llama and other provider wrappers
are outside this hosted slice. The original `gpt-4-turbo` attacker/target loading
bug is not repaired. Its separate evaluator path does work and is supported,
along with the original `no-evaluator` option.

Install the author's exact conversation dependency separately, from the repository root:

```bash
python -m pip install --no-deps --target .author_envs/tap_fastchat fschat==0.2.32
```

The binding loads the package's unchanged, standard-library-only conversation
module and original GPT/default adapter definitions. It does not import FastChat's
model server or install its incompatible Pydantic requirements. The two supported
adapter matches are checked independently against all original adapter match
functions in verification. FastChat 0.2.32 gives `gpt-3.5-turbo` and `gpt-4` its
`chatgpt` template; `gpt-4-1106-preview` falls back to `one_shot`. This historical
behavior is retained.

Preview a configuration without inference:

```bash
python run_attacks.py --cfg job hydra/launcher=basic \
  model=openrouter attack=tap \
  attacks.tap.source_profile=gpt-4 \
  models.openrouter.id=TARGET_MODEL_ID models.openrouter.provider=TARGET_PROVIDER \
  attacks.tap.attack_model=gpt-4 \
  attacks.tap.attacker.id=ATTACKER_MODEL_ID attacks.tap.attacker.provider=ATTACKER_PROVIDER \
  attacks.tap.judge.id=EVALUATOR_MODEL_ID attacks.tap.judge.provider=EVALUATOR_PROVIDER \
  classifiers=null
```

Source labels select original branches and formatting; they are independent of
actual hosted model IDs. All three routes remain explicit, including when using
`no-evaluator`, consistent with the earlier disabled-judge configuration policy.
Each repository user request becomes the original `goal`; its assistant reference
answer becomes the original desired `target_str` response prefix. Unknown category
metadata is `null`. No local RNG seeding is added.

Defaults match the main CLI: branching factor 1, width/depth 10, one root stream,
three retained history turns, five proposal attempts, and token caps 500/150/10
for attacker/target/evaluator. The demo's branching factor 4 is not substituted.
Original nonpositive integer behavior is retained rather than adding positivity
limits. Python-generated conversation IDs and NumPy pruning RNG run continuously
across examples.

The legacy OpenAI 0.28 call interface is bound to the existing shared SDK client:
`request_timeout=20` becomes `timeout=20`, and responses are returned to the author
as mappings. A view of the same SDK transport disables modern SDK HTTP retries;
the author's 20-attempt loop, ten-second error sleeps, half-second post-error
sleeps, and `$ERROR$` marker remain. Low-level connection handling and native
vendor serialization are not claimed identical across SDKs.

The original tuple sort resolves ties by index despite shuffling first. All-off-topic
fallback can retain or duplicate rejected candidates. Exhausted malformed proposals
can raise `IndexError` in empty-frontier cleanup. Parser fallbacks, accepted
out-of-range ratings, exact-10 stopping, and GPT history overwrites all remain.
No corrective pruning, stricter parser, or extra evaluation is introduced.

W&B/parquet storage is replaced by repository results. Run records retain original
logger inputs per iteration; steps retain every target query and its original
rating, including candidates later pruned. Request traces retain all auxiliary
calls and retries. Logging statistics computed by the standalone logger are not
recreated. Search ratings are author search feedback, not separate effectiveness
verification.

For offline comparisons, set `TAP_AUTHOR_PYTHON` to an isolated environment using
`verification/tap/environment.lock.txt`, and `TAP_SHARED_PYTHON` to an interpreter
with `openai==1.109.1 httpx==0.27.2 numpy==1.26.4`. Both comparison paths use the
isolated pinned FastChat package. The baseline retains OpenAI 0.28.0 and NumPy
1.26.3, without claiming a full reconstruction of the author's GPU environment.

```bash
python -m pytest -q tests/test_attacks/test_tap.py
python verification/verify_source.py tap --official-repo /path/to/official-clone
```
