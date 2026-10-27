# Attack overview

Methods in the incremental PR plan, including the Direct baseline and held
methods. Planned and held entries link to author repositories; their adapters
are not present on this branch yet.

| Attack | What it does | How it works | Status | Code / source |
| --- | --- | --- | --- | --- |
| Direct | Provides an unmodified-prompt baseline. | Sends the original user request to the target and records its response. | PR 01 | [Adapter](../adversariallm/attacks/direct.py) |
| FlipAttack | Disguises a request by flipping text. | Reverses words or characters, then gives the target instructions to reconstruct and answer it. Optional guidance adds step-by-step instructions and examples. | PR 02 | [Adapter](../adversariallm/attacks/flip_attack.py), [author builder](../adversariallm/attacks/utils/flip_attack/flip_attack.py), [verification](../verification/flip_attack/SOURCE.json) |
| EquaCode | Presents a request as equations or code. | Inserts the request into author templates that ask the target to solve equations, complete code, or combine both. | PR 03 | [Adapter](../adversariallm/attacks/equacode.py), [author templates](../adversariallm/attacks/utils/equacode/equacode_prompts.py), [verification](../verification/equacode/SOURCE.json) |
| GTA | Uses a game-theoretic interrogation scenario. | A Police attacker alternates between Thomas and Marcos, each with separate history; an author Judge feeds ratings into later questions. | PR 04, this branch | [Adapter](../adversariallm/attacks/gta.py), [author driver](../adversariallm/attacks/utils/gta/gta_driver.py), [verification](../verification/gta/SOURCE.json) |
| Mousetrap | Hides a request behind layered text transformations. | Applies reversible transformations and asks the target to decode them before answering; a judge evaluates attempts. | Planned: PR 05 | [Author source](https://github.com/EVIGBYEN/Mousetrap/tree/3906547a9d18fe0e54d27bb541676162cbdbe31c) |
| QueryAttack | Rewrites a request into code or query-like text. | A helper extracts and translates the request into a selected language/template, then the target receives the assembled prompt. | Planned: PR 06 | [Author source](https://github.com/horizonsinzqs/QueryAttack/tree/93a19806cf0e21a0b28b6fee18a895d57f170bb6) |
| TAP | Searches a tree of candidate attack prompts. | An attacker model proposes branches; an evaluator prunes off-topic or weaker candidates and scores target responses. | Held for fidelity audit | [Author source](https://github.com/RICommunity/TAP) |
| CoT Forgery | Adds fabricated reasoning or policy text to a request. | A helper generates the forgery, which is appended to the target prompt to encourage answering. | Held for fidelity audit | [Author source](https://github.com/role-confusion/prompt-injection-as-role-confusion/tree/ec333c40fd43fe991e1ebf66765051b6d7e35784) |

Use [author attack execution through OpenRouter](author_execution.md) for FlipAttack, EquaCode, and GTA,
[OpenRouter targets](openrouter.md) for the Direct baseline, and
[Reviewing author-code fidelity](../verification/verification.md)
for verification. Source identity and offline comparisons do not establish
paper-result reproduction. The linked verification manifests record source
pins, integration departures, and test commands/results.
