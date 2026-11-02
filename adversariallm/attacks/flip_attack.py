"""Repository binding for the original FlipAttack driver, clients, and evaluator."""

from dataclasses import dataclass, field

from .author_adapter import AuthorAttack


@dataclass
class FlipAttackConfig:
    # Keep repository identification, but do not introduce an RNG seed.
    name: str = 'flip_attack'
    type: str = 'discrete'
    version: str = '0.2.0'
    seed: None = None

    # Preserve the original CLI and LLM constructor defaults.
    flip_mode: str = 'FCS'
    cot: bool = False
    lang_gpt: bool = False
    few_shot: bool = False
    temperature: float = 0
    max_token: int = -1
    retry_time: int = 1000
    failed_sleep_time: int = 1
    round_sleep_time: int = 1

    # Match the optional --eval switch and the original judge-model default.
    evaluate: bool = False
    judge_model: str = 'gpt-4-0613'

    # Require an explicit OpenRouter model and provider for the author judge.
    judge: dict = field(default_factory=lambda: {'id': None, 'provider': None})


class FlipAttack(AuthorAttack):
    # Select the driver handled by the shared author entry point.
    AUTHOR_METHOD = 'flip_attack'
