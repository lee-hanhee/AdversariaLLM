"""Repository binding for the original FlipAttack driver, clients, and evaluator."""

from dataclasses import dataclass, field

from ..defenses.api_target import APITarget
from .author_adapter import normalize_config, run_author
from .attack import Attack


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
    judge: dict = field(default_factory=lambda: {'id': None, 'provider': None})


class FlipAttack(Attack):
    AUTHOR_METHOD = 'flip_attack'

    def __init__(self, config):
        # Avoid the repository base initializer's seeding; the author does not seed.
        self.config = normalize_config(config)

    def run(self, target, dataset):
        # The author driver owns prompt construction, API calls, and evaluation.
        if not isinstance(target, APITarget):
            raise ValueError('FlipAttack requires the OpenRouter author-driver target.')
        return run_author(self.AUTHOR_METHOD, target, self.config, dataset)
