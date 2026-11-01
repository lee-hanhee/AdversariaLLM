"""Repository entry point for the unchanged author TAP search."""

from dataclasses import dataclass, field

from .author_adapter import AuthorAttack


@dataclass
class TAPConfig:
    # Keep repository identification without adding local RNG seeding.
    name: str = 'tap'
    type: str = 'discrete'
    version: str = '0.1.0'
    seed: None = None

    # Require a supported attacker source profile; retain the original evaluator default.
    attack_model: str | None = None
    evaluator_model: str = 'gpt-3.5-turbo'

    # Preserve the main script's generation controls and malformed-output attempt limit.
    attack_max_n_tokens: int = 500
    target_max_n_tokens: int = 150
    evaluator_max_n_tokens: int = 10
    evaluator_temperature: float = 0
    max_n_attack_attempts: int = 5

    # Use author CLI defaults rather than the notebook's larger branching demonstration.
    branching_factor: int = 1
    width: int = 10
    depth: int = 10
    n_streams: int = 1
    keep_last_n: int = 3

    # Require distinct explicit routes even when source labels or hosted model IDs coincide.
    attacker: dict = field(default_factory=lambda: {'id': None, 'provider': None})
    judge: dict = field(default_factory=lambda: {'id': None, 'provider': None})


class TAPAttack(AuthorAttack):
    # Select original search, client, and evaluator definitions through the shared author interface.
    AUTHOR_METHOD = 'tap'
