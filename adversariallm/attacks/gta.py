"""Repository entry point for the pinned author Game-Theory Attack."""

from dataclasses import dataclass, field

from .author_adapter import AuthorAttack


@dataclass
class GTAConfig:
    # Identify the repository attack without adding RNG seeding.
    name: str = 'gta'
    type: str = 'discrete'
    version: str = '0.1.0'
    seed: None = None

    # Retain the five-round default from the published run script.
    rounds: int = 5

    # Published source profiles are fixed; actual OpenRouter routes must be supplied.
    attacker: dict = field(default_factory=lambda: {'id': None, 'provider': None})
    judge: dict = field(default_factory=lambda: {'id': None, 'provider': None})


class GTAAttack(AuthorAttack):
    # Select the driver handled by the shared author entry point.
    AUTHOR_METHOD = 'gta'
