"""Repository entry point for the pinned author Game-Theory Attack."""

from dataclasses import dataclass, field

from .author_adapter import AuthorAttack


@dataclass
class GTAConfig:
    name: str = 'gta'
    type: str = 'discrete'
    version: str = '0.1.0'
    seed: None = None
    rounds: int = 5

    # Published source profiles are fixed; actual OpenRouter routes must be supplied.
    attacker: dict = field(default_factory=lambda: {'id': None, 'provider': None})
    judge: dict = field(default_factory=lambda: {'id': None, 'provider': None})


class GTAAttack(AuthorAttack):
    AUTHOR_METHOD = 'gta'
