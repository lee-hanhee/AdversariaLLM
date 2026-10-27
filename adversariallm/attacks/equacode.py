"""Repository binding for the original EquaCode driver and mandatory judge flow."""

from dataclasses import dataclass, field

from .author_adapter import AuthorAttack


@dataclass
class EquaCodeConfig:
    # Use repository identification without adding author-absent RNG seeding.
    name: str = 'equacode'
    type: str = 'discrete'
    version: str = '0.2.0'
    seed: None = None

    # Retain the main-script template selection and literal judge ID.
    variant: str = 'equacode'
    judge_model: str = 'GPT-4'
    judge: dict = field(default_factory=lambda: {'id': None, 'provider': None})


class EquaCodeAttack(AuthorAttack):
    AUTHOR_METHOD = 'equacode'
