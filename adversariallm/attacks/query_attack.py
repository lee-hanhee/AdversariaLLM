"""Repository entry point for the original QueryAttack translation and target stages."""

from dataclasses import dataclass, field

from .author_adapter import AuthorAttack


@dataclass
class QueryAttackConfig:
    # Keep repository identification separate from the author's API request seed.
    name: str = 'query_attack'
    type: str = 'discrete'
    version: str = '0.1.0'
    seed: None = None

    # Preserve the independent defaults of translator.py and test.py.
    translation_language: str = 'ALL'
    target_language: str = 'SQL'
    trans_verify: bool = False

    # Retain the original target CLI controls, including its temperature spelling.
    temperate: float = 0
    top_p: float = 1
    max_new_tokens: int = 512
    request_seed: int = 0

    # Require one explicit route for splitting, extraction, and translation verification.
    translator: dict = field(default_factory=lambda: {'id': None, 'provider': None})


class QueryAttack(AuthorAttack):
    # Select the source driver through the common hosted author interface.
    AUTHOR_METHOD = 'query_attack'
