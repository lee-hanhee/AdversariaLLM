"""Repository entry point for the preserved Mousetrap notebook protocol."""

from dataclasses import dataclass, field

from .author_adapter import AuthorAttack


@dataclass
class MousetrapConfig:
    # The notebook uses one unseeded random stream and a fixed three-by-three protocol.
    name: str = 'mousetrap'
    type: str = 'discrete'
    version: str = '0.1.0'
    seed: None = None

    # Require an explicit OpenRouter route for the original author judge.
    judge: dict = field(default_factory=lambda: {'id': None, 'provider': None})


class MousetrapAttack(AuthorAttack):
    # Select the driver handled by the shared author entry point.
    AUTHOR_METHOD = 'mousetrap'
