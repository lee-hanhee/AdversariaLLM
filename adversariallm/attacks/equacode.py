"""Repository binding for the original EquaCode driver and mandatory judge flow."""

from dataclasses import dataclass, field

from ..defenses.api_target import APITarget
from .author_adapter import normalize_config, run_author
from .attack import Attack


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


class EquaCodeAttack(Attack):
    AUTHOR_METHOD = 'equacode'

    def __init__(self, config):
        # Keep the original unseeded behavior rather than invoking Attack.__init__.
        self.config = normalize_config(config)

    def run(self, target, dataset):
        # The unchanged driver owns target and judge control flow; transport uses OpenRouter.
        if not isinstance(target, APITarget):
            raise ValueError('EquaCode requires the OpenRouter author-driver target.')
        return run_author(self.AUTHOR_METHOD, target, self.config, dataset)
