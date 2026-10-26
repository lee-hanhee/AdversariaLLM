import copy
import time
from dataclasses import dataclass, field

from ..defenses.api_target import APITarget
from .attack import Attack, AttackResult, AttackStepResult, SingleAttackRunResult
from .data.flip_attack import FlipAttack as AuthorFlipAttack


@dataclass
class FlipAttackConfig:

    # Repository identity and seed settings used by the runner.
    name: str = "flip_attack"
    type: str = "discrete"
    version: str = "0.1.0"
    seed: int = 0

    # Author prompt options; each guidance switch is independent.
    flip_mode: str = "FCS"
    cot: bool = False
    lang_gpt: bool = False
    few_shot: bool = False

    # Match author defaults: one response, temperature zero, no output cap.
    generation_config: dict = field(default_factory=lambda: {
        "temperature": 0.0, "max_new_tokens": None, "num_return_sequences": 1,
    })


class FlipAttack(Attack):
    """Construct one author-style prompt and query the configured target once per example."""

    SUPPORTS_HOSTED_TARGET = True

    @staticmethod
    def validate_config(config):
        """Reject unsupported modes and response counts before requests."""

        # Check prompt options before constructing the author builder.
        if config.flip_mode not in {"FWO", "FCW", "FCS", "FMM"}:
            raise ValueError("FlipAttack flip_mode must be FWO, FCW, FCS, or FMM.")
        for flag in ("cot", "lang_gpt", "few_shot"):
            if not isinstance(getattr(config, flag), bool):
                raise ValueError(f"FlipAttack {flag} must be a boolean.")

        # Require one response and allow an optional positive output-token limit.
        generation = config.generation_config
        if generation.get("num_return_sequences", 1) != 1:
            raise ValueError("FlipAttack uses one response per example.")
        limit = generation.get("max_new_tokens")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            raise ValueError("FlipAttack max_new_tokens must be null or a positive integer.")

    def __init__(self, config):

        # Validate settings, then apply the repository's seed initialization.
        self.validate_config(config)
        super().__init__(config)

    def run(self, target, dataset):
        """Preserve author messages and save responses for separate evaluation."""

        # This slice supports hosted targets only; local chat is added separately.
        if not isinstance(target, APITarget):
            raise ValueError("FlipAttack requires a configured OpenRouter target.")

        result = AttackResult()
        for original in dataset:

            # Read the user request while retaining the original reference answer.
            if len(original) != 2 or [turn["role"] for turn in original] != ["user", "assistant"]:
                raise ValueError("FlipAttack requires a user request and an assistant target response.")

            started = time.perf_counter()

            # The author runner creates a fresh builder for every example.
            builder = AuthorFlipAttack(
                flip_mode=self.config.flip_mode, cot=self.config.cot,
                lang_gpt=self.config.lang_gpt, few_shot=self.config.few_shot,
                victim_llm=target.config["id"],
            )
            _, messages = builder.generate(original[0]["content"])

            # Send the author's messages unchanged and require one usable answer.
            generated = target.generate([messages], **self.config.generation_config)
            if (len(generated.gen) != 1 or len(generated.gen[0]) != 1
                    or not isinstance(generated.gen[0][0], str) or not generated.gen[0][0]):
                raise ValueError("FlipAttack requires one nonempty target response.")

            # Store the response and API metadata; scoring happens separately.
            elapsed = time.perf_counter() - started
            step = AttackStepResult(
                step=0, model_input=messages, model_completions=generated.gen[0],
                api_metadata=generated.api_metadata[0] if generated.api_metadata is not None else None,
                time_taken=elapsed,
            )
            result.runs.append(SingleAttackRunResult(
                original_prompt=copy.deepcopy(original), steps=[step], total_time=elapsed,
            ))

        return result
