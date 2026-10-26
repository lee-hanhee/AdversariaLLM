from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..lm_utils.text_generation import validate_generator_config
from .base import TargetSystem, UndefendedTarget
from .polyguard import PolyGuardDefense


_DEFENSE_REGISTRY: dict[str, type[TargetSystem]] = {
    UndefendedTarget.NAME: UndefendedTarget,
    PolyGuardDefense.NAME: PolyGuardDefense,
}

DEFENSE_COMPATIBLE_ATTACKS = frozenset(
    {
        "actor",
        "ample_gcg",
        "bon",
        "crescendo",
        "direct",
        "inpainting",
        "jailbreak_r1",
        "pair",
    }
)


def _as_dict(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return cfg


def build_target_system(
    defense_cfg: DictConfig | dict[str, Any] | None,
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    default_generate_kwargs: dict[str, Any] | None = None,
) -> TargetSystem:
    if defense_cfg is None:
        defense_cfg = {"name": "none"}
    cfg = _as_dict(defense_cfg)
    defense_name = cfg.get("name", "none")
    if defense_name is None:
        defense_name = "none"
    cfg = {**cfg, "name": defense_name}

    defense_cls = _DEFENSE_REGISTRY.get(defense_name)
    if defense_cls is None:
        raise ValueError(f"Unknown defense: {defense_name}")
    return defense_cls.from_config(
        cfg,
        model=model,
        tokenizer=tokenizer,
        default_generate_kwargs=default_generate_kwargs,
    )


def validate_defense_compatibility(
    attack_name: str,
    defense_cfg: DictConfig | dict[str, Any] | None,
) -> None:
    if defense_cfg is None:
        return
    cfg = _as_dict(defense_cfg)
    defense_name = cfg.get("name", "none")
    if defense_name in {None, "none"}:
        return
    if attack_name not in DEFENSE_COMPATIBLE_ATTACKS:
        raise ValueError(
            f"Attack '{attack_name}' is incompatible with runtime defenses. "
            "Switch to an attack that supports runtime defenses."
        )


def validate_hosted_target(attack_cls, model_config, defense_config, attack_config=None):
    """Reject unsupported hosted combinations before dataset or model loading."""
    method = getattr(attack_cls, "AUTHOR_METHOD", None)
    if method is not None:
        from ..attacks.author_execution import validate_author_config

        validate_author_config(method, model_config, attack_config or {})
        if defense_config is not None and defense_config.get("name", "none") not in (None, "none"):
            raise ValueError("Original author execution does not support runtime defenses.")
        return
    validate_generator_config(model_config)
    if model_config.get("backend", "local") == "local":
        return
    if not attack_cls.SUPPORTS_HOSTED_TARGET:
        raise ValueError(f"Attack '{attack_cls.__name__}' has not been adapted for hosted targets.")
    if defense_config is not None and defense_config.get("name", "none") not in (None, "none"):
        raise ValueError("Hosted target defenses are not implemented yet.")
