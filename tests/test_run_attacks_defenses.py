from __future__ import annotations

from omegaconf import DictConfig, OmegaConf
import pytest

import run_attacks


def test_collect_configs_includes_defense(monkeypatch):
    class _DummyDataset:
        config_idx = [0]

        def __len__(self):
            return 1

    class _DummyPromptDataset:
        @staticmethod
        def from_name(_name):
            return lambda _cfg: _DummyDataset()

    monkeypatch.setattr(run_attacks, "PromptDataset", _DummyPromptDataset)
    monkeypatch.setattr(run_attacks, "filter_config", lambda rc, _dset_len, overwrite=False: rc)

    cfg = OmegaConf.create(
        {
            "models": {"m": {"id": "m"}},
            "datasets": {"d": {"idx": [0]}},
            "attacks": {"actor": {"name": "actor"}},
            "defenses": {
                "none": {"name": "none"},
                "polyguard": {"name": "polyguard"},
            },
            "model": "m",
            "dataset": "d",
            "attack": "actor",
            "defense": "polyguard",
            "overwrite": True,
        }
    )

    run_configs = run_attacks.collect_configs(cfg)
    assert len(run_configs) == 1
    assert run_configs[0].defense == "polyguard"
    assert isinstance(run_configs[0].attack_params, DictConfig)
    assert isinstance(run_configs[0].defense_params, DictConfig)
    assert "defense" not in run_configs[0].attack_params
    assert run_configs[0].defense_params["name"] == "polyguard"


def test_collect_configs_rejects_unsupported_attack_before_loading_dataset(monkeypatch):
    def _unexpected_dataset_lookup(_name):
        pytest.fail("Compatibility validation must happen before loading the dataset")

    monkeypatch.setattr(run_attacks.PromptDataset, "from_name", _unexpected_dataset_lookup)
    cfg = OmegaConf.create(
        {
            "models": {"m": {"id": "m"}},
            "datasets": {"d": {"idx": [0]}},
            "attacks": {"gcg": {"name": "gcg"}},
            "defenses": {"polyguard": {"name": "polyguard"}},
            "model": "m",
            "dataset": "d",
            "attack": "gcg",
            "defense": "polyguard",
            "overwrite": True,
        }
    )

    with pytest.raises(ValueError, match="incompatible with runtime defenses"):
        run_attacks.collect_configs(cfg)


def test_runner_builds_and_passes_target_system(monkeypatch):
    from adversariallm.io_utils.config import RunConfig

    sentinel_target = object()
    calls = {}
    events = []

    class _DummyAttack:
        def __init__(self, attack_params):
            events.append("attack")
            calls["attack_params"] = attack_params

        def run(self, target, dataset):
            calls["run_args"] = (target, dataset)
            return "results"

    monkeypatch.setattr(run_attacks, "load_model_and_tokenizer", lambda _params: ("model", "tokenizer"))
    monkeypatch.setattr(run_attacks, "PromptDataset", type("PD", (), {"from_name": staticmethod(lambda _n: (lambda _p: []))}))
    def _build_target_system(cfg, *, model, tokenizer):
        events.append("target")
        return sentinel_target

    monkeypatch.setattr(run_attacks, "build_target_system", _build_target_system)
    monkeypatch.setattr(run_attacks.Attack, "from_name", classmethod(lambda _cls, _name: _DummyAttack))
    monkeypatch.setattr(run_attacks, "log_attack", lambda *_args, **_kwargs: None)

    rc = RunConfig(
        model="m",
        dataset="d",
        attack="direct",
        defense=None,
        model_params={},
        dataset_params={},
        attack_params={"name": "direct"},
        defense_params=None,
    )
    cfg = OmegaConf.create({"save_dir": "/tmp", "embed_dir": "/tmp"})

    run_attacks.run_attacks([rc], cfg, "2026-04-06/00-00-00")

    assert events == ["attack", "target"]
    assert calls["run_args"] == (sentinel_target, [])


def test_collect_configs_resolves_interpolations_with_defense(monkeypatch):
    class _DummyDataset:
        config_idx = [0]

        def __len__(self):
            return 1

    class _DummyPromptDataset:
        @staticmethod
        def from_name(_name):
            return lambda _cfg: _DummyDataset()

    def _filter_and_resolve(rc, _dset_len, overwrite=False):
        del overwrite
        OmegaConf.resolve(rc.attack_params)
        return rc

    monkeypatch.setattr(run_attacks, "PromptDataset", _DummyPromptDataset)
    monkeypatch.setattr(run_attacks, "filter_config", _filter_and_resolve)

    cfg = OmegaConf.create(
        {
            "models": {"m": {"id": "m"}},
            "datasets": {"d": {"idx": [0]}},
            "attacks": {
                "_default": {"generation_config": {"max_new_tokens": 42}},
                "inpainting": {"generation_config": "${attacks._default.generation_config}"},
            },
            "defenses": {"polyguard": {"name": "polyguard"}, "none": {"name": "none"}},
            "model": "m",
            "dataset": "d",
            "attack": "inpainting",
            "defense": "polyguard",
            "overwrite": True,
        }
    )

    run_configs = run_attacks.collect_configs(cfg)
    assert len(run_configs) == 1
    assert run_configs[0].attack_params["generation_config"]["max_new_tokens"] == 42


def test_runner_hosted_target_does_not_load_local_model(monkeypatch):
    """Check hosted runs pass the API target through the normal attack interface."""

    from adversariallm.io_utils.config import RunConfig

    target = object()
    calls = []

    class DummyAttack:
        def __init__(self, config):
            pass

        def run(self, actual_target, dataset):
            assert actual_target is target and dataset == []
            calls.append("attack")
            return "results"

    monkeypatch.setattr(run_attacks, "APITarget", lambda cfg: target)
    monkeypatch.setattr(run_attacks, "load_model_and_tokenizer",
                        lambda cfg: pytest.fail("Hosted target loaded a local model"))
    monkeypatch.setattr(run_attacks.PromptDataset, "from_name", lambda name: (lambda cfg: []))
    monkeypatch.setattr(run_attacks.Attack, "from_name", classmethod(lambda cls, name: DummyAttack))
    monkeypatch.setattr(run_attacks, "log_attack", lambda *args: calls.append("saved"))
    rc = RunConfig(
        model="hosted", dataset="d", attack="direct", defense=None,
        model_params={"backend": "openrouter", "id": "test/model", "provider": "test"},
        dataset_params={}, attack_params={}, defense_params=None,
    )
    run_attacks.run_attacks([rc], OmegaConf.create({}), "test")
    assert calls == ["attack", "saved"]


@pytest.mark.parametrize("attack,backend,provider,defense,error", [
    ("gcg", "openrouter", "test", "none", "not been adapted"),
    ("direct", "openrouter", None, "none", "provider slug"),
    ("direct", "openrouter", "test", "polyguard", "Hosted target defenses"),
    ("direct", "unknown", "test", "none", "Unknown model backend"),
])
def test_hosted_validation_precedes_dataset_and_storage(monkeypatch, attack, backend, provider, defense, error):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid hosted configuration reached dataset or storage access")

    monkeypatch.setattr(run_attacks.PromptDataset, "from_name", unexpected)
    monkeypatch.setattr(run_attacks, "filter_config", unexpected)
    cfg = OmegaConf.create({
        "models": {"hosted": {"backend": backend, "id": "test/model", "provider": provider}},
        "model": "hosted", "datasets": {"d": {}}, "dataset": "d",
        "attacks": {attack: {"name": attack}}, "attack": attack,
        "defenses": {defense: {"name": defense}}, "defense": defense,
    })
    with pytest.raises(ValueError, match=error):
        run_attacks.collect_configs(cfg)


def test_runner_switches_between_local_and_hosted_targets(monkeypatch):
    from adversariallm.io_utils.config import RunConfig

    targets = []
    model_loads = []
    logged = []
    hosted = object()

    class DummyAttack:
        def __init__(self, config):
            pass

        def run(self, target, dataset):
            targets.append(target)
            return "results"

    def load_model(config):
        model_loads.append(config["id"])
        return config["id"], "tokenizer"

    monkeypatch.setattr(run_attacks, "APITarget", lambda cfg: hosted)
    monkeypatch.setattr(run_attacks, "load_model_and_tokenizer", load_model)
    monkeypatch.setattr(run_attacks, "build_target_system", lambda cfg, **kwargs: kwargs["model"])
    monkeypatch.setattr(run_attacks.PromptDataset, "from_name", lambda name: (lambda cfg: []))
    monkeypatch.setattr(run_attacks.Attack, "from_name", classmethod(lambda cls, name: DummyAttack))
    monkeypatch.setattr(run_attacks, "log_attack", lambda *args: logged.append(args))
    runs = [RunConfig(
        model=name, dataset="d", attack="direct", defense=None,
        model_params={"backend": backend, "id": name, "provider": "test"},
        dataset_params={}, attack_params={}, defense_params=None,
    ) for name, backend in [("local-a", "local"), ("hosted", "openrouter"), ("local-b", "local")]]
    run_attacks.run_attacks(runs, OmegaConf.create({}), "test")
    assert model_loads == ["local-a", "local-b"]
    assert targets == ["local-a", hosted, "local-b"]
    assert len(logged) == 3
