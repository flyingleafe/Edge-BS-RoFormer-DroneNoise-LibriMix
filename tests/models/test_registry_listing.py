"""Tests for the unified model-type listing (``models.registry.model_types``).

Cross-checks the listing against the ``conf/model/*.yaml`` tree: every
``model_type`` a config names must be a listed legacy type, and every
``_target_`` a config names must resolve to an importable callable.
CPU-fast — nothing here instantiates a model.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from models.registry import (
    DIRECT_FACTORY_TYPES,
    LEGACY_MODEL_BUILDERS,
    NOISE_GEN_MODEL_REGISTRY,
    RPS_MODEL_REGISTRY,
    model_types,
)

CONF_MODEL = Path(__file__).resolve().parents[2] / "conf" / "model"


def _conf_model_files() -> list[Path]:
    files = sorted(CONF_MODEL.glob("*.yaml"))
    assert files, f"no model configs found under {CONF_MODEL}"
    return files


def test_model_types_merges_every_kind():
    listing = model_types()
    # One known key per kind, with the expected descriptor.
    assert listing["simple_conv_v2"] == {
        "kind": "rps",
        "ref": "models.registry.RPS_MODEL_REGISTRY",
    }
    assert listing["dcunet"] == {
        "kind": "legacy",
        "ref": "models.registry.LEGACY_MODEL_BUILDERS",
    }
    assert listing["positional_harmonic_gen"] == {
        "kind": "noise_gen",
        "ref": "models.registry.NOISE_GEN_MODEL_REGISTRY",
    }
    assert listing["tfgridnet"] == {
        "kind": "factory",
        "ref": "models.tfgridnet.build_tfgridnet",
    }
    # Every registry key is present.
    for keys in (
        RPS_MODEL_REGISTRY,
        LEGACY_MODEL_BUILDERS,
        NOISE_GEN_MODEL_REGISTRY,
        DIRECT_FACTORY_TYPES,
    ):
        assert set(keys) <= set(listing)


def test_registry_names_do_not_collide():
    total = (
        len(RPS_MODEL_REGISTRY)
        + len(LEGACY_MODEL_BUILDERS)
        + len(NOISE_GEN_MODEL_REGISTRY)
        + len(DIRECT_FACTORY_TYPES)
    )
    assert len(model_types()) == total


def test_every_conf_model_type_is_listed():
    listing = model_types()
    found: set[str] = set()
    for f in _conf_model_files():
        cfg = OmegaConf.load(f)
        assert isinstance(cfg, DictConfig)
        for v in (cfg.get("model_type"), OmegaConf.select(cfg, "params.model_type")):
            if v:
                found.add(str(v))
    assert found, "no model_type strings found in conf/model — glob or schema drift?"
    for name in sorted(found):
        assert name in listing, f"conf/model names model_type {name!r} but model_types() lacks it"
        assert listing[name]["kind"] == "legacy"


def test_every_conf_target_is_importable():
    targets: set[str] = set()
    for f in _conf_model_files():
        cfg = OmegaConf.load(f)
        assert isinstance(cfg, DictConfig)
        target = cfg.get("_target_")
        if target:
            targets.add(str(target))
    assert targets, "no _target_ strings found in conf/model — glob or schema drift?"
    for target in sorted(targets):
        mod_path, _, attr = target.rpartition(".")
        assert mod_path, f"_target_ {target!r} has no module part"
        mod = importlib.import_module(mod_path)
        fn = getattr(mod, attr)
        assert callable(fn), f"_target_ {target!r} resolved to a non-callable"


def test_direct_factories_match_their_refs():
    for name, ref in DIRECT_FACTORY_TYPES.items():
        mod_path, _, attr = ref.rpartition(".")
        fn = getattr(importlib.import_module(mod_path), attr)
        assert callable(fn), f"factory {name!r} ref {ref!r} is not callable"
