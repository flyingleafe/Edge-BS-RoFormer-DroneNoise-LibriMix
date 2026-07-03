"""Composition-only check for every ``conf/experiment/*.yaml``.

See REPLICATION.md § "Composition check". Composes each experiment against
the root ``conf/config.yaml`` + the registered ``RootConfig`` structured
schema (``training.config.register_configs``) via Hydra's ``compose()`` API,
then fully resolves the result (``OmegaConf.to_container(..., resolve=True,
throw_on_missing=True)``) to surface missing-mandatory-value, wrong-type, and
unknown-key errors — the same checks ``@hydra.main`` performs when
``train.py``/``eval.py`` compose a config, minus dataset/model
instantiation (deliberately: this machine has no training data, and a real
one-batch smoke test needs ``training.validate.validate_config`` on a
data-ful machine, e.g. via ``python train.py experiment=<name>
validate_only=true``).

Usage::

    uv run python scripts/check_experiment_configs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_DIR = REPO_ROOT / "conf" / "experiment"


def _experiment_names() -> list[str]:
    return sorted(p.stem for p in EXPERIMENT_DIR.glob("*.yaml"))


def check_all() -> tuple[list[str], list[tuple[str, str]]]:
    """Compose+resolve every experiment. Returns (ok_names, [(name, error)])."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from training.config import register_configs

    register_configs()

    ok: list[str] = []
    failed: list[tuple[str, str]] = []
    names = _experiment_names()

    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()

    with initialize(version_base=None, config_path="../conf"):
        for name in names:
            try:
                cfg = compose(
                    config_name="config",
                    overrides=[f"experiment={name}", "validate_only=true"],
                )
                OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            except Exception as exc:  # collect every per-file failure, report them all at the end
                failed.append((name, f"{type(exc).__name__}: {exc}"))
            else:
                ok.append(name)

    return ok, failed


def main() -> int:
    ok, failed = check_all()
    print(f"Composed OK: {len(ok)}/{len(ok) + len(failed)}")
    for name in ok:
        print(f"  OK   {name}")
    if failed:
        print(f"\nFAILED: {len(failed)}")
        for name, err in failed:
            print(f"  FAIL {name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
