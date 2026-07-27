#!/usr/bin/env python
"""Verify an online-mix stream ACTUALLY has the properties its policy intends.

Born from the CKLA staging bug (docs/experiments/ckla.md § "THE STAGING BUG"):
``OnlineMixFrameDataset(flatten_channels=True)`` expands each generated chunk
into C=8 mono frames, so policy stage boundaries (``until:``, in *generated
chunk* units) sat at effective epoch ~80 instead of 10 and the staged
augmentations silently never fired in any <=57-epoch run — for ~3 weeks of
experiments. This tool measures, on the real stream, (1) the chunk->frame
expansion ratio, (2) where each stage boundary lands in effective epochs, and
(3) the EMPIRICAL per-key augmentation fire rates at chosen epochs, compared
against the configured probabilities. Run it against any new/edited policy
BEFORE submitting training jobs, and after any data-path refactor.

Usage::

    python scripts/check_stream.py --policy conf/online_mix/<name>.yaml \
        [--flatten] [--samples-per-epoch 5000] [--epochs 0 5 10 12] [--probes 48]

    python scripts/check_stream.py --experiment <name>   # Hydra-composed:
        # pulls policy path + flatten_channels + samples_per_validation from
        # exactly what `python train.py experiment=<name>` would train on.

Exit code is nonzero on any FAIL, so it can gate job submission.

How fire detection works (the RNG-stream subtlety): a present-but-missed aug
block still consumes one ``rng.random()`` fire-decision draw, so *removing*
the key from a control policy would shift every downstream draw and make all
samples differ (measured rate ~1.0 regardless of the true one). The control
therefore keeps the block but sets ``probability`` to 1e-9: the decision draw
is still consumed (byte-identical stream on the miss path) while the block
essentially never fires — real-vs-control outputs then differ iff the real
policy actually fired on that sample id. Caveat: a fired block whose chosen
transform is an exact no-op (e.g. ``channel_drop`` on a mono stream) is
counted as not-fired; for the multichannel RPS stream all choices are
observable.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_ROOT / ".env", override=False)

import torch  # noqa: E402

#: Policy keys that are probabilistic augmentation blocks (fire/choice schema).
AUG_KEYS = ("augmentations", "noise_augmentations", "noise_time_warp")

#: Control probability: >0 so the fire-decision draw is still consumed (keeps
#: the RNG stream aligned with the real policy's miss path), but so small the
#: block never actually fires.
NEVER_FIRE_P = 1e-9

#: Two-sided exact binomial p-value below which an observed fire count is
#: declared incompatible with the configured probability. Generous on purpose:
#: this is a smoke detector for "never fires" / "fires in the wrong stage",
#: not a calibration test.
P_VALUE_FLOOR = 1e-3


def _stage_blocks(policy: Mapping[str, Any]) -> list[Any]:
    stages = policy.get("stages")
    return list(stages) if stages else [policy]


def _key_probability(stage: Mapping[str, Any], key: str) -> float:
    spec = stage.get(key)
    if not isinstance(spec, Mapping):
        return 0.0
    p = float(spec.get("probability", 0.0))
    if key != "noise_time_warp" and not spec.get("choices"):
        return 0.0  # a block with no choices never changes anything
    return min(max(p, 0.0), 1.0)


def _label_rule(policy: Mapping[str, Any], key: str) -> str | None:
    """PASS/FAIL rule for the label-diff count of ``key``, or None (report only).

    ``augmentations`` is post-mix on the mixture only — labels must NEVER
    change ("zero"). A ``noise_augmentations`` block whose every choice is
    ``freq_scale`` must change labels on EVERY fire whose chunk has nonzero
    RPS ("match_fired") — the check that label augmentation actually reaches
    the targets. (All-zero-RPS chunks — full-flight policies with
    ``min_motor_rps: 0`` contain pre-takeoff ground windows — are exempt:
    ``0 * alpha == 0``.)
    """
    if key == "augmentations":
        return "zero"
    if key != "noise_augmentations":
        return None
    for stage in _stage_blocks(policy):
        spec = stage.get(key)
        if not isinstance(spec, Mapping):
            continue
        for choice in spec.get("choices", []):
            name = choice if isinstance(choice, str) else next(iter(choice))
            if name != "freq_scale":
                return None
    return "match_fired"


def _fire_verdict(k: int, n: int, p: float) -> bool:
    from scipy.stats import binomtest

    if p <= 0.0:
        return k == 0
    if p >= 1.0:
        return k == n
    return float(binomtest(k, n, p).pvalue) >= P_VALUE_FLOOR


def _tensors_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and bool(torch.equal(a, b))


def load_experiment(name: str) -> tuple[str, bool, int | None]:
    """(policy_path, flatten_channels, samples_per_validation) of an experiment.

    Composed exactly like ``train.py experiment=<name>`` (and like
    ``scripts/rps_predictor_vk_eval.py::load_model``), so the tool checks the
    stream the training run would actually consume.
    """
    from hydra import compose, initialize_config_dir

    from training.config import register_configs

    register_configs()
    with initialize_config_dir(config_dir=str(_ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=[f"experiment={name}"])
    train = cfg.data.train
    target = str(train.get("_target_", ""))
    if "OnlineMixFrameDataset" not in target:
        raise SystemExit(
            f"experiment {name!r}: data.train._target_ = {target!r} is not an "
            "online-mix stream — nothing to check"
        )
    params = train.params
    spe = cfg.get("samples_per_validation")
    return (
        str(params.path),
        bool(params.get("flatten_channels", False)),
        int(spe) if spe else None,
    )


def make_control(ds: Any, key: str) -> Any | None:
    """A dataset sharing ``ds``'s pools whose ``key`` blocks never fire.

    Returns None when ``key`` has probability 0 / is absent in every stage
    (the control would be byte-identical to the real stream).
    """
    from data_processing.online_mixing import OnlineMixIterableDataset

    policy = copy.deepcopy(ds.policy)
    changed = False
    for stage in _stage_blocks(policy):
        spec = stage.get(key)
        if isinstance(spec, dict) and float(spec.get("probability", 0.0)) > 0.0:
            spec["probability"] = NEVER_FIRE_P
            changed = True
    if not changed:
        return None
    return OnlineMixIterableDataset(
        ds.noise_pool,
        ds.source_pool,
        policy=policy,
        base_seed=ds.base_seed,
        duration_s=ds.duration_s,
        sample_rate=ds.sample_rate,
        n_fft=ds.n_fft,
        hop_length=ds.hop_length,
        start_sample_id=ds.start_sample_id,
        task=ds.task,
    )


def stage_index(policy: Mapping[str, Any], gid: int) -> int:
    """Index (1-based) of the stage active for ``gid`` (0 = flat policy)."""
    stages = policy.get("stages")
    if not stages:
        return 0
    for i, stage in enumerate(stages, start=1):
        until = stage.get("until")
        if until is None or int(gid) < int(until):
            return i
    return len(stages)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify an online-mix stream against its policy's intent "
        "(frame expansion, stage boundaries, empirical fire rates, determinism).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--policy", help="online-mix policy YAML (conf/online_mix/*.yaml)")
    src.add_argument("--experiment", help="Hydra experiment name (conf/experiment/*.yaml)")
    ap.add_argument(
        "--flatten",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="flatten_channels used at train time (with --experiment: taken from "
        "the composed config unless given explicitly)",
    )
    ap.add_argument(
        "--samples-per-epoch",
        type=int,
        default=None,
        help="training frames per validation interval / 'epoch' "
        "(samples_per_validation; default 5000, or the experiment's value)",
    )
    ap.add_argument("--epochs", type=float, nargs="+", default=[0, 5, 10, 12])
    ap.add_argument("--probes", type=int, default=48, help="sample ids probed per epoch")
    ap.add_argument("--c-chunks", type=int, default=16, help="chunks used to measure C")
    ap.add_argument(
        "--determinism-ids", type=int, default=4, help="ids regenerated for the determinism check"
    )
    ap.add_argument(
        "--warn-boundary-epoch",
        type=float,
        default=40.0,
        help="WARN when a finite stage boundary lands beyond this effective epoch",
    )
    args = ap.parse_args()

    t0 = time.perf_counter()
    failures: list[str] = []
    warnings: list[str] = []

    if args.experiment:
        policy_path, flatten, spe = load_experiment(args.experiment)
        if args.flatten is not None:
            flatten = bool(args.flatten)
        if args.samples_per_epoch is not None:
            spe = args.samples_per_epoch
        if not spe:
            raise SystemExit(
                f"experiment {args.experiment!r} sets no samples_per_validation; "
                "pass --samples-per-epoch"
            )
    else:
        policy_path = args.policy
        flatten = bool(args.flatten) if args.flatten is not None else False
        spe = args.samples_per_epoch or 5000

    from omegaconf import OmegaConf

    from data_processing.online_mixing import OnlineMixIterableDataset, _resolve_policy, make_rng

    # Self-check: the per-sample RNG must be a pure function of (seed, id).
    assert make_rng(1, 7).random() == make_rng(1, 7).random(), "make_rng is not deterministic"

    print(f"== Stream check: {policy_path} ==")
    print(f"flatten_channels={flatten} samples_per_epoch={spe} probes={args.probes}")

    t_build = time.perf_counter()
    ds = OnlineMixIterableDataset.from_config(OmegaConf.load(str(policy_path)))
    print(f"[build] pools ready in {time.perf_counter() - t_build:.1f} s (task={ds.task})")
    policy = ds.policy

    # ── [1] frame expansion ratio C ────────────────────────────────────────
    t1 = time.perf_counter()
    chan_counts: Counter[int] = Counter()
    for gid in range(args.c_chunks):
        audio, _labels = ds.generate_sample(gid)
        chan_counts[int(audio.shape[0]) if audio.ndim >= 2 else 1] += 1
    modal_c = chan_counts.most_common(1)[0][0]
    fpc = modal_c if (flatten and modal_c > 1) else 1  # training frames per generated chunk
    print(f"\n[1] Frame expansion ({args.c_chunks} chunks, {time.perf_counter() - t1:.1f} s)")
    print(f"  audio channels per chunk: {dict(sorted(chan_counts.items()))}")
    print(f"  frames per chunk (flatten={'on' if flatten else 'off'}): C = {fpc}")
    if len(chan_counts) > 1:
        warnings.append(f"mixed channel counts across chunks: {dict(chan_counts)}")

    # ── [2] stage boundaries in effective epochs ───────────────────────────
    print(
        f"\n[2] Stage boundaries (until is in GENERATED CHUNKS; 1 chunk = {fpc} training frame(s))"
    )
    stages = _stage_blocks(policy)
    print(f"  {'stage':<7}{'until(chunks)':<15}{'frames':<12}{'epoch':<10}keys")
    for i, stage in enumerate(stages, start=1):
        until = stage.get("until")
        keys = ",".join(sorted(k for k in stage if k != "until"))
        if until is None:
            print(f"  {i:<7}{'inf':<15}{'-':<12}{'-':<10}{keys}")
            continue
        frames = int(until) * fpc
        epoch = frames / spe
        print(f"  {i:<7}{int(until):<15}{frames:<12}{epoch:<10.2f}{keys}")
        if epoch > args.warn_boundary_epoch:
            warnings.append(
                f"stage {i} boundary at effective epoch {epoch:.1f} "
                f"(> {args.warn_boundary_epoch:g}) — staging-bug territory, check units"
            )

    # ── [3] empirical fire rates per probed epoch ──────────────────────────
    present_keys = [
        k for k in AUG_KEYS if any(_key_probability(s, k) > 0.0 for s in _stage_blocks(policy))
    ]
    absent_keys = [k for k in AUG_KEYS if k not in present_keys]
    controls = {k: make_control(ds, k) for k in present_keys}
    assert all(c is not None for c in controls.values())

    print(f"\n[3] Empirical fire rates ({args.probes} probes/epoch)")
    for k in absent_keys:
        print(f"  {k}: absent (p=0) in every stage — structural 0, not measured")
    if present_keys:
        print(
            f"  {'epoch':<7}{'chunk ids':<16}{'stage':<7}{'key':<22}"
            f"{'p_cfg':<8}{'fired':<10}{'rate':<8}{'label-diff':<12}verdict"
        )
    t3 = time.perf_counter()
    real_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for epoch in args.epochs:
        start = int(round(epoch * spe / fpc))
        ids = list(range(start, start + args.probes))
        for gid in ids:
            if gid not in real_cache:
                real_cache[gid] = ds.generate_sample(gid)
        sidx = {stage_index(policy, gid) for gid in ids}
        stage_str = "/".join(str(s) for s in sorted(sidx))
        for key in present_keys:
            ctl = controls[key]
            assert ctl is not None
            fired = 0
            label_diff = 0
            fired_nonzero_rps = 0  # fires on chunks whose labels are not all-zero
            for gid in ids:
                ra, rl = real_cache[gid]
                ca, cl = ctl.generate_sample(gid)
                this_fired = not (_tensors_equal(ra, ca) and _tensors_equal(rl, cl))
                if this_fired:
                    fired += 1
                    if bool(torch.any(cl != 0)):
                        fired_nonzero_rps += 1
                if not _tensors_equal(rl, cl):
                    label_diff += 1
            p_cfg = float(
                sum(_key_probability(_resolve_policy(policy, g), key) for g in ids) / len(ids)
            )
            ok = _fire_verdict(fired, len(ids), p_cfg)
            rule = _label_rule(policy, key)
            label_ok = True
            if rule == "zero":
                label_ok = label_diff == 0
            elif rule == "match_fired":
                label_ok = label_diff == fired_nonzero_rps
            verdict = "PASS" if (ok and label_ok) else "FAIL"
            if verdict == "FAIL":
                expect = fired_nonzero_rps if rule == "match_fired" else 0
                failures.append(
                    f"epoch {epoch:g} {key}: fired {fired}/{len(ids)} vs p_cfg={p_cfg:.2f}"
                    + (
                        ""
                        if label_ok
                        else f"; label-diff {label_diff} != expected {expect} (rule {rule!r})"
                    )
                )
            print(
                f"  {epoch:<7g}{f'{ids[0]}..{ids[-1]}':<16}{stage_str:<7}{key:<22}"
                f"{p_cfg:<8.2f}{f'{fired}/{len(ids)}':<10}{fired / len(ids):<8.3f}"
                f"{f'{label_diff}/{len(ids)}':<12}{verdict}"
            )
    print(f"  ({time.perf_counter() - t3:.1f} s)")

    # ── [4] determinism ────────────────────────────────────────────────────
    t4 = time.perf_counter()
    all_ids = sorted(real_cache) or [0]
    step = max(1, len(all_ids) // max(1, args.determinism_ids))
    det_ids = all_ids[::step][: args.determinism_ids]
    bad = []
    for gid in det_ids:
        ra, rl = real_cache.get(gid) or ds.generate_sample(gid)
        a2, l2 = ds.generate_sample(gid)
        if not (_tensors_equal(ra, a2) and _tensors_equal(rl, l2)):
            bad.append(gid)
    det_ok = not bad
    if not det_ok:
        failures.append(f"determinism: ids {bad} not bit-identical on regeneration")
    print(
        f"\n[4] Determinism: {'PASS' if det_ok else 'FAIL'} "
        f"({len(det_ids) - len(bad)}/{len(det_ids)} ids bit-identical, "
        f"{time.perf_counter() - t4:.1f} s)"
    )

    for w in warnings:
        print(f"WARN: {w}")
    total = time.perf_counter() - t0
    if failures:
        print(f"\nRESULT: FAIL ({len(failures)} failure(s)) — total {total:.1f} s")
        for f in failures:
            print(f"  FAIL: {f}")
        return 1
    print(f"\nRESULT: PASS — total {total:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
