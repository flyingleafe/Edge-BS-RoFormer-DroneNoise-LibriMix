"""Acceptance tests of the trajectory goodness-of-fit harness (issue 17 phase 6a).

The core of the verification is a SYNTHETIC comb whose true trajectory is
known. The harness must rank truth above a 0.5 %-scaled corruption and above a
quantisation-staircase corruption, on every one of its three components; and
the off-comb null — the same statistic at a half-integer comb, where no line
can exist — must be blind to the difference. A harness that cannot do that on
data it was handed the answer to cannot be trusted on DREGON.
"""

from __future__ import annotations

import numpy as np
import pytest

from tracking import fitness as F

SR = 16000
DUR_S = 8.0
HOP_S = 0.008  # 125 Hz frame grid: fine enough to carry the 49.7 Hz staircase
#: Well-separated rates: the conditioning gate (a k-scaled band of b0 rev/s)
#: empties the harmonic set whenever two rotors sit within ~1 rev/s.
RATES = (70.0, 95.0)


def _cfg(**over):
    base = {"k_min": 2, "k_max": 20, "n_blocks": 4, "fs_env": 250.0, "b0_revs": 1.0}
    base.update(over)
    return F.FitnessConfig(**base)


def synth_window(seed: int = 0, noise: float = 0.05, n_ch: int = 2, rates=RATES, wobble=0.5):
    """``(audio (C, T), ft (N,), r_true (R, N))`` — a comb with a known trajectory.

    ``rates`` defaults to the well-separated pair; pass a DREGON-like twin pair
    (with a ``wobble`` smaller than their separation, or they cross and every
    gate closes) to exercise the two admission gates against each other.
    """
    rng = np.random.default_rng(seed)
    n_t = int(DUR_S * SR)
    t = np.arange(n_t) / SR
    ft = np.arange(0.0, DUR_S - HOP_S / 2, HOP_S)
    r_true = np.stack(
        [
            base + wobble * np.sin(2 * np.pi * (0.3 + 0.1 * i) * ft + i)
            for i, base in enumerate(rates)
        ]
    )
    ks = np.arange(1, 26)
    audio = np.zeros((n_ch, n_t))
    for i in range(r_true.shape[0]):
        r_aud = np.interp(t, ft, r_true[i])
        phi = 2 * np.pi * np.cumsum(r_aud) / SR
        for k in ks:
            amp = 1.0 / k
            for c in range(n_ch):
                audio[c] += amp * np.cos(k * phi + rng.uniform(0, 2 * np.pi))
    audio += noise * rng.standard_normal(audio.shape) * np.abs(audio).mean()
    return audio, ft, r_true


def staircase(r: np.ndarray, ft: np.ndarray) -> np.ndarray:
    """The DREGON tachometer's own corruption: quantise, then zero-order hold."""
    q = np.round(r / F.TACH_STEP_REV_S) * F.TACH_STEP_REV_S
    tick = np.floor(ft * F.TACH_REFRESH_HZ).astype(int)
    idx = np.searchsorted(tick, np.arange(tick[-1] + 1), side="left")
    held = np.empty_like(q)
    for j, start in enumerate(idx):
        stop = idx[j + 1] if j + 1 < len(idx) else q.shape[-1]
        held[:, start:stop] = q[:, start : start + 1]
    return held


@pytest.fixture(scope="module")
def window():
    return synth_window()


@pytest.fixture(scope="module")
def candidates(window):
    _audio, ft, r_true = window
    return {
        "truth": r_true,
        "scaled": r_true * 1.005,
        "staircase": staircase(r_true, ft),
    }


def _score(window, cand, ref, *, control="none", cfg=None, holdout=None):
    audio, ft, _ = window
    cfg = cfg or _cfg()
    cells = F.window_cells(audio, ft, cand, ref, cfg=cfg, control=control)
    return F.score_cells(cells, holdout or F.Holdout.none(), cfg=cfg), cells


# ---------------------------------------------------------------------------
# the acceptance criteria


@pytest.mark.parametrize("corrupt", ["scaled", "staircase"])
def test_truth_outranks_corruption_on_every_component(window, candidates, corrupt):
    ref = candidates["truth"]
    good, _ = _score(window, candidates["truth"], ref)
    bad, _ = _score(window, candidates[corrupt], ref)
    for comp in ("broadband", "phase_noise", "roughness"):
        assert good.component(comp) < bad.component(comp), (
            f"{comp}: truth {good.component(comp):.6g} "
            f"not below {corrupt} {bad.component(comp):.6g}"
        )


@pytest.mark.parametrize("corrupt", ["scaled", "staircase"])
def test_offcomb_null_cannot_tell_them_apart(window, candidates, corrupt):
    """The null must be blind: no rotor line lives at the half-integer comb."""
    ref = candidates["truth"]
    good, _ = _score(window, candidates["truth"], ref, control="offcomb")
    bad, _ = _score(window, candidates[corrupt], ref, control="offcomb")
    for comp in ("broadband", "phase_noise"):
        a, b = good.component(comp), bad.component(comp)
        assert abs(a - b) / max(abs(a), abs(b), 1e-30) < 0.25, (
            f"off-comb {comp} separates the candidates: {a:.6g} vs {b:.6g}"
        )


def test_null_ratio_beats_the_on_comb_ratio(window, candidates):
    """The discrimination must be an on-comb property, not an estimator artifact."""
    ref = candidates["truth"]
    on_g, _ = _score(window, candidates["truth"], ref)
    on_b, _ = _score(window, candidates["scaled"], ref)
    off_g, _ = _score(window, candidates["truth"], ref, control="offcomb")
    off_b, _ = _score(window, candidates["scaled"], ref, control="offcomb")
    on = on_b.phase_noise / on_g.phase_noise
    off = off_b.phase_noise / off_g.phase_noise
    assert on > 2.0 * off, f"on-comb ratio {on:.3f} vs null ratio {off:.3f}"


# ---------------------------------------------------------------------------
# fixed degrees of freedom, hold-outs, controls


def test_degrees_of_freedom_are_identical_across_candidates_and_controls(window, candidates):
    ref = candidates["truth"]
    counts = set()
    for name in candidates:
        for control in ("none", "offcomb", "permute"):
            s, cells = _score(window, candidates[name], ref, control=control)
            counts.add((s.n_cells, s.n_cells_ridge, cells[0].shape))
    assert len(counts) == 1, f"the cell set moved between candidates/controls: {counts}"


# ---------------------------------------------------------------------------
# the ridge component (phase 6d)


@pytest.mark.parametrize("corrupt", ["scaled", "staircase"])
def test_ridge_ranks_truth_above_corruption(window, candidates, corrupt):
    """The component the eye uses: line power on the carrier over the local floor."""
    ref = candidates["truth"]
    good, _ = _score(window, candidates["truth"], ref)
    bad, _ = _score(window, candidates[corrupt], ref)
    assert good.ridge > bad.ridge + 3.0, (
        f"ridge: truth {good.ridge:.2f} vs {corrupt} {bad.ridge:.2f}"
    )


def test_ridge_reads_about_zero_on_the_off_comb_null(window, candidates):
    """No line can exist at a half-integer comb, so the ratio must be ~1 (0 dB).

    This is the property the median-to-mean correction buys: the raw median of
    an exponential annulus is ln 2 of its mean and would put every pure-noise
    cell at +1.6 dB, which is a floor the on-comb reading would have to clear.
    """
    for name in candidates:
        s, _ = _score(window, candidates[name], candidates["truth"], control="offcomb")
        assert abs(s.ridge) < 2.0, f"{name}: off-comb ridge {s.ridge:.3f} dB is not ~0"


def test_ridge_separates_on_comb_from_its_null_far_more_than_the_shares_do(window, candidates):
    """The 6d claim: the ridge sees a correct carrier where the shares saturate."""
    ref = candidates["truth"]
    on, _ = _score(window, ref, ref)
    off, _ = _score(window, ref, ref, control="offcomb")
    assert on.ridge - off.ridge > 10.0


def test_ridge_gate_survives_a_twin_pair_that_empties_the_conditioning_gate():
    """DREGON's geometry: two rotors 0.42 rev/s apart, in band at every harmonic.

    The conditioning gate needs an empty band and finds none; the ridge gate
    needs the sibling resolved away from DC and finds nearly all of them. That
    difference is what phase 6c was blind with, so it is an assertion.
    """
    rates = (86.10, 85.68)
    audio, ft, r_true = synth_window(rates=rates, wobble=0.05)
    cfg = _cfg(b0_revs=1.0)
    cells = F.window_cells(audio, ft, r_true, r_true, cfg=cfg)
    conditioning = float(np.mean([c.admit.mean() for c in cells]))
    ridge = float(np.mean([c.admit_ridge.mean() for c in cells]))
    assert conditioning < 0.10, f"the twin geometry no longer empties the gate ({conditioning:.3f})"
    assert ridge > 0.5, f"the ridge gate lost its coverage ({ridge:.3f})"
    score = F.score_cells(cells, cfg=cfg)
    assert score.ridge > 10.0, f"no lock on a twin pair's own truth ({score.ridge:.2f} dB)"


def test_line_power_is_the_phase_seven_reading(window):
    """One implementation: the promoted readout must equal the code it replaced."""
    rng = np.random.default_rng(3)
    freqs = np.linspace(0.0, 4000.0, 2001)
    power = rng.exponential(1.0, freqs.size)
    power[1000] += 500.0
    for center, half_bw in ((2000.0, 8.0), (100.0, 4.0)):
        off = freqs - center
        band = np.abs(off) <= half_bw
        ann = (np.abs(off) > 3.0 * half_bw) & (np.abs(off) <= 8.0 * half_bw)
        floor = float(np.median(power[ann]))
        resid = np.clip(power[band] - floor, 0.0, None)
        lp = F.line_power(power, freqs, center, half_bw)
        assert np.isclose(float(lp.total), float(resid.sum()))
        assert np.isclose(
            float(lp.spread_hz),
            float(np.sqrt(np.sum(resid * off[band] ** 2) / resid.sum())),
        )


def test_admission_gate_keeps_some_cells(window, candidates):
    _s, cells = _score(window, candidates["truth"], candidates["truth"])
    assert all(c.admit.any() for c in cells)
    assert np.mean([c.admit.mean() for c in cells]) > 0.2


def test_holdout_masks_partition_the_axes():
    ks = tuple(range(2, 8))
    even = F.Holdout.harmonics(0).score_mask(3, ks, 4)
    odd = F.Holdout.harmonics(1).score_mask(3, ks, 4)
    assert not (even & odd).any()
    assert (even | odd).all()
    ch = F.Holdout.channels((0,)).score_mask(3, ks, 4)
    assert not ch[0].any() and ch[1].all() and ch[2].all()
    bl = F.Holdout.blocks((0, 2)).score_mask(3, ks, 4)
    assert not bl[..., 0].any() and bl[..., 1].all()


@pytest.mark.parametrize("kind", ["harmonics", "channels", "blocks"])
def test_holdout_scores_still_rank_truth_first(window, candidates, kind):
    """§A: a correct trajectory lines up the units it was never fitted on."""
    ref = candidates["truth"]
    holdout = {
        "harmonics": F.Holdout.harmonics(0),
        "channels": F.Holdout.channels((0,)),
        "blocks": F.Holdout.blocks((0, 2)),
    }[kind]
    good, _ = _score(window, candidates["truth"], ref, holdout=holdout)
    bad, _ = _score(window, candidates["scaled"], ref, holdout=holdout)
    assert good.phase_noise < bad.phase_noise
    assert good.broadband < bad.broadband


def test_rotor_permutation_leaves_the_acoustic_score_alone(window, candidates):
    """Documented invariance: the components see the carrier SET, not the slots."""
    ref = candidates["truth"]
    plain, _ = _score(window, candidates["truth"], ref)
    perm, _ = _score(window, candidates["truth"], ref, control="permute")
    for comp in ("broadband", "phase_noise", "roughness"):
        a, b = plain.component(comp), perm.component(comp)
        assert abs(a - b) / max(abs(a), abs(b), 1e-30) < 1e-6


def test_rotor_permutation_destroys_the_pairing(window, candidates):
    """...which is why the permutation null belongs to the residual half."""
    _audio, ft, r_true = window
    carriers, _skip, _half = F.apply_control(r_true, "permute")
    plain = F.residual_decompose(r_true, r_true, ft, cfg=_cfg())
    perm = F.residual_decompose(carriers, r_true, ft, cfg=_cfg())
    assert perm["pooled"]["d_rms"] > 50.0 * max(plain["pooled"]["d_rms"], 1e-6)


def test_mismatch_control_needs_a_partner(window, candidates):
    with pytest.raises(ValueError, match="partner"):
        F.apply_control(candidates["truth"], "mismatch")
    with pytest.raises(ValueError, match="unknown control"):
        F.apply_control(candidates["truth"], "nope")


def test_mismatched_telemetry_scores_worse_than_the_truth(window, candidates):
    """Correspondence-breaking null: a different window's trajectory."""
    ref = candidates["truth"]
    other, _ft, _r = synth_window(seed=7)
    _ = other
    partner = ref[:, ::-1].copy()  # the same rates, the wrong time correspondence
    audio, ft, _ = window
    cfg = _cfg()
    good = F.score_cells(F.window_cells(audio, ft, ref, ref, cfg=cfg), cfg=cfg)
    mis = F.score_cells(
        F.window_cells(audio, ft, ref, ref, cfg=cfg, control="mismatch", partner=partner),
        cfg=cfg,
    )
    assert mis.phase_noise > good.phase_noise


# ---------------------------------------------------------------------------
# residual decomposition and bootstrap


def test_residual_decompose_recovers_a_pure_scale(window):
    _audio, ft, r_true = window
    out = F.residual_decompose(r_true * 1.005, r_true, ft, cfg=_cfg())
    assert out["pooled"]["scale_pct"] == pytest.approx(0.5, abs=0.02)
    assert abs(out["pooled"]["lag_s"]) < 0.01
    assert out["pooled"]["resid_rms"] < 1e-3


def test_residual_decompose_recovers_a_pure_lag(window):
    _audio, ft, r_true = window
    lag = 0.05
    shifted = np.stack([np.interp(ft - lag, ft, r_true[i]) for i in range(len(r_true))])
    out = F.residual_decompose(shifted, r_true, ft, cfg=_cfg())
    assert out["pooled"]["lag_s"] == pytest.approx(lag, abs=0.01)


def test_residual_decompose_reads_the_tachometer_signature(window):
    _audio, ft, r_true = window
    out = F.residual_decompose(staircase(r_true, ft), r_true, ft, cfg=_cfg())
    assert out["pooled"]["f_tach_resolved"] is True
    assert out["pooled"]["tach_bound_frac"] > 0.9  # bounded by half a step
    assert out["per_rotor"][0]["tach_line_ratio"] is not None


def test_bootstrap_brackets_the_point_estimate(window, candidates):
    ref = candidates["truth"]
    audio, ft, _ = window
    cfg = _cfg()
    cells = F.window_cells(audio, ft, ref, ref, cfg=cfg)
    point = F.score_cells(cells, cfg=cfg)
    boot = F.bootstrap_scores(cells, cfg=cfg, n_boot=40, seed=1)
    for comp in ("broadband", "phase_noise", "roughness"):
        ci = boot[comp]
        assert ci["lo"] <= point.component(comp) <= ci["hi"]
        assert ci["sd"] > 0


def test_score_window_payload_is_json_shaped(window, candidates):
    import json

    audio, ft, _ = window
    out = F.score_window(audio, ft, candidates["truth"], candidates["truth"], cfg=_cfg(), n_boot=8)
    json.dumps(out)  # must not raise
    assert set(out["scores"]) >= {"none", "fit_k_even", "fit_ch_0"}
    assert out["cells"]["n_rotors"] == len(RATES)


def test_fitness_stage_appends_diagnostics(window, candidates):
    from tracking.stages import tracking_frame

    audio, ft, r_true = window
    frame = tracking_frame(audio, SR, rps=r_true, frame_times=ft, rps_meas=r_true)
    out = F.fitness_stage(cfg=_cfg())(frame)
    log = out["meta"]["tracking"]
    assert log[-1]["stage"] == "fitness"
    assert "scores" in log[-1]
