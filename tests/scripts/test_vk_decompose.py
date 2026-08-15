"""Unit tests for the importable core of ``scripts/vk_decompose.py``.

The heavy test builds a synthetic two-rotor comb with a KNOWN amplitude per
harmonic, a KNOWN injected phase drift and a KNOWN white floor, then runs the
solve-and-reconstruct path the script uses (not the CLI, which needs the
published dataset). Four properties are pinned:

- The decomposition adds up: ``audio - (tracks + residual)`` is float noise.
- The recovered amplitude is the injected one.
- A COMMON (shaft) drift gives a much larger top-eigenvalue share than an
  independent per-harmonic drift — the discrimination the report is built on.
- The residual power spectral density is the injected floor.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import vk_decompose as V

from data_processing.derivations import (
    _decomp_frame_grid,
    _decomp_interp_rps,
    _decomp_to_audio_grid,
)

SR = 8000
DUR_S = 4.0
#: Two rates far enough apart that no low harmonic of one lands on the other.
RATES = (50.0, 61.0)
K_MAX = 8
AMPS = {k: 0.4 / k for k in range(1, K_MAX + 1)}
FLOOR = 0.02


# ---------------------------------------------------------------------------
# the pure helpers


def test_window_bounds_tiles_and_right_aligns() -> None:
    bounds = V.window_bounds(1996, 16.0, 12.0)
    assert bounds[0] == (0, 500)
    assert bounds[-1] == (1496, 1996)
    covered = np.zeros(1996, dtype=bool)
    for i0, i1 in bounds:
        covered[i0:i1] = True
    assert covered.all()


def test_window_span_snaps_to_the_envelope_stride() -> None:
    ft = V.frame_grid(16000 * 40, V.SR)
    a0, a1 = V.window_span(ft, 125, 250, 16000 * 40, 160)
    assert a0 % 160 == 0
    assert (a1 - a0) % 160 == 0
    assert a0 == 64000  # 125 frames of 0.032 s at 16 kHz, already aligned


def test_fade_weights_ramp_and_floor() -> None:
    w = V.fade_weights(10, 4)
    assert w[0] == pytest.approx(0.2)
    assert w[5] == pytest.approx(1.0)
    assert (w > 0).all()
    assert np.allclose(w, w[::-1])
    assert np.allclose(V.fade_weights(6, 0), 1.0)


def test_track_bands_partition_the_harmonics() -> None:
    k = np.arange(1, 81)
    masks = V.track_bands(k)
    assert sorted(masks) == ["k1-9", "k10-24", "k25-49", "k50-80"]
    assert sum(int(m.sum()) for m in masks.values()) == 80


def test_reference_mic_picks_the_loudest_channel() -> None:
    audio = np.stack([np.ones(100), 3.0 * np.ones(100), np.zeros(100)])
    assert V.reference_mic(audio, -1) == 1
    assert V.reference_mic(audio, 0) == 0


def test_reconstruct_is_a_linear_interpolation_of_the_envelope() -> None:
    # One constant unit envelope at k = 1 must rebuild cos(phi) exactly.
    stride, n_env = 10, 5
    x = np.ones((1, 1, n_env), dtype=np.complex64)
    phase = np.linspace(0.0, 3.0, n_env * stride)[None, :]
    recon, energy = V.reconstruct(x, np.array([1]), np.array([0]), phase, stride)
    assert recon.shape == (1, n_env * stride)
    assert recon[0] == pytest.approx(np.cos(phase[0]), abs=1e-6)
    assert energy[0] == pytest.approx(float((np.cos(phase[0]) ** 2).sum()), rel=1e-5)


def test_rank_one_share_separates_common_from_independent_drift() -> None:
    rng = np.random.default_rng(0)
    common = rng.normal(size=(1, 4000))
    k = np.arange(1, 13)[:, None]
    assert V.rank_one_share(k * common)["lambda1_share"] == pytest.approx(1.0, abs=1e-6)
    indep = V.rank_one_share(rng.normal(size=(12, 4000)))
    assert indep["lambda1_share"] is not None
    # The Marchenko-Pastur bar for 12 rows and 4000 columns is about 0.093.
    assert indep["lambda1_share"] < 0.15


def test_rank_one_share_refuses_a_short_window() -> None:
    assert V.rank_one_share(np.zeros((10, 4)))["lambda1_share"] is None


def test_bandwidth_schedule_is_the_cli_spelling() -> None:
    # The schedule travels as ONE string through the unit parameters and the
    # report's provenance, so the CLI spelling must round trip through JSON.
    sched = V.BandwidthSchedule.parse("3,0,1.5,3")
    assert sched is not None
    assert sched.as_dict() == {
        "bw0_hz": 3.0,
        "slope_hz_per_k": 0.0,
        "cap_frac_of_sep": 1.5,
        "bw_abs_max": 3.0,
    }
    assert V.BandwidthSchedule.parse(json.loads(json.dumps(sched.text()))) == sched
    assert V.BandwidthSchedule.parse("") is None


def test_solve_window_forwards_the_schedule_to_the_solver() -> None:
    # The driver's own seam: the same window solved flat and scheduled must come
    # back with DIFFERENT achieved bandwidths, and the scheduled one wider.
    from tracking.decompose import solve_window

    audio, rates = _synth("common")
    cfg = V.fvk_config(K_MAX, mics=2, sr=SR)
    flat = solve_window(audio, rates, cfg, k_hi=K_MAX, mics=2)
    wide = solve_window(
        audio,
        rates,
        cfg,
        k_hi=K_MAX,
        mics=2,
        bw_schedule=V.BandwidthSchedule(3.0, 0.0, 1.5, 3.0),
    )
    # The schedule is a FLOOR of 3 Hz here, so the low harmonics widen and the
    # high ones (whose sparse-comb base is already wider) stay where they were.
    assert (wide.bw_track >= flat.bw_track - 1e-9).all()
    assert float(wide.bw_track.mean()) > float(flat.bw_track.mean())
    low = np.asarray(flat.k) <= 2
    assert (wide.bw_track[low] > flat.bw_track[low] + 1e-9).all()


def test_sample_rate_and_f_max_both_cap_the_harmonic_set() -> None:
    # Two ceilings, and the SMALLER one wins: a 8 kHz f_max is inert at 16 kHz
    # because the geometry holds every line under 0.375 * sr = 6 kHz.
    r_ref = np.full((4, 100), 91.0)
    assert V.recording_k_hi(r_ref, 80, sr=16000, f_max=8000.0) == V.recording_k_hi(
        r_ref, 80, sr=16000, f_max=6000.0
    )
    assert V.recording_k_hi(r_ref, 80, sr=32000, f_max=8000.0) > V.recording_k_hi(
        r_ref, 80, sr=16000, f_max=8000.0
    )
    # The v2 configuration reaches the k_max it asks for on a DREGON rate peak.
    assert V.recording_k_hi(r_ref, 80, sr=32000, f_max=8000.0) == 80


def test_group_plan_reports_transitive_coupling_and_its_memory() -> None:
    # Coupling is TRANSITIVE, so a group is a chain and not a pair: the 16 lines
    # of 50 and 61 rev/s run 50, 61, 100, 122 ... 488 Hz, and the chain holds
    # wherever the step stays under the 50 Hz coupling distance. It breaks at
    # 250 -> 300 and at 427 -> 488, which leaves three groups and a longest
    # chain of nine. That chain length is what the memory law is written in.
    rates = np.stack([np.full(2 * SR, r) for r in RATES])
    from tracking.decompose import group_plan

    plan = group_plan(rates, K_MAX, V.fvk_config(K_MAX, mics=2, sr=SR))
    assert plan["n_tracks"] == 2 * K_MAX
    assert plan["n_groups"] == 3
    assert plan["max_group"] == 9
    g, n_env = plan["max_group"], plan["n_env"]
    assert plan["banded_gb"] == round(2 * (2 * g + 1) * g * n_env * 16 / 1e9, 3)


def test_energy_ledger_names_the_cross_term() -> None:
    audio = np.array([[1.0, 1.0, 1.0, 1.0]])
    recon = np.array([[0.5, 0.5, 0.5, 0.5]])
    led = V.energy_ledger(audio, recon, np.array([0.5, 0.6]), np.array([1, 12]))
    assert led["total"] == pytest.approx(4.0)
    assert led["residual"] == pytest.approx(1.0)
    assert led["tracks"] == pytest.approx(1.1)
    assert led["cross_term"] == pytest.approx(4.0 - 1.0 - 1.1)
    assert led["band_share_of_tracks"]["k1-9"] == pytest.approx(0.5 / 1.1, abs=1e-6)
    assert led["band_share_of_tracks"]["k50-80"] == 0.0


# ---------------------------------------------------------------------------
# the synthetic comb


def _synth(drift: str, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``(audio (2, T), rates (2, T))`` — a two-rotor comb with a known floor.

    ``drift`` is ``"common"`` (one random walk per ROTOR, seen by harmonic ``k``
    as ``k`` times that walk — the shaft model) or ``"independent"`` (one walk
    per rotor AND harmonic — the pi-kalman model). The rates handed back are the
    ones the solver is given, so the walk is exactly the phase error the
    envelopes must absorb.
    """
    rng = np.random.default_rng(seed)
    n_t = int(DUR_S * SR)
    rates = np.stack([np.full(n_t, r) for r in RATES])
    phase = 2.0 * np.pi * np.cumsum(rates, axis=-1) / float(SR)
    audio = rng.normal(scale=FLOOR, size=(2, n_t))

    # A Brownian walk of about 0.6 rad over the clip. Its rate of change is far
    # below the VK passband, so the envelope can follow it and the drift ends up
    # in the envelope phase, which is what the test reads.
    def walk() -> np.ndarray:
        w = np.cumsum(rng.normal(scale=1.0, size=n_t)) / np.sqrt(n_t)
        return 0.6 * (w - w.mean())

    for r in range(2):
        common = walk()
        for k, amp in AMPS.items():
            extra = k * common if drift == "common" else walk()
            # Two microphones, one gain each, so the array is not degenerate.
            for c, gain in enumerate((1.0, 0.7)):
                audio[c] += gain * amp * np.cos(k * phase[r] + extra + 0.3 * c)
    return audio, rates


def _decompose(drift: str) -> dict[str, object]:
    """Solve the synthetic clip and read the script's own statistics off it."""
    from tracking.decompose import solve_window

    audio, rates = _synth(drift)
    env = solve_window(audio, rates, V.fvk_config(K_MAX, mics=2, sr=SR), k_hi=K_MAX, mics=2)
    stride = int(round(SR / env.fs_env))
    phase = V.shaft_phase(rates, SR)
    recon, energy = V.reconstruct(env.x, env.k, env.rotor, phase, stride)
    residual = audio - recon
    amp = np.abs(env.x[0])
    pherr = np.unwrap(np.angle(env.x[0].astype(np.complex128)), axis=-1)
    # Drop the tapered window ends: the solver fades its data term there, so the
    # envelopes are not estimates of anything in the first and last tenth.
    mask = np.zeros(amp.shape[-1], dtype=bool)
    mask[amp.shape[-1] // 10 : -amp.shape[-1] // 10] = True
    mean, cv, drift_std = V.per_track_stats(amp, pherr, mask, env.fs_env)
    inc = V.drift_increments(pherr, env.fs_env)
    pair = mask[:-1] & mask[1:]
    shares = [
        V.rank_one_share(inc[np.ix_(np.flatnonzero(env.rotor == r), np.flatnonzero(pair))])[
            "lambda1_share"
        ]
        for r in (0, 1)
    ]
    return {
        "audio": audio,
        "recon": recon,
        "residual": residual,
        "energy": energy,
        "k": env.k,
        "amp_mean": mean,
        "amp_cv": cv,
        "drift_std": drift_std,
        "shares": shares,
    }


@pytest.fixture(scope="module")
def common_drift() -> dict[str, object]:
    return _decompose("common")


def test_decomposition_is_exact(common_drift: dict[str, object]) -> None:
    audio = np.asarray(common_drift["audio"])
    recon = np.asarray(common_drift["recon"])
    residual = np.asarray(common_drift["residual"])
    assert float(np.abs(audio - (recon + residual)).max()) < 1e-9


def test_recovered_amplitude_matches_the_injection(common_drift: dict[str, object]) -> None:
    mean = np.asarray(common_drift["amp_mean"])
    ks = np.asarray(common_drift["k"])
    for k, want in AMPS.items():
        got = float(mean[ks == k].mean())  # both rotors carry the same amplitude
        assert got == pytest.approx(want, rel=0.2), f"k={k}: {got} against {want}"


def test_common_drift_grows_with_the_harmonic(common_drift: dict[str, object]) -> None:
    # A shaft jitter is seen by harmonic k as k times one phase, so the drift
    # standard deviation must rise with k. Independent per-harmonic drift does
    # not — this is the second reading of the same increments.
    drift = np.asarray(common_drift["drift_std"])
    ks = np.asarray(common_drift["k"])
    for r_lo, r_hi in ((1, 2), (2, 4), (4, 8)):
        assert float(drift[ks == r_hi].mean()) > 1.5 * float(drift[ks == r_lo].mean())


def test_common_drift_is_more_rank_one_than_independent_drift(
    common_drift: dict[str, object],
) -> None:
    indep = _decompose("independent")
    common_shares = [float(v) for v in common_drift["shares"]]  # type: ignore[union-attr]
    indep_shares = [float(v) for v in indep["shares"]]  # type: ignore[union-attr]
    assert min(common_shares) > 0.6
    assert max(indep_shares) < 0.5
    assert min(common_shares) > max(indep_shares)


def test_residual_spectrum_is_the_injected_floor(common_drift: dict[str, object]) -> None:
    residual = np.asarray(common_drift["residual"])
    _, psd = V.welch_psd(residual, SR)
    # White noise of standard deviation FLOOR has the flat density
    # 2 * FLOOR^2 / SR. The median is read, not the mean: the residual keeps a
    # small line at each of the 16 modelled harmonics.
    want = 2.0 * FLOOR**2 / SR
    assert float(np.median(psd[0])) == pytest.approx(want, rel=0.3)


# ---------------------------------------------------------------------------
# the v3 joint mode


def test_joint_config_round_trips_the_cli_spelling() -> None:
    # Every joint knob travels as a scalar or a comma-separated string through
    # the unit parameters and the report's provenance, so it must survive JSON.
    params = {"iters": 4, "k_trust": "3,12,80", "bw_psi": "0.6,8", "bw_theta": 1.5}
    jc = V.joint_config(json.loads(json.dumps(params)))
    assert jc.iters == 4
    assert jc.k_trust == (3, 12, 80)
    assert jc.bw_psi_slope == 0.6
    assert jc.bw_psi_max == 8.0
    assert jc.bw_theta_hz == 1.5
    assert jc.whiten is True
    assert V.joint_config({}).k_trust == (3, 12, 80)  # the shipped ladder
    assert V.joint_config({"whiten": False}).whiten is False


def test_objective_pool_sums_the_windows_and_normalizes() -> None:
    # The MAP objective is EXTENSIVE, so windows pool by addition; the per-cell
    # view is what makes two recordings of different length comparable.
    def row(a0: int, total: float, cells: int) -> dict:
        return {
            "a0": a0,
            "start_s": a0 / 16000.0,
            "objective": {
                "total": total,
                "data": total / 2,
                "rent": total / 2,
                "phase_priors": 0.0,
                "envelope_prior": 0.0,
                "n_cells": cells,
            },
        }

    assert V.objective_pool([{"a0": 0, "used": True}]) is None
    got = V.objective_pool([row(320, -2.0, 30), row(0, -1.0, 10)])
    assert got is not None
    assert got["n_windows"] == 2
    assert got["pooled"]["total"] == pytest.approx(-3.0)
    assert got["pooled"]["n_cells"] == 40
    assert got["per_cell"]["total"] == pytest.approx(-3.0 / 40)
    assert [w["a0"] for w in got["windows"]] == [0, 320]  # ordered in time


def test_joint_solve_writes_the_extra_arrays_and_stitches(tmp_path) -> None:
    """The v3 unit -> npz -> stitch path, without the published dataset.

    Two overlapping windows are solved jointly, written exactly as the worker
    writes them, and stitched. What is pinned is the SEAM: the extra arrays are
    on disk, the stitch recognises them, it hands back a CORRECTED carrier, and
    the reconstruction against that carrier still explains the clip.
    """
    from tracking import joint_solve_window
    from tracking.decompose import solve_config
    from tracking.joint_decompose import JointConfig, theta_rate

    audio, rates = _synth("common")
    cfg = solve_config(K_MAX, sr=SR, mics=2)
    stride = int(round(SR / 100.0))
    n_t = audio.shape[-1]
    raw = tmp_path / "raw"
    raw.mkdir()

    rows = []
    for w, a0 in enumerate((0, (n_t // 2 // stride) * stride)):
        a1 = n_t if w else (3 * n_t // 4 // stride) * stride
        res = joint_solve_window(
            audio[:, a0:a1],
            rates[:, a0:a1],
            cfg,
            k_hi=K_MAX,
            mics=2,
            jcfg=JointConfig(iters=2, k_trust=(2, 6), profile_n_fft=2048, psd_n_fft=2048),
        )
        env = res.env
        np.savez(
            raw / f"w{w}.npz",
            allow_pickle=False,
            x=np.asarray(env.x, dtype=np.complex64),
            valid=np.asarray(env.valid, dtype=bool),
            rotor=np.asarray(env.rotor, dtype=np.int64),
            k=np.asarray(env.k, dtype=np.int64),
            bw_track=np.asarray(env.bw_track, dtype=np.float64),
            theta=res.theta_env,
            dr=theta_rate(res.theta_env, float(env.fs_env)),
            psi=np.asarray(res.psi, dtype=np.float32),
            psd_freq=res.psd.freq,
            psd_t=res.psd.t_block,
            psd_log_s=np.asarray(res.psd.log_s, dtype=np.float32),
        )
        rows.append({"used": True, "npz": f"raw/w{w}.npz", "a0": int(a0), "kind": "window"})
        assert res.iterations[0]["k_trust"] == 2
        assert "order_cell" in res.iterations[-1]

    phi = V.shaft_phase(rates, SR)
    st = V.stitch_envelopes(rows, tmp_path, phi, stride, ramp=12, r_audio=rates, sr=SR)
    assert st["joint"] is True
    assert np.asarray(st["dr_global"]).shape[0] == rates.shape[0]
    assert st["theta_stitch_max_rate_hz"] < 50.0  # inside the 100 Hz envelope grid
    # The corrected carrier is the one the stitched bank belongs to, and the
    # reconstruction against it must explain most of the clip.
    a_min, a_max = int(st["a_min"]), int(st["a_max"])
    recon, _ = V.reconstruct(
        st["x"], st["k"], st["rotor"], np.asarray(st["phi"])[:, a_min:a_max], stride
    )
    clip = audio[:, a_min:a_max]
    assert float(((clip - recon) ** 2).sum() / (clip**2).sum()) < 0.35


def _stitch_one(tmp_path, **over) -> dict:
    """Run the driver's whole stitch on a synthetic recording, and read it back.

    ``get_recording`` is fed through the per-process cache the pool warms, so
    the published dataset is never touched and the path under test is the real
    one — solve unit, ``.npz``, stitch, report.
    """
    from tracking.decompose import solve_window

    audio, rates = _synth("common")
    n_t = int(audio.shape[-1])
    env = solve_window(audio, rates, V.fvk_config(K_MAX, mics=2, sr=SR), k_hi=K_MAX, mics=2)
    stride = int(round(SR / env.fs_env))
    raw = tmp_path / "raw"
    raw.mkdir()
    np.savez(
        raw / "w0.npz",
        allow_pickle=False,
        x=np.asarray(env.x, dtype=np.complex64),
        valid=np.asarray(env.valid, dtype=bool),
        rotor=np.asarray(env.rotor, dtype=np.int64),
        k=np.asarray(env.k, dtype=np.int64),
        bw_track=np.asarray(env.bw_track, dtype=np.float64),
    )
    (raw / "w0.json").write_text(
        json.dumps(
            {
                "used": True,
                "kind": "window",
                "npz": "raw/w0.npz",
                "a0": 0,
                "recording": "REC",
                "reason": "ok",
            }
        )
    )
    ft = V.frame_grid(n_t, SR)
    V.cache_recordings(
        [
            {
                "recording_id": "REC",
                "audio": audio.astype(np.float32),
                "ft": ft,
                "r_ref": V.interp_rps(rates[:, :: max(1, n_t // ft.size)][:, : ft.size], ft, ft),
                "r_audio": rates,
                "t0_offset_s": 0.0,
                "rps_key": "motors_measured",
                "sr": SR,
            }
        ],
        SR,
    )
    params = {
        "window_s": DUR_S,
        "hop_s": DUR_S,
        "fs_env": 100.0,
        "stride": stride,
        "ref_mic": -1,
        "sr": SR,
        **over,
    }
    V.stitch(tmp_path, "spec", "labels", params, only={"REC"})
    return json.loads((tmp_path / "REC" / "report.json").read_text())


def test_stochastic_is_off_by_default_in_the_stitch(tmp_path) -> None:
    report = _stitch_one(tmp_path)
    assert "stochastic" not in report
    assert sorted(report["order_cell"]) == ["original", "residual"]
    with np.load(tmp_path / "REC" / "residual.npz", allow_pickle=False) as data:
        assert "stochastic" not in data


def test_stochastic_writes_the_channel_and_the_gate_reading(tmp_path) -> None:
    # The flag ON adds ONE array and TWO report blocks, and the identity the
    # whole split rests on is written into the report so a consumer can read it
    # without recomputing anything.
    report = _stitch_one(tmp_path, stochastic=True)
    assert report["params"]["stochastic"] is True
    assert "residual_final" in report["order_cell"]
    st = report["stochastic"]
    assert st["carrier"] == "labels"  # a v2 stitch has no corrected carrier
    assert 0.0 <= st["stochastic_fraction"] <= 1.0
    assert st["identity_max_abs"] < 1e-6 * st["residual_energy"] ** 0.5
    assert st["wola_min_weight"] > 0.0
    with np.load(tmp_path / "REC" / "residual.npz", allow_pickle=False) as data:
        resid = np.asarray(data["residual"], dtype=np.float64)
        stoch = np.asarray(data["stochastic"], dtype=np.float64)
        assert stoch.shape == resid.shape
        # residual = stochastic + broadband, so the consumer's own subtraction
        # is the broadband channel and nothing has to be shipped twice.
        assert float(np.abs(resid - stoch).max()) <= 2.0 * float(np.abs(resid).max())


def test_v2_stitch_still_hands_back_its_own_carrier(tmp_path) -> None:
    # A v2 unit has no ``dr`` array, so the joint branch must not fire and the
    # carrier must come back untouched — the regression that keeps v2 working.
    from tracking.decompose import solve_window

    audio, rates = _synth("common")
    env = solve_window(audio, rates, V.fvk_config(K_MAX, mics=2, sr=SR), k_hi=K_MAX, mics=2)
    stride = int(round(SR / env.fs_env))
    raw = tmp_path / "raw"
    raw.mkdir()
    np.savez(
        raw / "w0.npz",
        allow_pickle=False,
        x=np.asarray(env.x, dtype=np.complex64),
        valid=np.asarray(env.valid, dtype=bool),
        rotor=np.asarray(env.rotor, dtype=np.int64),
        k=np.asarray(env.k, dtype=np.int64),
        bw_track=np.asarray(env.bw_track, dtype=np.float64),
    )
    phi = V.shaft_phase(rates, SR)
    rows = [{"used": True, "npz": "raw/w0.npz", "a0": 0, "kind": "window"}]
    st = V.stitch_envelopes(rows, tmp_path, phi, stride, ramp=0, r_audio=rates, sr=SR)
    assert st["joint"] is False
    assert np.array_equal(np.asarray(st["phi"]), phi)


# parity with the decomp-frames-v1 derivation


def test_carrier_helpers_match_vk_decompose():
    """Byte-for-byte agreement with the script that solved the envelopes.

    The amplitudes are only meaningful against the carrier they were demodulated
    with, and ``scripts/`` cannot be imported from ``src/`` — so the derivation
    carries its own copy of these three functions and this test is the pin.
    """
    rng = np.random.default_rng(0)
    n_t = 5 * SR
    stamps = np.sort(rng.uniform(0.0, 5.0, size=200))
    stamps[3] = stamps[2]  # a duplicate stamp: both copies must drop it the same way
    vals = rng.uniform(40.0, 90.0, size=(4, stamps.size))

    ft_a, ft_b = _decomp_frame_grid(n_t, SR), V.frame_grid(n_t, SR)
    np.testing.assert_array_equal(ft_a, ft_b)

    r_a = _decomp_interp_rps(vals, stamps, ft_a)
    r_b = V.interp_rps(vals, stamps, ft_b)
    np.testing.assert_array_equal(r_a, r_b)

    np.testing.assert_array_equal(
        _decomp_to_audio_grid(r_a, ft_a, n_t, SR), V.to_audio_grid(r_b, ft_b, n_t, SR)
    )
