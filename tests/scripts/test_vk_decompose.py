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


def test_stochastic_k_split_reads_a_low_in_flight_rate() -> None:
    """The regions reach the ceiling at the SLOWEST in-flight moment.

    A robust low rate, and not the mean: at the mean the top lines would fall
    short of the ceiling for half of the flight. Idle frames are not rates the
    comb exists at, so they must not set the scale.
    """
    r = np.stack([np.full(100, 50.0), np.full(100, 61.0)])
    assert V.stochastic_k_split(6000.0, r, 8) == 120  # ceil(6000 / 50)
    # A ceiling under the coherent top line takes no region away: the flag only
    # ever EXTENDS, so the split stays exactly where it was.
    assert V.stochastic_k_split(300.0, r, 8) == 8
    assert V.stochastic_k_split(0.0, r, 8) == 8
    # Idle frames are excluded, so the slow half of a takeoff does not blow the
    # harmonic count up.
    ramp = np.concatenate([np.full((1, 50), 2.0), np.full((1, 50), 60.0)], axis=-1)
    assert V.stochastic_k_split(600.0, ramp, 1) == 10
    # Nothing above idle at all: the span's mean is the only rate there is.
    assert V.stochastic_k_split(100.0, np.full((1, 10), 10.0), 1) == 10


def test_stochastic_f_ceil_at_the_coherent_cap_reproduces_the_default_path(tmp_path) -> None:
    """The explicit floor IS the floor the split fits for itself.

    With the ceiling at the coherent top line the two paths must agree bit for
    bit: same harmonic set for the regions, same block grid, same mask. That is
    what makes the flag readable as an EXTENSION of the shipped split and not as
    a second split with its own floor.
    """
    a, b = tmp_path / "default", tmp_path / "ceiling"
    a.mkdir()
    b.mkdir()
    base = _stitch_one(a, stochastic=True)
    k_coh = int(base["k_hi"])
    r_lo = float(min(RATES))  # the 5th percentile of two constant rates
    got = _stitch_one(b, stochastic=True, stochastic_f_ceil=k_coh * r_lo)

    assert base["stochastic"]["f_ceil"] == 0.0
    assert base["stochastic"]["k_split"] == k_coh
    assert got["stochastic"]["k_split"] == k_coh
    assert got["stochastic"]["f_ceil"] == pytest.approx(k_coh * r_lo)
    assert got["stochastic"]["k_hi"] == k_coh  # the split's own reading agrees
    with (
        np.load(a / "REC" / "residual.npz", allow_pickle=False) as za,
        np.load(b / "REC" / "residual.npz", allow_pickle=False) as zb,
    ):
        assert np.array_equal(za["stochastic"], zb["stochastic"])


def test_stochastic_f_ceil_opens_regions_above_the_coherent_cap(tmp_path) -> None:
    # The point of the flag: bins the coherent cap never reached now sit inside
    # a search region. The band bin fraction is the geometry, and it is what
    # must move.
    a, b = tmp_path / "default", tmp_path / "ceiling"
    a.mkdir()
    b.mkdir()
    base = _stitch_one(a, stochastic=True)
    k_coh = int(base["k_hi"])
    got = _stitch_one(b, stochastic=True, stochastic_f_ceil=2 * k_coh * float(min(RATES)))
    assert got["stochastic"]["k_split"] == 2 * k_coh
    assert got["stochastic"]["band_bin_fraction"] > base["stochastic"]["band_bin_fraction"]
    assert got["stochastic"]["stochastic_fraction"] >= base["stochastic"]["stochastic_fraction"]
    # The identity is a property of the transform, not of the region set.
    assert (
        got["stochastic"]["identity_max_abs"] < 1e-6 * got["stochastic"]["residual_energy"] ** 0.5
    )


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


# ---------------------------------------------------------------------------
# v4: the unified model, through the driver


def _v4_unit(tmp_path, **over):
    """One ``--v4`` solve unit through the REAL worker, on a cached synthetic."""
    from utils.gridrun import Unit

    audio, rates = _synth("common")
    n_t = int(audio.shape[-1])
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
    stride = int(round(SR / 100.0))
    params = {
        "recording": "REC",
        "i0": 0,
        "i1": int(ft.size),
        "spec": "spec",
        "label_dir": "labels",
        "out": str(tmp_path),
        "kind": "window",
        "window_s": DUR_S,
        "hop_s": DUR_S,
        "k_max": K_MAX,
        "mics": 2,
        "sr": SR,
        "f_max": 3000.0,
        "bw_rps": 1.0,
        "bw_schedule": "",
        "ref_mic": -1,
        "mem_budget_gb": 0.0,
        "fs_env": 100.0,
        "stride": stride,
        "joint": True,
        "v4": True,
        "iters": 2,
        "k_trust": "2,6",
        "bw_psi": "0.6,8,1.5",
        "bw_theta": 1.5,
        "whiten": True,
        "stochastic": False,
        **over,
    }
    row = V.solve_worker(Unit("w0", params))
    (tmp_path / "raw" / "w0.json").write_text(json.dumps(row))
    return row, params


def test_v4_writes_the_line_powers_and_reads_its_own_objective(tmp_path) -> None:
    """The v4 unit: the ``H`` table on disk, and ``J_v4`` in the row.

    The line powers are a PRODUCT of the v4 model — the generator's amplitude
    targets by construction — so the seam that carries them is pinned the same
    way the envelope bank's is.
    """
    row, _ = _v4_unit(tmp_path)
    assert row["used"] is True
    assert row["joint"]["config"]["v4"] is True
    # The (S, H) fit reports itself, and the two-start guard is in the record.
    fit = row["joint"]["h_fit"]
    assert fit["n_lines"] == 2 * K_MAX
    assert 0.0 <= fit["low_start_frac"] <= 1.0
    assert fit["b_f_hz"] > 0.0
    # J_v4 is the row's own measure, and it is built from its own terms.
    obj = row["objective"]
    assert obj["total_v4"] == pytest.approx(
        obj["data_v4"] + obj["phase_priors"] + obj["floor_penalty"]
    )

    with np.load(tmp_path / "raw" / "w0.npz", allow_pickle=False) as data:
        assert {"h_rotor", "h_k", "h_t", "h_lines", "h_half", "h_power"} <= set(data)
        h = np.asarray(data["h_power"], dtype=np.float64)
        assert h.shape == (2, len(data["h_t"]), 2 * K_MAX)
        assert np.all(h >= 0.0)
        # The line table is the SOLVER's track table, rotor major.
        assert np.array_equal(np.asarray(data["k"]), np.asarray(data["h_k"]))
        assert np.array_equal(np.asarray(data["rotor"]), np.asarray(data["h_rotor"]))


def test_v4_stitches_and_keeps_the_two_channel_identity(tmp_path) -> None:
    """``comb + broadband == audio`` at the DRIVER level, with no third channel."""
    _, params = _v4_unit(tmp_path)
    V.stitch(tmp_path, "spec", "labels", params, only={"REC"})
    report = json.loads((tmp_path / "REC" / "report.json").read_text())

    assert report["resynthesis_max_abs"] < 1e-6
    # Regime 3 never ran, so the residual IS the broadband channel.
    assert "stochastic" not in report
    with np.load(tmp_path / "REC" / "residual.npz", allow_pickle=False) as data:
        assert "stochastic" not in data

    v4 = report["v4"]
    assert v4["n_windows"] == 1
    assert v4["b_f_hz"] > 0.0
    assert v4["n_lines"] == 2 * K_MAX
    assert 0.0 <= v4["h_fit"]["low_start_frac"] <= 1.0
    # The pooled objective carries the v4 terms beside the v3 ones.
    assert "total_v4" in report["objective"]["pooled"]
    assert "total_v4" in report["objective"]["per_cell"]
    assert "total" in report["objective"]["pooled"]


#: Length of the twin-rig window. Short, because the fallback test needs a
#: 120-track coupled group and that group's banded system is the memory.
TWIN_S = 2.0


def _twin_recording(tmp_path, k_max: int, spread: float = 0.3, **over):
    """A TWIN rig at SPIN-UP: four rotors within a rev/s of each other, crossing.

    DREGON's own geometry, where the pairs sit 0.43 and 0.81 rev/s apart. Every
    harmonic pair is then inside one linewidth at the v4 bands, so the uncapped
    system is singular and the worker has to fall back. The rates are the only
    thing that matters here — the audio just has to have a comb in it.
    """
    from utils.gridrun import Unit

    rng = np.random.default_rng(0)
    n_t = int(TWIN_S * SR)
    t = np.arange(n_t) / SR
    off = np.array([-1.5, -0.5, 0.5, 1.5]) * spread
    rates = np.stack([42.0 + 6.0 * t / TWIN_S + o * (1.0 - 2.0 * t / TWIN_S) for o in off])
    phase = 2.0 * np.pi * np.cumsum(rates, axis=-1) / SR
    audio = rng.normal(scale=0.001, size=(2, n_t))
    for r in range(len(off)):
        for k in range(1, k_max + 1):
            for c, gain in enumerate((1.0, 0.8)):
                audio[c] += gain * (1.0 / k**0.5) * np.cos(k * phase[r] + 2 * np.pi * rng.random())

    ft = V.frame_grid(n_t, SR)
    V.cache_recordings(
        [
            {
                "recording_id": "TWIN",
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
    stride = int(round(SR / 100.0))
    params = {
        "recording": "TWIN",
        "i0": 0,
        "i1": int(ft.size),
        "spec": "spec",
        "label_dir": "labels",
        "out": str(tmp_path),
        "kind": "window",
        "window_s": TWIN_S,
        "hop_s": TWIN_S,
        "k_max": k_max,
        "mics": 2,
        "sr": SR,
        "f_max": 3000.0,
        "bw_rps": 1.0,
        "bw_schedule": "3,0,1.5,3",
        "ref_mic": -1,
        "mem_budget_gb": 0.0,
        "fs_env": 100.0,
        "stride": stride,
        "joint": True,
        "v4": True,
        "iters": 1,
        "k_trust": "3",
        "bw_psi": "0.6,8,1.5",
        "bw_theta": 1.5,
        "whiten": True,
        "stochastic": False,
        **over,
    }
    row = V.solve_worker(Unit("w0", params))
    (tmp_path / "raw" / "w0.json").write_text(json.dumps(row))
    return row, params


def test_v4_falls_back_to_the_capped_bands_on_a_twin_rig(tmp_path, capsys) -> None:
    """A window the UNCAPPED law cannot carry still produces a decomposition.

    Two rotors half a rev/s apart put every harmonic pair inside one linewidth,
    so the v4 bands make a singular system — DREGON's own geometry, where six of
    seven windows failed. What the model estimates is `(S, H)`, and that comes
    from the F1 fit, which never looks at a band; so the right answer is to keep
    the fit, the prior and `J_v4`, and take the bands from the schedule.
    """
    row, _ = _twin_recording(tmp_path, k_max=30)
    with capsys.disabled():
        print(
            f"\n  twin rig (4 rotors, 0.3 rev/s spread, 120 tracks): v4_band_fallback={row.get('v4_band_fallback')}"
            f"  band_law={row.get('v4_band_law')}"
            f"  bw_track={row['bw_track_hz_by_band']}"
        )
        print(f"    reason: {str(row.get('v4_band_fallback_reason'))[:88]}...")
    assert row["used"] is True
    assert row["v4_band_fallback"] is True
    assert row["v4_band_law"] is False
    assert "non positive definite" in row["v4_band_fallback_reason"]
    # Everything else about the window is still v4: the joint fit ran and the
    # marginal objective was read.
    assert row["joint"]["config"]["v4"] is True
    assert row["joint"]["h_fit"]["n_lines"] == 4 * 30
    assert "total_v4" in row["objective"]
    with np.load(tmp_path / "raw" / "w0.npz", allow_pickle=False) as data:
        assert "h_power" in data


def test_a_window_that_needs_no_fallback_does_not_take_one(tmp_path) -> None:
    """The flag is a record of what happened, not a mode the run is put in."""
    row, _ = _v4_unit(tmp_path)
    assert row.get("v4_band_fallback") is None
    assert row["v4_band_law"] is True


def test_mixed_band_laws_stitch_and_the_report_says_so(tmp_path, capsys) -> None:
    """Mixed windows stitch: the bands may differ, the TRACK SET may not.

    The stitch's one compatibility check is the harmonic set, and `k_hi` comes
    from the recording's reference trajectory rather than from any window's
    bands — so a recording whose windows used different band laws stitches
    exactly as one whose windows did not. The report has to SAY the set is
    mixed, because `bw_track_hz_by_band` is read off the first window alone.
    """
    fell, params = _twin_recording(tmp_path, k_max=30)
    assert fell["v4_band_fallback"] is True
    # A second window of the same recording, solved with the band law forced
    # off-limits-free: same track set, different bands.
    row2 = dict(fell)
    row2["v4_band_fallback"] = False
    row2["v4_band_law"] = True
    row2["a0"] = 0
    (tmp_path / "raw" / "w0.json").write_text(json.dumps(fell))
    (tmp_path / "raw" / "w1.json").write_text(json.dumps({**row2, "npz": fell["npz"]}))

    V.stitch(tmp_path, "spec", "labels", params, only={"TWIN"})
    report = json.loads((tmp_path / "TWIN" / "report.json").read_text())
    with capsys.disabled():
        print(
            f"\n  mixed stitch: n_band_fallback={report['v4']['n_band_fallback']}"
            f"  band_law_mixed={report['v4']['band_law_mixed']}"
            f"  resynthesis={report['resynthesis_max_abs']:.2e}"
        )
    assert report["v4"]["n_band_fallback"] == 1
    assert report["v4"]["band_law_mixed"] is True
    assert report["resynthesis_max_abs"] < 1e-6


def test_v4_is_off_by_default_and_writes_no_line_table(tmp_path) -> None:
    """The switch is a switch: without it the unit is the v3 unit, key for key."""
    row, _ = _v4_unit(tmp_path, v4=False)
    assert row["joint"]["config"]["v4"] is False
    assert "h_fit" not in row["joint"]
    assert not [key for key in row["objective"] if key.endswith("_v4")]
    with np.load(tmp_path / "raw" / "w0.npz", allow_pickle=False) as data:
        assert "h_power" not in data
