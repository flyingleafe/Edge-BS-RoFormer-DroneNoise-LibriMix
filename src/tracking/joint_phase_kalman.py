"""Joint four-rotor phase-increment Kalman filter.

WHY THIS EXISTS. `phase_increment_tracker.pi_kalman_refine` estimates one
rotor at a time. Each rotor gets a scalar state, and every measurement is
attributed to that one rotor. When a harmonic of rotor `i` shares a demod band
with a harmonic of rotor `j`, that attribution is false: the band holds a sum
of two phasors, whose argument advances at neither line's rate. The sequential
filter can only DISCARD such a measurement, which is what its twin gate does.

Measured cost of the sequential design on an exact initialization of a static
comb: it injects 0.0160 rev/s with four rotors and 0.0006 rev/s with one. The
gate is not the cure — a wider guard makes the four-rotor number worse (0.0192
at 20 Hz), because gating removes information without removing the bias.

THE JOINT MODEL. Keep the collided measurements and attribute them correctly.
The state becomes the full correction vector `dr = (dr_1 ... dr_R)`. For a
demod band that holds lines `m` with powers `P_m`, a first-order expansion of
the argument of the phasor sum gives

    dpsi ~= sum_m w_m * 2 pi k_m dt * dr_rot(m),    w_m = P_m / sum P

— the power-weighted mean of the individual increments. This is the general
form of the two-phasor winding-number rule: the sum advances at the stronger
component's rate. The observation row is therefore DENSE in `dr` instead of
sparse, and a clean band (one line, `w = 1`) gives back the sequential filter's
own row exactly. Nothing is discarded, and nothing is misattributed.

THE BEAT IS NOISE, NOT BIAS. Two lines in one band make the instantaneous
increment swing about that weighted mean at the beat rate. The mean is what the
row models, so the swing enters as extra variance. The data-driven excess pass
already in the sequential design measures exactly such excess, so this module
runs it SEPARATELY over collided and clean frames of each band. A collided band
then down-weights itself by the amount its own beat justifies.

WHAT IS REUSED. Demodulation, the noise probe, the SNR and wrap gates, the
variance model and its band corrections all come from
`phase_increment_tracker`. Only the state, the observation rows and the
smoother recursions are new. The purity rule of `src/tracking` holds: numpy and
scipy only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .phase_increment_tracker import (
    _MAX_CHANNELS,
    _TINY,
    _abs2,
    _band_corrections,
    _band_corrections_k,
    _increment_phase,
    _seam,
    demod_bank,
    thread_pool,
)

__all__ = ["joint_pi_kalman_refine"]


# ---------------------------------------------------------------------------
# the matrix smoother


def _mat_kalman_rts(
    info: np.ndarray, mean_info: np.ndarray, q: np.ndarray, p0: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vector random-walk Kalman filter and RTS smoother, information-fed.

    `info[t]` is the `(R, R)` information matrix of the measurements on frame
    `t`, and `mean_info[t]` the matching `(R,)` information vector. `q` is the
    per-step process covariance and `p0` the prior covariance. Returns the
    smoothed mean `(n, R)` and covariance `(n, R, R)`.

    This is the matrix form of `phase_increment_tracker._rw_kalman_rts`. A
    frame with no measurement has zero information, and its step becomes a pure
    prediction.
    """
    n, r = mean_info.shape
    m_f = np.zeros((n, r))
    p_f = np.zeros((n, r, r))
    m_p = np.zeros((n, r))
    p_p = np.zeros((n, r, r))
    m_prev = np.zeros(r)
    p_prev = np.array(p0, dtype=np.float64)
    for j in range(n):
        pp = p_prev + (q if j > 0 else 0.0)
        mp = m_prev
        m_p[j] = mp
        p_p[j] = pp
        pp_inv = np.linalg.inv(pp)
        pf = np.linalg.inv(pp_inv + info[j])
        mf = pf @ (pp_inv @ mp + mean_info[j])
        m_f[j] = mf
        p_f[j] = pf
        m_prev, p_prev = mf, pf
    m_s = m_f.copy()
    p_s = p_f.copy()
    for j in range(n - 2, -1, -1):
        a = p_f[j] @ np.linalg.inv(p_p[j + 1])
        m_s[j] = m_f[j] + a @ (m_s[j + 1] - m_p[j + 1])
        p_s[j] = p_f[j] + a @ (p_s[j + 1] - p_p[j + 1]) @ a.T
    return m_s, p_s


# ---------------------------------------------------------------------------
# the line table and its band membership


def _line_freqs(r: np.ndarray, ft: np.ndarray, t_mid: np.ndarray, lines: list[tuple[int, int]]):
    """Instantaneous line frequencies on the increment grid: `(L, n_m)` Hz."""
    r_mid = np.stack([np.interp(t_mid, ft, r[i]) for i in range(r.shape[0])])
    return np.stack([float(k) * r_mid[i] for i, k in lines])


def _membership(
    f_lines: np.ndarray, band_hz_line: np.ndarray, own: int
) -> np.ndarray:
    """Which lines sit inside band `own`: `(L, n_m)` bool, own line included.

    The band is the demod passband of its own line, so a line enters when its
    frequency is inside `+-band_hz` of the band centre. The centre moves with
    the estimate, so membership is evaluated per frame.
    """
    d = np.abs(f_lines - f_lines[own][None, :])
    inside = d < band_hz_line[own]
    inside[own] = True
    return inside


# ---------------------------------------------------------------------------
# one outer iteration


def _joint_pass(
    y32_of,
    t_aud: np.ndarray,
    r: np.ndarray,
    ft: np.ndarray,
    sr: int,
    stride: int,
    n_env: int,
    dt: float,
    t_mid: np.ndarray,
    k_cap: int,
    *,
    band_hz: float,
    off_comb_hz: float,
    f_max: float,
    snr_gate: float,
    wrap_guard_rad: float,
    n_trim: int,
    q_step: np.ndarray,
    p0: np.ndarray,
    min_rate: float,
    b0: np.ndarray | None,
    weight_mode: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """One joint outer iteration: `(delta (R, N) on the ft grid | None, diag)`."""
    n_rot = r.shape[0]
    n_m = n_env - 1
    fs_e = 1.0 / dt
    d: dict[str, Any] = {"k_cap": int(k_cap)}

    # --- which rotors carry a comb at all
    r_aud = np.stack([np.interp(t_aud, ft, r[i]) for i in range(n_rot)])
    mean_rate = r_aud.mean(axis=1)
    live = [i for i in range(n_rot) if mean_rate[i] >= min_rate]
    d["mean_rate"] = [round(float(v), 3) for v in mean_rate]
    if not live:
        d["skipped"] = "no rotor above min_rate"
        return None, d

    # --- demodulate every rotor's comb; keep every harmonic
    lines: list[tuple[int, int]] = []
    dpsi_l: list[np.ndarray] = []
    var_l: list[np.ndarray] = []
    valid_l: list[np.ndarray] = []
    a2_l: list[np.ndarray] = []
    band_l: list[float] = []
    ess_l: list[float] = []
    tmask = np.zeros(n_m, dtype=bool)
    tmask[n_trim : max(n_trim, n_m - n_trim)] = True

    for i in live:
        k_top = min(k_cap, int(np.floor(f_max / max(float(r_aud[i].max()), 1e-3))))
        if k_top < 1:
            continue
        ks = list(range(1, k_top + 1))
        ka = np.asarray(ks, dtype=np.float64)
        if b0 is None:
            band_k = np.full(k_top, float(band_hz))
            cn_k = np.full(k_top, _band_corrections(band_hz, dt)[0])
            cd_k = np.full(k_top, _band_corrections(band_hz, dt)[1])
        else:
            band_k = np.minimum(ka * float(b0[i]), 0.45 * fs_e)
            cn_k, cd_k = _band_corrections_k(band_k, dt)
        # The off-comb probe must clear its own band.
        off_k = np.maximum(off_comb_hz, band_k + max(band_k.max(), 1.0))
        phi = 2.0 * np.pi * np.cumsum(r_aud[i]) / sr
        z, z_off = demod_bank(
            y32_of(i), phi, t_aud, ks, off_comb_hz, stride, n_env,
            float(band_hz) / sr, band_k / sr, off_k, sr,
        )
        interior = slice(n_trim, max(n_trim + 1, n_env - n_trim))
        noise_pow = np.maximum(np.mean(_abs2(z_off[..., interior]), axis=-1), _TINY)  # (C, K)
        a2 = _abs2(z)
        dpsi = _increment_phase(z)  # (C, K, n_m)
        var = (
            cn_k[None, :, None]
            * 0.5
            * noise_pow[..., None]
            * (1.0 / np.maximum(a2[..., :-1], _TINY) + 1.0 / np.maximum(a2[..., 1:], _TINY))
        )
        ok = (a2[..., 1:] > snr_gate * noise_pow[..., None]) & (
            a2[..., :-1] > snr_gate * noise_pow[..., None]
        )
        ok &= np.abs(dpsi) < wrap_guard_rad
        ok &= tmask[None, None, :]
        for a, k in enumerate(ks):
            lines.append((i, k))
            dpsi_l.append(dpsi[:, a, :])
            var_l.append(var[:, a, :])
            valid_l.append(ok[:, a, :])
            a2_l.append(0.5 * (a2[:, a, :-1] + a2[:, a, 1:]))
            band_l.append(float(band_k[a]))
            ess_l.append(min(1.0, 2.0 * float(band_k[a]) * dt))

    if not lines:
        d["skipped"] = "no harmonic below f_max"
        return None, d

    n_l = len(lines)
    n_c = dpsi_l[0].shape[0]
    dpsi_a = np.stack(dpsi_l)  # (L, C, n_m)
    var_a = np.stack(var_l)
    valid_a = np.stack(valid_l)
    a2_a = np.stack(a2_l)
    band_a = np.asarray(band_l)
    ess_a = np.asarray(ess_l)
    rot_of = np.asarray([i for i, _ in lines])
    k_of = np.asarray([k for _, k in lines], dtype=np.float64)

    # --- band membership, per band and frame
    f_lines = _line_freqs(r, ft, t_mid, lines)  # (L, n_m)
    memb = np.stack([_membership(f_lines, band_a, b) for b in range(n_l)])  # (B, L, n_m)
    n_members = memb.sum(axis=1)  # (B, n_m)
    collided = n_members > 1
    d["collided_frac"] = round(float(collided[:, tmask].mean()), 4)
    d["n_lines"] = n_l

    # --- per-line power, read where the line is alone in its own band
    clean_own = ~collided  # (B, n_m); band b is clean exactly when its line is alone
    p_line = np.empty((n_l, n_c))
    n_clean = np.zeros(n_l, dtype=int)
    for b in range(n_l):
        sel = clean_own[b] & tmask
        n_clean[b] = int(sel.sum())
        src = a2_a[b][:, sel] if n_clean[b] >= 8 else a2_a[b][:, tmask]
        p_line[b] = np.median(src, axis=1) if src.shape[1] else _TINY
    p_line = np.maximum(p_line, _TINY)
    d["n_lines_never_clean"] = int((n_clean < 8).sum())

    # --- observation rows: (B, C, R, n_m)
    g = np.zeros((n_l, n_c, n_rot, n_m))
    h_line = 2.0 * np.pi * k_of * dt  # (L,)
    if weight_mode == "drop":
        # Control: gate collided measurements out, as the sequential twin gate
        # does. The rows are then all sparse, so the joint filter decouples into
        # the per-rotor scalar filters and must reproduce the sequential answer.
        valid_a &= ~collided[:, None, :]
    for b in range(n_l):
        idx = np.nonzero(memb[b].any(axis=1))[0]  # lines that ever enter band b
        if weight_mode == "drop" or idx.size == 1:
            g[b, :, rot_of[b], :] = h_line[b]
            continue
        inb = memb[b][idx].astype(np.float64)  # (Mi, n_m)
        w = p_line[idx].T[:, :, None] * inb[None, :, :]  # (C, Mi, n_m)
        if weight_mode == "hard":
            keep = w == w.max(axis=1, keepdims=True)
            w = np.where(keep, w, 0.0)
        w = w / np.maximum(w.sum(axis=1, keepdims=True), _TINY)
        contrib = w * h_line[idx][None, :, None]  # (C, Mi, n_m)
        for a, m in enumerate(idx):
            g[b, :, rot_of[m], :] += contrib[:, a, :]

    # --- pass A: no beat inflation, to measure the residual excess
    def _fuse(var_use: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = np.where(valid_a, ess_a[:, None, None] / np.maximum(var_use, _TINY), 0.0)
        info = np.einsum("bct,bcit,bcjt->tij", w, g, g, optimize=True)
        mean_info = np.einsum("bct,bct,bcit->ti", w, dpsi_a, g, optimize=True)
        return _mat_kalman_rts(info, mean_info, q_step, p0)

    m0, _ = _fuse(var_a)
    resid = dpsi_a - np.einsum("bcrt,tr->bct", g, m0)

    # --- excess variance, measured apart on collided and clean frames
    q_add = np.zeros((n_l, n_m))
    exc_rep: dict[str, list[float]] = {"clean": [], "collided": []}
    for b in range(n_l):
        for name, sel in (("clean", ~collided[b]), ("collided", collided[b])):
            m = valid_a[b] & sel[None, :]
            rr = resid[b][m]
            if rr.size < 16:
                continue
            var_rob = (1.4826 * float(np.median(np.abs(rr - np.median(rr))))) ** 2
            excess = max(0.0, var_rob - float(np.mean(var_a[b][m])))
            q_add[b, sel] = excess
            exc_rep[name].append(excess)
    for name, vals in exc_rep.items():
        if vals:
            d[f"excess_{name}_med"] = round(float(np.median(vals)), 6)

    m1, p1 = _fuse(var_a + q_add[:, None, :])

    d["delta_rms"] = [round(float(v), 4) for v in np.sqrt(np.mean(m1**2, axis=0))]
    p_diag = np.diagonal(p1, axis1=1, axis2=2)
    d["post_std_med"] = [round(float(v), 4) for v in np.median(np.sqrt(p_diag), axis=0)]
    d["n_meas"] = int(valid_a.sum())
    delta = np.stack([np.interp(ft, t_mid, m1[:, i]) for i in range(n_rot)])
    for i in range(n_rot):
        if i not in live:
            delta[i] = 0.0
    return delta, d


# ---------------------------------------------------------------------------
# public API


def joint_pi_kalman_refine(
    audio: np.ndarray,
    r_init: np.ndarray,
    ft: np.ndarray,
    *,
    sr: int = 16000,
    n_iter: int = 6,
    fs_env: float = 100.0,
    band_hz: float = 6.0,
    off_comb_hz: float = 40.0,
    k_max: int = 60,
    f_max: float = 7500.0,
    k_caps: tuple[int, ...] = (12, 24, 40, 60),
    sigma_process: float = 0.5,
    sigma_prior: float = 2.0,
    snr_gate: float = 2.0,
    wrap_guard_rad: float = 1.2,
    max_step: float = 3.0,
    edge_trim_s: float = 0.15,
    min_rate: float = 5.0,
    band_mode: str = "k_scaled",
    band_b0: float = 0.15,
    weight_mode: str = "power",
    peel_audio=None,
    threads: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Refine every rotor's speed track jointly from phase increments.

    Drop-in alternative to `phase_increment_tracker.pi_kalman_refine` with one
    change of model: the state is the whole correction vector, so a demod band
    that holds harmonics of several rotors contributes ONE dense observation
    row instead of being discarded by a twin gate.

    Args:
        audio: `(C, N)` or `(N,)` waveform.
        r_init: `(R, N_frames)` initial rotor speeds in rev/s.
        ft: `(N_frames,)` frame times in seconds.
        n_iter: outer iterations. Each one re-demodulates at the current
            estimate, so the capture range applies per iteration.
        band_mode: `"k_scaled"` gives harmonic `k` a band of `k * band_b0`
            rev/s, which makes the capture range about constant in rev/s.
            `"fixed"` uses `band_hz` for every harmonic.
        weight_mode: `"power"` splits a collided band's row between its lines
            by power, which is the model above. `"hard"` gives the whole row to
            the strongest line — the control that shows what the soft split
            buys. `"drop"` gates collided measurements out, which decouples the
            filter and reproduces the sequential answer.
        peel_audio: optional `{rotor: audio}` seam, as in the sequential
            refiner.

    Returns:
        `(r_refined, diagnostics)`.
    """
    x = np.atleast_2d(np.asarray(audio, dtype=np.float64))[:_MAX_CHANNELS]
    r = np.array(r_init, dtype=np.float64, copy=True)
    if r.ndim != 2:
        raise ValueError(f"r_init must be (R, N), got shape {r.shape}")
    ft = np.asarray(ft, dtype=np.float64)
    if r.shape[-1] != len(ft):
        raise ValueError(f"r_init has {r.shape[-1]} frames, ft has {len(ft)}")
    if band_mode not in ("fixed", "k_scaled"):
        raise ValueError(f"unknown band_mode {band_mode!r}")
    if weight_mode not in ("power", "hard", "drop"):
        raise ValueError(f"unknown weight_mode {weight_mode!r}")

    n_rot = r.shape[0]
    n_t = x.shape[-1]
    t_aud = np.arange(n_t) / sr
    stride = max(1, int(round(sr / fs_env)))
    dt = stride / sr
    n_env = len(range(0, n_t, stride))
    if n_env < 8:
        raise ValueError(f"clip too short: {n_env} envelope frames")
    t_env = np.arange(n_env) * dt
    t_mid = t_env[:-1] + 0.5 * dt
    y32 = x.astype(np.float32)
    n_trim = max(1, int(round(edge_trim_s / dt)))
    q_step = (sigma_process**2 * dt) * np.eye(n_rot)
    p0 = (sigma_prior**2) * np.eye(n_rot)
    schedule = [int(min(k_caps[min(j, len(k_caps) - 1)], k_max)) for j in range(n_iter)]
    b0 = np.full(n_rot, float(band_b0)) if band_mode == "k_scaled" else None

    iters: list[dict[str, Any]] = []
    with thread_pool(threads):
        for it, k_cap in enumerate(schedule):
            delta, d = _joint_pass(
                lambda i: _seam(y32, peel_audio, i),
                t_aud, r, ft, sr, stride, n_env, dt, t_mid, k_cap,
                band_hz=band_hz,
                off_comb_hz=off_comb_hz,
                f_max=f_max,
                snr_gate=snr_gate,
                wrap_guard_rad=wrap_guard_rad,
                n_trim=n_trim,
                q_step=q_step,
                p0=p0,
                min_rate=min_rate,
                b0=b0,
                weight_mode=weight_mode,
            )
            d["iter"] = it + 1
            if delta is not None:
                step = np.clip(delta, -max_step, max_step)
                r += step
                d["step_rms"] = [round(float(v), 4) for v in np.sqrt(np.mean(step**2, axis=1))]
            iters.append(d)

    diagnostics = {
        "params": {
            "n_iter": n_iter, "fs_env": fs_env, "band_hz": band_hz,
            "off_comb_hz": off_comb_hz, "k_max": k_max, "f_max": f_max,
            "k_caps": list(k_caps), "sigma_process": sigma_process,
            "sigma_prior": sigma_prior, "snr_gate": snr_gate,
            "wrap_guard_rad": wrap_guard_rad, "max_step": max_step,
            "edge_trim_s": edge_trim_s, "min_rate": min_rate,
            "band_mode": band_mode, "band_b0": band_b0, "weight_mode": weight_mode,
        },
        "k_schedule": schedule,
        "iters": iters,
    }
    return r, diagnostics
