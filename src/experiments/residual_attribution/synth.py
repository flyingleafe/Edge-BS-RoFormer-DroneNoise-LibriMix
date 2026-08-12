"""Synthetic validation: render known per-rotor noises through the repo's own
propagation model and try to recover their PSDs.

The point of using ``models.generative.positional_harmonic_gen.propagate``
rather than a private re-implementation is that the *only* thing under test is
the estimator: if the estimator recovers ``P_r`` here, then it recovers exactly
what the generator's broadband branch would have to emit.

Ground truth is unambiguous because ``ref_distance = 1``: the source waveform
handed to ``propagate`` *is* the rotor's signal at 1 m, so its Welch PSD is the
``P_r(f)`` the fit is supposed to return.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .csd import welch_csd
from .design import index_plan
from .fit import fit_offdiag
from .steering import steering

__all__ = ["SynthResult", "make_sources", "render", "run_case"]


def make_sources(
    n_rotor: int,
    n_samples: int,
    fs: float,
    *,
    levels_db: np.ndarray | None = None,
    slopes: np.ndarray | None = None,
    rng: np.random.Generator,
) -> np.ndarray:
    """``(R, T)`` independent shaped-noise rotor sources (at 1 m).

    Each rotor gets its own level and its own spectral tilt ``f^{-slope/2}`` in
    amplitude, so a per-frequency recovery error is visible as a shape error and
    not only as a level error.
    """
    levels = np.zeros(n_rotor) if levels_db is None else np.asarray(levels_db, dtype=np.float64)
    tilt = np.zeros(n_rotor) if slopes is None else np.asarray(slopes, dtype=np.float64)
    f = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    fnorm = np.maximum(f, f[1]) / 1000.0
    out = np.empty((n_rotor, n_samples))
    for r in range(n_rotor):
        w = rng.standard_normal(n_samples)
        spec = np.fft.rfft(w) * (fnorm ** (-tilt[r] / 2.0))
        y = np.fft.irfft(spec, n=n_samples)
        y *= 10.0 ** (levels[r] / 20.0) / max(y.std(), 1e-12)
        out[r] = y
    return out


def render(
    sources: np.ndarray,
    mic_pos: np.ndarray,
    rotor_pos: np.ndarray,
    fs: float,
) -> np.ndarray:
    """``(M, T)`` array signal, via the repo's ``propagate``."""
    import torch

    from models.generative.positional_harmonic_gen import propagate

    rel = np.asarray(mic_pos)[:, None, :] - np.asarray(rotor_pos)[None, :, :]  # (M, R, 3)
    src = torch.as_tensor(sources[None], dtype=torch.float64)  # (1, R, T)
    rp = torch.as_tensor(rel[None], dtype=torch.float64)  # (1, M, R, 3)
    y = propagate(src, rp, sample_rate=fs)  # (1, M, T)
    return y[0].numpy()


@dataclass
class SynthResult:
    freqs: np.ndarray
    p_true: np.ndarray  # (F, R)
    p_hat: np.ndarray  # (F, R)
    share_true: np.ndarray  # (R+1,)
    share_hat: np.ndarray  # (R+1,)
    diag_to_coh_db: float
    off_explained: np.ndarray


def run_case(
    mic_pos: np.ndarray,
    rotor_pos: np.ndarray,
    *,
    fs: float = 16000.0,
    duration_s: float = 30.0,
    diag_to_coh_db: float = 0.0,
    levels_db: np.ndarray | None = None,
    slopes: np.ndarray | None = None,
    nperseg: int = 4096,
    seed: int = 0,
) -> SynthResult:
    """One synthetic recovery experiment.

    ``diag_to_coh_db`` sets the level of the independent per-microphone noise
    relative to the total coherent (propagated) power at the array — the
    quantity that actually controls attribution difficulty, since the per-mic
    term is what hides the cross-spectra.
    """
    rng = np.random.default_rng(seed)
    n_rot = len(rotor_pos)
    n_t = int(round(duration_s * fs))
    sources = make_sources(n_rot, n_t, fs, levels_db=levels_db, slopes=slopes, rng=rng)
    coh = render(sources, mic_pos, rotor_pos, fs)

    coh_power = float((coh**2).mean())
    d_power = coh_power * 10.0 ** (diag_to_coh_db / 10.0)
    self_noise = rng.standard_normal(coh.shape) * np.sqrt(max(d_power, 0.0))
    x = coh + self_noise

    c = welch_csd(x, fs, nperseg=nperseg)
    g = steering(mic_pos, rotor_pos, c.freqs)
    plan = index_plan(x.shape[0])
    att = fit_offdiag(c.matrix(), g, plan)

    # ground-truth per-rotor PSD, same Welch convention
    c_src = welch_csd(sources, fs, nperseg=nperseg)
    p_true = np.real(np.einsum("fmm->fm", c_src.matrix()))  # (F, R)

    a_diag = np.abs(g) ** 2
    recv_true = p_true * a_diag.sum(axis=1)
    diag_true = np.full(len(c.freqs), d_power / (fs / 2.0)) * x.shape[0]
    st = np.concatenate([recv_true.sum(axis=0), [diag_true.sum()]])
    sh = np.concatenate([att.recv_rotor.sum(axis=0), [att.recv_diag.sum()]])
    return SynthResult(
        freqs=c.freqs,
        p_true=p_true,
        p_hat=att.p_rotor,
        share_true=st / st.sum(),
        share_hat=sh / sh.sum(),
        diag_to_coh_db=diag_to_coh_db,
        off_explained=att.off_explained,
    )
