#!/usr/bin/env python
"""Per-harmonic comb readout of trained noise generators against REAL DREGON audio.

The real-data counterpart of ``scripts/gen_label_sensitivity_eval.py`` (the
phase-7 synthetic readout). The aggregate multi-resolution STFT distance cannot
see a harmonic-comb loss — it is dominated by the broadband floor, and a
generator that drops every line above ``k = 50`` moves it very little — so a
label A/B needs an instrument that reads the comb tooth by tooth. This script is
that instrument.

Three readouts, plus the old scalar for continuity:

1. **Fidelity along the reference tracks** (``delta_logmag_db``). Comb-masked mean
   ``|Delta log-mag|`` between the generated and the recorded audio, sampled at
   the harmonic bins of the REFERENCE trajectory. One measuring stick for every
   arm; an arm is never scored on its own labels here.
2. **Line sharpness along the arm's own tracks** (``line_delta_db``). The
   floor-subtracted line power of the GENERATED audio along the trajectory the
   arm was conditioned on, against the same quantity for the RECORDING along the
   reference trajectory. Readout 1 alone cannot separate "the line is displaced"
   from "the line is washed out": both raise ``|Delta log-mag|``. This one can —
   a displaced line keeps its own power, a washed-out line does not.
3. **Peak-to-floor** (``ptf_db``), generated and real, per ``k`` band, plus their
   paired difference ``ptf_delta_db``. How far the tooth stands above the local
   floor, which is what the eye reads off a spectrogram.
4. **Aggregate mrstft** (``AuraMRSTFTMetric``), for continuity with the older
   generator tables.

The reference trajectory is the refined sidecar (``--refined-dir``) where one
exists, and the original telemetry otherwise. The fall-back is loud, because it
matters: on the original telemetry the recording's own peak-to-floor reaches the
estimator's null by ``k`` = 25, so readouts 2 and 3 have no head-room above it.

Every line reading goes through :func:`tracking.fitness.line_power` — a FIXED
band against a local floor, never a peak search. A peak-pick inside a window of
half-width ``W`` returns about ``W / 2`` on pure noise and has already withdrawn
two claims in this project (``docs/experiments/dregon-comb-displacement.md``).

Three numbers anchor the table, and none of them is 0: ``STOCHASTIC_FLOOR_DB``
(6.02 dB) is what readout 1 scores for a model that is perfect but random,
:func:`estimator_null_db` is where readout 3 sits when there is no line at all,
and ``ptf_delta_db`` is the paired difference in which both cancel.

Usage::

    python scripts/eval_gen_comb_real.py --self-test
    python scripts/eval_gen_comb_real.py --experiments gen_v1_recal_mm \\
        --out results/gen_comb_real --illustrate 1
    python scripts/eval_gen_comb_real.py --out results/gen_comb_real
    python scripts/eval_gen_comb_real.py --rigs dregon,michaels \\
        --split-filter valid,boundary --experiments gen_m3_refined_all_perrotor

The default is DREGON alone. ``--rigs dregon,michaels`` adds the Michael's rig
(FLY124/FLY125): its chunks carry ``meta.drone = "michaels"``, the michaels array
geometry and the arm's own michaels conditioning labels, so a per-drone codebook
generator is scored symmetrically on both rigs and every row says which rig it
came from.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

#: The label A/B arms. Each one loads its OWN Hydra experiment config, so the
#: label variant it trained with is discovered rather than assumed.
DEFAULT_ARMS = (
    "gen_r1_orig",
    "gen_r1_scaled",
    "gen_r1_refined",
    "gen_r2_orig_perrotor",
    "gen_r2_refined_perrotor",
)

#: The evaluation source, exactly as the training stream sees it
#: (``conf/data/noise_rps_dregon_stream_multimic.yaml``): the published
#: rich-frame DREGON dataset, the ``in_flight_noise`` split, measured motor
#: speeds. Only ``free-flight_nosource_room1`` logs ``motors_measured``, so this
#: selection resolves to one 64 s 8-microphone recording.
DREGON_FRAMES = "frames:DREGON-frames"
DREGON_RPS_KEY = "motors_measured"
DREGON_SPLITS = ["in_flight_noise"]

#: The second rig, exactly as the swapped DREGON+Michael's stream sees it
#: (``conf/data/noise_rps_dregon_michaels_swapped_stream_multimic*.yaml``): the
#: published rich-frame Michael's dataset, every recording, the recalibrated
#: ``rps`` telemetry. Its refined sidecars (``FLY124.npz`` / ``FLY125.npz``)
#: share the DREGON sidecars' format and time reference, so one
#: ``rps_override_dir`` serves both rigs.
MICHAELS_FRAMES = "frames:michaels-frames"
MICHAELS_RPS_KEY = "rps"

#: The rigs this script can tile. ``meta.drone`` uses exactly these names, which
#: are also the per-drone codebook keys of a ``_CodebookConditionedNoiseGen``.
RIGS = ("dregon", "michaels")

#: The phase-7 constant correction of the measured DREGON telemetry bias. Only a
#: default for the self-test's sensitivity row — each arm's own scale comes from
#: its Hydra config.
PHASE7_SCALE = 0.99458

_R2_CKPT = "r2://ml-data/artifacts/{exp}/checkpoints/best.ckpt"

#: k bands of the summary table — the same convention as the phase-7 readout, so
#: the synthetic and the real numbers land in the same rows.
K_BANDS: dict[str, range] = {
    "k1-9": range(1, 10),
    "k10-24": range(10, 25),
    "k25-49": range(25, 50),
    "k50-80": range(50, 81),
}

# --- analysis geometry ------------------------------------------------------
#
# n_fft = 2048 at 16 kHz gives 7.8 Hz bins. DREGON cruises at 70-80 rev/s, so
# one tooth is 9-10 bins wide and harmonics stay separated to k = 80 (6.4 kHz,
# comfortably below Nyquist). Two competing errors set this value:
#
#   * too coarse and the floor annulus has nowhere to sit. The floor is read in
#     the valley at (k + 0.5) f0, which is 4-5 bins wide at n_fft = 2048 and
#     only 2 bins at n_fft = 1024.
#   * too fine and the window is long enough for the line to sweep. The measured
#     rotor acceleration on this recording is 1.4 rev/s^2 (median over 1 s
#     differences; shorter differences read the 0.269 rev/s tachometer lattice,
#     not the rotor), which moves harmonic 80 by 80 * 1.4 * 0.128 = 14 Hz over
#     the 128 ms window — inside the +/-15.6 Hz band read below.
#
# The same n_fft/hop as the phase-7 synthetic readout, deliberately: the two
# scripts are then the same instrument on synthetic and on real audio.
N_FFT = 2048
HOP = 512
SAMPLE_RATE = 16000

#: Line band half-width: 2 FFT bins, or 10 % of the tooth spacing where that is
#: wider (it is not, at DREGON cruise). Never a peak search inside it.
_BAND_BINS = 2.0
_BAND_FRAC_OF_F0 = 0.10
#: Floor slot: centred in the valley at (k + 0.5) f0 — the "off-comb" position
#: of ``tracking.fitness``'s own control, where no rotor line can exist.
_FLOOR_FRAC_OF_F0 = 0.22
_FLOOR_BINS = 3.5

# There is deliberately NO shifted "off-comb" control here, and the reason is
# geometric rather than a matter of taste. ``tracking.fitness`` can afford one
# because it works on demodulated envelopes, where DC and the rest of the band are
# far apart. In the raw STFT the assembly of one cell — a band of +/-0.21 f0 plus
# two floor slots centred +/-0.5 f0 away and 0.22 f0 wide — already spans
# +/-0.72 f0 of a 1.0 f0 tooth spacing, so there is nowhere to move it: shift the
# band half a tooth and the floor slots land ON the teeth. A multiplicative shift
# fails for a second reason — by the three-distance theorem, for ANY factor some
# k <= 80 lands within 1/81 of a tooth, and measuring that k reads a real line and
# calls it noise. The estimator's null is therefore computed directly instead, by
# `estimator_null_db`.

#: The value readout 1 takes when the model is PERFECT but stochastic. Two
#: independent draws from the same complex-Gaussian bin give a log power ratio
#: with a standard logistic distribution, whose mean absolute value is
#: ``2 ln 2`` nepers = 6.02 dB. So a mean ``|Delta log-mag|`` of 6 dB is a model
#: that has the right distribution everywhere, not a bad one; only the excess
#: above it is a fit error. Print it beside the metric, never subtract it — the
#: coherent part of the comb is NOT stochastic, so the true floor of a
#: partly-tonal band lies somewhere between 0 and this.
STOCHASTIC_FLOOR_DB = 20.0 * float(np.log10(2.0))

_EPS = 1e-12


# ---------------------------------------------------------------------------
# spectra and the line reading


def stft_power(x: np.ndarray, *, n_fft: int, hop: int, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed power spectrogram ``(mic, frame, freq)`` and its bin centres.

    Frequency is the LAST axis, which is what :func:`tracking.fitness.line_power`
    reduces over, so a whole microphone array is one call.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    n_frames = 1 + (arr.shape[-1] - n_fft) // hop
    if n_frames < 1:
        raise ValueError(f"signal of {arr.shape[-1]} samples is shorter than n_fft={n_fft}")
    win = np.hanning(n_fft)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = arr[:, idx] * win  # (mic, frame, n_fft)
    spec = np.fft.rfft(frames, axis=-1)
    return np.abs(spec) ** 2, np.fft.rfftfreq(n_fft, 1.0 / sr)


def frame_centres(n_frames: int, *, n_fft: int, hop: int) -> np.ndarray:
    """Sample index at the CENTRE of each analysis frame.

    Frame ``f`` spans ``[f * hop, f * hop + n_fft)``, so its rate is the one at
    its centre. Point-sampling at ``f * hop`` instead is a 64 ms lead, which at
    k = 80 displaces the read bin by more than the effect being measured.
    """
    return np.arange(n_frames) * hop + n_fft // 2


def rates_at_frames(rps: np.ndarray, n_frames: int, *, n_fft: int, hop: int) -> np.ndarray:
    """Audio-rate rotor speeds resampled onto the STFT frame grid — ``(rotor, frame)``."""
    centres = frame_centres(n_frames, n_fft=n_fft, hop=hop)
    return np.asarray(rps, dtype=np.float64)[:, np.clip(centres, 0, rps.shape[-1] - 1)]


def _nearest_foreign(centres: np.ndarray, foreign: np.ndarray) -> np.ndarray:
    """Distance from each of ``centres`` to the nearest entry of ``foreign`` (Hz)."""
    if foreign.size == 0:
        return np.full(centres.shape, np.inf)
    srt = np.sort(foreign)
    pos = np.searchsorted(srt, centres)
    lo = srt[np.clip(pos - 1, 0, srt.size - 1)]
    hi = srt[np.clip(pos, 0, srt.size - 1)]
    return np.minimum(np.abs(centres - lo), np.abs(centres - hi))


def admit_cells(
    rates: np.ndarray, ks: np.ndarray, *, sr: int, df: float, gate_frac: float
) -> np.ndarray:
    """Which ``(rotor, k)`` cells may be read at one time frame — ``(rotor, k)`` bool.

    Only two conditions by default: the line band must sit above DC and the floor
    slot must sit below Nyquist.

    **Why there is no interference gate here by default.** ``tracking.fitness``
    offers two ways to protect a contested cell — drop it, or excise the intruder
    from the FLOOR region — and on a four-rotor drone only the second is usable.
    DREGON flies two near-equal rotor pairs (0.56 and 0.64 rev/s apart), so a twin's
    harmonic ``k`` sits within one FFT bin of its partner's for every ``k`` below
    about 25; requiring a clean band there admitted 2 % of the cells and left the
    per-``k`` curves reading single accidental cells. :func:`line_table` therefore
    excises foreign lines from the floor slot instead, and the band is left alone —
    which is sound here because the measurement geometry is pinned to the reference
    trajectory, so an intruder lands in the generated and the recorded reading
    alike and cancels in their difference.

    ``gate_frac > 0`` re-enables the strict rule (drop a cell whose band centre is
    within ``gate_frac * f0`` of a foreign harmonic) for anyone who wants it.
    """
    n_rot = rates.shape[0]
    carriers = np.outer(rates, ks)  # (rotor, k)
    nyq = sr / 2.0
    admit = np.zeros(carriers.shape, dtype=bool)
    for j in range(n_rot):
        f0 = float(rates[j])
        if f0 <= 0.0:
            continue
        half_bw = max(_BAND_BINS * df, _BAND_FRAC_OF_F0 * f0)
        floor_hw = min(_FLOOR_FRAC_OF_F0 * f0, _FLOOR_BINS * df)
        cen = carriers[j]
        ok = (cen > half_bw) & (cen + 0.5 * f0 + floor_hw < 0.98 * nyq)
        if gate_frac > 0.0:
            others = np.delete(carriers, j, axis=0).ravel()
            ok &= _nearest_foreign(cen, others) > gate_frac * f0
        admit[j] = ok
    return admit


def clean_floor_mask(
    sub_freqs: np.ndarray,
    centre: float,
    half_width: float,
    foreign: np.ndarray,
    exclude_hw: float,
) -> np.ndarray:
    """The floor slot at ``centre``, minus every foreign rotor line inside it.

    ``tracking.fitness.line_masks``'s ``exclude`` argument in the geometry this
    script needs (a valley slot, not an annulus): "the sibling's line is removed
    from the FLOOR region rather than the cell from the report".
    """
    mask = np.abs(sub_freqs - centre) <= half_width
    if not mask.any() or foreign.size == 0:
        return mask
    lo, hi = centre - half_width - exclude_hw, centre + half_width + exclude_hw
    i0, i1 = np.searchsorted(foreign, lo), np.searchsorted(foreign, hi, side="right")
    for line in foreign[i0:i1]:
        mask &= np.abs(sub_freqs - line) > exclude_hw
    return mask


@dataclass
class LineTable:
    """Accumulated line and floor power per harmonic, folded over mics as samples.

    Every value is a ``(mic,)`` mean over the admitted ``(rotor, frame)`` cells, of
    which ``count[k]`` were read:

    * ``line[k]`` — the phase-7 reading: floor-subtracted band power, each bin
      clipped at zero (``LinePower.total``). An amplitude, so a negative excursion
      is not one.
    * ``band[k]`` — the band power with nothing subtracted and nothing clipped.
    * ``floor[k]`` — the floor's own contribution across the same band.

    ``line_db`` reports the first; ``ptf_db`` is ``band / floor``, the ratio the
    unclipped pair is for. Reading the ratio off ``line`` instead would put its null
    near -2.3 dB rather than near 0, because the clip throws away half of a noise
    band. Even ``band / floor`` is not exactly 0 on noise — see
    :func:`estimator_null_db`, which measures where it actually lands.
    """

    line: dict[int, np.ndarray] = field(default_factory=dict)
    band: dict[int, np.ndarray] = field(default_factory=dict)
    floor: dict[int, np.ndarray] = field(default_factory=dict)
    count: dict[int, int] = field(default_factory=dict)
    #: Mean number of bins in the line band / the floor slots, per k. Only
    #: :func:`estimator_null_db` uses them, and it needs them because the null of a
    #: median-based floor depends on how many bins the median was taken over.
    n_band_bins: dict[int, float] = field(default_factory=dict)
    n_floor_bins: dict[int, float] = field(default_factory=dict)

    def line_db(self, k: int) -> np.ndarray:
        return 10.0 * np.log10(np.maximum(self.line[k], _EPS))

    def ptf_db(self, k: int) -> np.ndarray:
        return 10.0 * np.log10(np.maximum(self.band[k], _EPS) / np.maximum(self.floor[k], _EPS))


def line_table(
    power: np.ndarray,
    freqs: np.ndarray,
    rates: np.ndarray,
    *,
    ks: np.ndarray,
    sr: int,
    stride: int,
    gate_frac: float,
    ref_rates: np.ndarray | None = None,
) -> LineTable:
    """Floor-subtracted line power per ``k``, read along ``rates``.

    ``power`` is ``(mic, frame, freq)``; ``rates`` is ``(rotor, frame)``. Every
    reading is one :func:`tracking.fitness.line_power` call over the whole
    microphone array at once (mics are leading axes there), inside a fixed band
    at ``k * f0`` with the floor taken from the valley at ``(k + 0.5) f0``.

    **Fixed degrees of freedom.** Only the band CENTRE follows ``rates``. The band
    width, the floor offset and the admission gate all come from ``ref_rates`` (the
    reference trajectory, defaulting to ``rates`` itself). This is the rule
    ``tracking.fitness`` is built around, and it is not optional here: an arm whose
    comb is displaced would otherwise be gated on its OWN carriers, get a different
    and easier cell set, and be compared against the recording over cells the
    recording was never read on. Measured on this recording, letting the gate move
    with the candidate turned a 0.542 % label scale into a spurious +3.6 dB line-power
    gain — the readout was measuring its own cell selection, not the audio.
    """
    from tracking.fitness import line_power

    ref = rates if ref_rates is None else ref_rates
    df = float(freqs[1] - freqs[0])
    n_mic = power.shape[0]
    nyq = sr / 2.0
    tab = LineTable()
    acc_line = {int(k): np.zeros(n_mic) for k in ks}
    acc_band = {int(k): np.zeros(n_mic) for k in ks}
    acc_floor = {int(k): np.zeros(n_mic) for k in ks}
    acc_n = {int(k): 0 for k in ks}
    acc_nb = {int(k): 0 for k in ks}
    acc_nf = {int(k): 0 for k in ks}

    for t in range(0, power.shape[1], stride):
        rates_t = rates[:, t]
        ref_t = ref[:, t]
        admit = admit_cells(ref_t, ks, sr=sr, df=df, gate_frac=gate_frac)
        all_carriers = np.outer(ref_t, ks)
        spec = power[:, t, :]  # (mic, freq)
        for j in range(rates.shape[0]):
            f0, f0_ref = float(rates_t[j]), float(ref_t[j])
            if f0 <= 0.0 or f0_ref <= 0.0:
                continue
            half_bw = max(_BAND_BINS * df, _BAND_FRAC_OF_F0 * f0_ref)
            floor_hw = min(_FLOOR_FRAC_OF_F0 * f0_ref, _FLOOR_BINS * df)
            # Foreign lines come from the REFERENCE carriers, so the two readings
            # of one cell excise exactly the same bins.
            foreign = np.sort(np.delete(all_carriers, j, axis=0).ravel())
            for i, k in enumerate(ks):
                if not admit[j, i]:
                    continue
                centre = float(k) * f0  # the CANDIDATE's line
                ref_centre = float(k) * f0_ref
                # The valleys hang off the REFERENCE tooth, not the candidate's.
                # Pinning them there is what makes the generated and the recorded
                # reading of a cell share one floor region bin for bin; letting
                # them follow the candidate lets a displaced comb shift its own
                # floor slot on or off an interloper, which changes the admitted
                # cell set and un-pairs the comparison.
                up, down = ref_centre + 0.5 * f0_ref, ref_centre - 0.5 * f0_ref
                if ref_centre <= half_bw or up + floor_hw >= nyq:
                    continue
                # A narrow slice around the band and both valley slots: line_power
                # only ever touches those bins, and 1025-bin masks per cell would
                # otherwise dominate the runtime.
                b0 = max(int(np.floor((min(down, centre) - 2.0 * floor_hw) / df)), 0)
                b1 = min(int(np.ceil((max(up, centre) + 2.0 * floor_hw) / df)) + 1, freqs.size)
                sub_f = freqs[b0:b1]
                band = np.abs(sub_f - centre) <= half_bw
                if not band.any():
                    continue
                # BOTH valleys, unioned — two fixed regions, each minus its known
                # interferers. Not "whichever is quieter" and not even "whichever
                # kept more bins": any per-cell choice biases the floor, and the
                # union simply doubles the sample, which is what a median floor
                # needs (its null moves from -0.55 dB at 5 bins to -0.32 dB at 10;
                # see estimator_null_db).
                ann = clean_floor_mask(sub_f, up, floor_hw, foreign, half_bw)
                if down - floor_hw > 0.0:
                    ann = ann | clean_floor_mask(sub_f, down, floor_hw, foreign, half_bw)
                if not ann.any():
                    continue
                lp = line_power(
                    spec[:, b0:b1],
                    sub_f,
                    centre,
                    half_bw,
                    masks=(band, ann),
                    # line_power's documented setting for an UNBIASED mean density:
                    # a periodogram bin is exponential, whose median is ln2 of its
                    # mean. Needed because the peak-to-floor here is a ratio.
                    floor_scale=float(np.log(2.0)),
                )
                acc_line[int(k)] += np.asarray(lp.total, dtype=np.float64)
                floor_band = np.asarray(lp.floor, dtype=np.float64) * lp.n_bins
                # The band power itself: raw is the band MINUS the floor, unclipped.
                acc_band[int(k)] += np.asarray(lp.raw, dtype=np.float64) + floor_band
                acc_floor[int(k)] += floor_band
                acc_n[int(k)] += 1
                acc_nb[int(k)] += int(band.sum())
                acc_nf[int(k)] += int(ann.sum())

    for k in acc_n:
        if acc_n[k] == 0:
            continue
        tab.line[k] = acc_line[k] / acc_n[k]
        tab.band[k] = acc_band[k] / acc_n[k]
        tab.floor[k] = acc_floor[k] / acc_n[k]
        tab.count[k] = acc_n[k]
        tab.n_band_bins[k] = acc_nb[k] / acc_n[k]
        tab.n_floor_bins[k] = acc_nf[k] / acc_n[k]
    return tab


def estimator_null_db(
    n_band: float, n_floor: float, *, trials: int = 20000, seed: int = 0
) -> float:
    """What :meth:`LineTable.ptf_db` reads on a band that holds NO line.

    The zero of the peak-to-floor axis, and it is not at 0 dB. A periodogram bin of
    line-free audio is exponential; the floor is the MEDIAN of ``n_floor`` such bins
    divided by ``ln 2``, which is the distribution's median-to-mean ratio and so is
    right only in the limit. For few bins the SAMPLE median sits above the
    distribution median (its expectation is 1.13x it at 5 bins, 1.08x at 10), which
    makes the floor too big and the null NEGATIVE: -0.55 dB at 5 floor bins,
    -0.32 dB at 10, -0.09 dB at 40. Monte-Carlo rather than a closed form, because
    that small-sample bias is exactly what is being measured.

    This is the estimator's own null, so it does NOT carry the recording's
    spectral slope across the cell. It is an anchor for reading the column, not a
    substitute for the paired ``ptf_gen - ptf_real`` difference, which cancels
    everything of this kind by construction.
    """
    nb, nf = max(int(round(n_band)), 1), max(int(round(n_floor)), 1)
    rng = np.random.default_rng(seed)
    band = rng.standard_exponential((trials, nb)).mean(axis=-1)
    floor = np.median(rng.standard_exponential((trials, nf)), axis=-1) / np.log(2.0)
    return float(10.0 * np.log10(np.mean(band) / np.mean(floor)))


def track_delta_db(
    gen_power: np.ndarray, real_power: np.ndarray, rates: np.ndarray, *, ks: np.ndarray, sr: int
) -> dict[int, np.ndarray]:
    """Comb-masked mean ``|Delta log-mag|`` per ``k``, per mic — the E6 measure.

    Both spectrograms are read at the SAME bin (the harmonic of the reference
    trajectory), so an interloper corrupts them equally and no gate is applied;
    this mirrors the phase-7 ``track_readout`` exactly.
    """
    n_bins_f = gen_power.shape[-1]
    s_g = 10.0 * np.log10(gen_power + _EPS)
    s_r = 10.0 * np.log10(real_power + _EPS)
    n_rot, n_frames = rates.shape
    t_all = np.tile(np.arange(n_frames), n_rot)
    out: dict[int, np.ndarray] = {}
    for k in ks:
        bins = np.rint(float(k) * rates * N_FFT / sr).astype(int).ravel()
        ok = (bins > 0) & (bins < n_bins_f)
        if not ok.any():
            continue
        out[int(k)] = np.abs(s_g[:, t_all[ok], bins[ok]] - s_r[:, t_all[ok], bins[ok]]).mean(
            axis=-1
        )
    return out


# ---------------------------------------------------------------------------
# evaluation chunks


@dataclass
class Chunk:
    """One deterministic 4 s tile of the recording, with every label variant."""

    index: int
    recording_id: str
    t_rel: float
    split: str
    mean_rps: float
    audio: np.ndarray  # (mic, time)
    labels: dict[str, np.ndarray]  # variant -> (rotor, time) at audio rate
    mic_pos: np.ndarray
    rotor_pos: np.ndarray
    sample_rate: int
    #: Which rig the tile came from — the ``meta.drone`` key the generator is
    #: conditioned on, and the geometry ``mic_pos``/``rotor_pos`` belong to.
    rig: str = "dregon"

    def label(self, variant: str, scale: float) -> np.ndarray:
        """The conditioning track of one arm.

        ``scaled`` is not stored separately: :func:`apply_rps_scale` multiplies the
        telemetry VALUES and the upsampling to audio rate is linear, so scaling
        commutes with it and ``orig * scale`` is exact.
        """
        if variant == "refined":
            if "refined" not in self.labels:
                raise KeyError("this chunk has no refined labels (sidecars absent)")
            return self.labels["refined"]
        return self.labels["orig"] * float(scale)


def _slice_chunk(
    src: Any, t_abs: float, seconds: float, sample_rate: int, rps_key: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """``(audio (mic, T), rps (rotor, T))`` for one tile, or ``None`` if short."""
    from data_processing.noise_rps_dataset import upsample_rps_to_audio_rate

    sliced = src.frame.time[t_abs : t_abs + seconds]
    audio_s = sliced["audio"]
    audio = np.asarray(audio_s.data, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[None, :]
    audio_ts = np.asarray(cast(Any, audio_s.tindex).sample_times(), dtype=np.float64)
    n = int(round(seconds * sample_rate))
    if audio.shape[-1] < n:
        return None
    audio, audio_ts = audio[..., :n], audio_ts[:n]
    motor = sliced[rps_key]
    motor_ts = np.asarray(cast(Any, motor.tindex).abs_stamps, dtype=np.float64)
    values = np.asarray(motor.data, dtype=np.float32)
    if motor_ts.size < 2 or values.shape[-1] < 2:
        return None
    return audio, upsample_rps_to_audio_rate(values, motor_ts, audio_ts)


def _rig_spec(rig: str) -> tuple[str, str, list[str] | None]:
    """``(frames spec, rps key, splits)`` — the training stream's own selection."""
    if rig == "dregon":
        return DREGON_FRAMES, DREGON_RPS_KEY, DREGON_SPLITS
    if rig == "michaels":
        return MICHAELS_FRAMES, MICHAELS_RPS_KEY, None
    raise ValueError(f"unknown rig {rig!r}; expected one of {RIGS}")


def build_chunks(
    *,
    sample_rate: int = SAMPLE_RATE,
    seconds: float = 4.0,
    hop_seconds: float = 4.0,
    min_flight_rps: float = 45.0,
    val_pct: float = 0.1,
    override_dir: str | Path | None,
    max_chunks: int | None = None,
    rigs: tuple[str, ...] | list[str] = ("dregon",),
    keep_splits: set[str] | None = None,
) -> list[Chunk]:
    """Tile the in-flight recording(s) of each rig deterministically.

    Same sources, same trim and same split convention as the training stream
    (``val_at_start: true``, ``val_pct: 0.1``), but a fixed tiling instead of the
    loader's random draws — a comb readout has to score the same audio for every
    arm. Chunks straddling the split boundary are tagged ``boundary`` and kept:
    dropping them would silently change the eval set with the tile phase.

    ``rigs`` defaults to DREGON alone, which is the historical behaviour. With
    ``michaels`` in the list the Michael's rig is tiled the same way and its
    chunks carry ``rig = "michaels"``, so the generator is conditioned on the
    michaels codebook entry and the michaels geometry. ``max_chunks`` caps each
    RECORDING separately — a rig-wide cap would spend the whole budget on the
    first Michael's flight and never reach the second — and it counts only the
    chunks ``keep_splits`` admits, because capping before the split filter would
    fill the budget with chunks that are then thrown away. On a one-recording
    selection (the DREGON default) the cap is the historical one.
    """
    from data_processing.frame_datasets import _noise_gen_geometry  # geometry of the stream
    from data_processing.frames import get_meta
    from data_processing.noise_rps_dataset import load_published_noise_sources

    chunks: list[Chunk] = []
    for rig in rigs:
        spec, rps_key, splits = _rig_spec(rig)
        base = load_published_noise_sources(
            spec,
            sample_rate,
            origin=rig,
            rps_key=rps_key,
            splits=splits,
        )
        refined_srcs: list[Any] | None = None
        if override_dir is not None:
            refined_srcs = load_published_noise_sources(
                spec,
                sample_rate,
                origin=rig,
                rps_key=rps_key,
                splits=splits,
                rps_override_dir=override_dir,
            )
            if len(refined_srcs) != len(base):
                raise RuntimeError(
                    f"{rig}: refined load returned {len(refined_srcs)} recordings, "
                    f"original {len(base)}"
                )

        # `_noise_gen_geometry` reads its second argument for DREGON only; the
        # michaels array comes from the source registry.
        mic_pos_full, rotor_pos = _noise_gen_geometry(rig, spec if rig == "dregon" else None)
        for s_idx, src in enumerate(base):
            rec_id = str(get_meta(src.frame, "recording_id", f"rec{s_idx}") or f"rec{s_idx}")
            t0 = float(src.frame["audio"].t_start)
            cut = src.duration * val_pct  # val_at_start: the FIRST val_pct is validation
            n_rec = 0
            start = 0.0
            while start + seconds <= src.duration + 1e-9:
                got = _slice_chunk(src, t0 + start, seconds, sample_rate, rps_key)
                if got is None:
                    start += hop_seconds
                    continue
                audio, rps_orig = got
                mean_rps = float(rps_orig.mean())
                if mean_rps < min_flight_rps:
                    start += hop_seconds
                    continue
                labels = {"orig": rps_orig}
                if refined_srcs is not None:
                    ref = _slice_chunk(
                        refined_srcs[s_idx], t0 + start, seconds, sample_rate, rps_key
                    )
                    if ref is not None:
                        labels["refined"] = ref[1]
                if start + seconds <= cut:
                    split = "valid"
                elif start >= cut:
                    split = "train"
                else:
                    split = "boundary"
                if keep_splits is not None and split not in keep_splits:
                    start += hop_seconds
                    continue
                chunks.append(
                    Chunk(
                        index=len(chunks),
                        recording_id=rec_id,
                        t_rel=start,
                        split=split,
                        mean_rps=mean_rps,
                        audio=audio,
                        labels=labels,
                        mic_pos=np.asarray(mic_pos_full[: audio.shape[0]], dtype=np.float32),
                        rotor_pos=np.asarray(rotor_pos, dtype=np.float32),
                        sample_rate=sample_rate,
                        rig=rig,
                    )
                )
                n_rec += 1
                if max_chunks is not None and n_rec >= max_chunks:
                    break
                start += hop_seconds
    return chunks


# ---------------------------------------------------------------------------
# arms


@dataclass
class Arm:
    """One trained generator plus the label variant it was conditioned on."""

    name: str
    checkpoint: str
    label_variant: str  # "orig" | "scaled" | "refined" — the DREGON side
    label_scale: float
    codec: Any
    model: Any
    metric_suite: Any
    #: The same pair per rig. An arm can train on refined DREGON labels and
    #: published michaels telemetry, so conditioning is decided per chunk.
    label_by_rig: dict[str, tuple[str, float]] = field(default_factory=dict)

    def labels_for(self, rig: str) -> tuple[str, float]:
        """``(variant, scale)`` this arm was conditioned on for ``rig``."""
        return self.label_by_rig.get(rig, (self.label_variant, self.label_scale))


def resolve_checkpoint(exp: str, ckpt: str | None = None) -> str:
    """The arm's checkpoint: an explicit path, else local ``results/``, else R2."""
    if ckpt:
        return ckpt
    local = _ROOT / "results" / exp / "best.ckpt"
    return str(local) if local.is_file() else _R2_CKPT.format(exp=exp)


def _compose(exp: str, checkpoint: str):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from training.config import register_configs

    register_configs()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_ROOT / "conf"), version_base=None):
        return compose(
            config_name="config",
            overrides=[
                f"experiment={exp}",
                f"checkpoint={checkpoint}",
                "logging.enabled=false",
                "artifacts.enabled=false",
            ],
        )


def label_variant_of(cfg: Any, rig: str = "dregon") -> tuple[str, float]:
    """Which labels of ``rig`` this experiment trained on — read from its own config.

    Conditioning at eval must use the arm's training labels: a generator that
    learned to place its comb on biased telemetry is not being tested if it is
    then driven by a corrected trajectory.

    The michaels side has one knob only (``michaels_rps_override_dir``): there is
    no michaels counterpart of ``dregon_rps_scale``, so an arm without the
    override conditioned on the published, recalibrated telemetry.
    """
    params: Any = {}
    try:
        params = cfg.data.train.params
    except Exception:  # noqa: BLE001 - a non-noise-gen config simply has no knobs
        return "orig", 1.0
    if rig == "michaels":
        return ("refined", 1.0) if params.get("michaels_rps_override_dir", None) else ("orig", 1.0)
    override = params.get("dregon_rps_override_dir", None)
    scale = float(params.get("dregon_rps_scale", 1.0) or 1.0)
    if override is not None:
        return "refined", 1.0
    if scale != 1.0:
        return "scaled", scale
    return "orig", 1.0


def load_arm(exp: str, *, device: Any, checkpoint: str | None = None, cfg: Any = None) -> Arm:
    """Compose an experiment, build its model and warm-start it from its checkpoint.

    Pass ``cfg`` to reuse a config the caller already composed — the caller has to,
    because :func:`label_variant_of` decides whether an arm is scoreable at all and
    that answer is worth having before an R2 checkpoint is downloaded.
    """
    from training.config import build_metrics, build_task_and_codec, instantiate_model
    from training.loop import _warm_start

    ckpt = resolve_checkpoint(exp, checkpoint)
    cfg = _compose(exp, ckpt) if cfg is None else cfg
    variant, scale = label_variant_of(cfg)
    _task, codec = build_task_and_codec(cfg.model)
    model = instantiate_model(cfg.model).to(device)
    _warm_start(model, str(cfg.checkpoint), device)
    # eval() is the deterministic render the readout needs: the emitter samples
    # rotor-speed jitter and random initial phases only while training
    # (models.generative.harmonic_gen_new: `do_jitter = self.training`,
    # `initial_phases is None and self.training`), so a module in eval mode
    # renders the zero-phase, jitter-free comb.
    model.eval()
    return Arm(
        name=exp,
        checkpoint=ckpt,
        label_variant=variant,
        label_scale=scale,
        codec=codec,
        model=model,
        metric_suite=build_metrics(cfg.metrics),
        label_by_rig={r: label_variant_of(cfg, r) for r in RIGS},
    )


def chunk_frame(chunk: Chunk, rps: np.ndarray) -> Any:
    """The per-sample Frame the noise-generation codec consumes.

    Byte-for-byte the layout of ``NoiseGenFrameDataset.__getitem__`` — same entry
    names, same dims, same ``meta.drone`` key — so the codec resolves geometry and
    the conditioning code exactly as it did in training.
    """
    import tdseries as td

    sr = chunk.sample_rate
    return td.Frame(
        {
            "audio": td.uniform(chunk.audio, sr, dims=("mic", "time"), t_start=0.0),
            "rps": td.uniform(
                np.asarray(rps, dtype=np.float32), sr, dims=("rotor", "time"), t_start=0.0
            ),
            "mic_pos": td.wrap(chunk.mic_pos, dims=("mic", None)),
            "rotor_pos": td.wrap(chunk.rotor_pos, dims=("rotor", None)),
            "meta": td.Frame({"drone": chunk.rig}),
        }
    )


def render(arm: Arm, chunk: Chunk, *, device: Any, seed: int) -> np.ndarray:
    """Generated audio ``(mic, time)`` for one chunk, on the arm's own labels."""
    import torch

    from data_processing.collate import frame_collate
    from training.loop import _forward, _to_device

    rps = chunk.label(*arm.labels_for(chunk.rig))
    batch = _to_device(frame_collate([chunk_frame(chunk, rps)]), device)
    # The filtered-noise branch draws an excitation even in eval mode; a fixed
    # seed per chunk makes every arm see the same draw.
    torch.manual_seed(seed)
    with torch.no_grad():
        pred = _forward(arm.codec, arm.model, batch, device=device, amp=False)
    gen = pred.map_data(lambda t: t.detach().cpu())["audio"].data
    out = np.asarray(gen, dtype=np.float64)
    while out.ndim > 2:
        out = out[0]
    return out


def mrstft_of(arm: Arm, gen: np.ndarray, chunk: Chunk) -> float:
    """The old aggregate scalar, computed through the arm's own metric config."""
    import tdseries as td

    sr = chunk.sample_rate
    pred = td.Frame(
        {"audio": td.uniform(np.asarray(gen, dtype=np.float32), sr, dims=("mic", "time"))}
    )
    target = td.Frame(
        {"audio": td.uniform(np.asarray(chunk.audio, dtype=np.float32), sr, dims=("mic", "time"))}
    )
    agg = arm.metric_suite.evaluate([(pred, target)]).aggregate("mean")
    return float(agg.get("mrstft", float("nan")))


# ---------------------------------------------------------------------------
# scoring


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


@dataclass
class ChunkScore:
    """Everything measured for one ``(arm, chunk)`` pair, per ``k``."""

    delta_logmag: dict[int, np.ndarray]  # per mic
    line_gen: LineTable
    gain_db: float
    mrstft: float


def score_chunk(
    *,
    gen: np.ndarray,
    chunk: Chunk,
    real_power: np.ndarray,
    freqs: np.ndarray,
    ref_rates: np.ndarray,
    arm_rps: np.ndarray,
    ks: np.ndarray,
    stride: int,
    gate_frac: float,
    match_level: bool = True,
) -> ChunkScore:
    """Readouts 1-3 for one generated rendering against the recording."""
    gain = _rms(chunk.audio) / max(_rms(gen), 1e-20) if match_level else 1.0
    gen_power, _ = stft_power(gen * gain, n_fft=N_FFT, hop=HOP, sr=chunk.sample_rate)
    n_frames = gen_power.shape[1]
    delta = track_delta_db(
        gen_power, real_power, ref_rates[:, :n_frames], ks=ks, sr=chunk.sample_rate
    )
    own = rates_at_frames(arm_rps, n_frames, n_fft=N_FFT, hop=HOP)
    tab = line_table(
        gen_power,
        freqs,
        own,
        ks=ks,
        sr=chunk.sample_rate,
        stride=stride,
        gate_frac=gate_frac,
        # Band width, floor offset and gate stay on the REFERENCE trajectory, so
        # the generated and the real readings cover exactly the same cells.
        ref_rates=ref_rates[:, :n_frames],
    )
    return ChunkScore(
        delta_logmag=delta,
        line_gen=tab,
        gain_db=20.0 * np.log10(max(gain, 1e-20)),
        mrstft=float("nan"),
    )


def band_of(k: int) -> str | None:
    for name, ks in K_BANDS.items():
        if k in ks:
            return name
    return None


# ---------------------------------------------------------------------------
# output


_ROW_FIELDS = [
    "arm",
    "label_variant",
    "rig",
    "chunk",
    "recording_id",
    "t_rel",
    "split",
    "mean_rps",
    "k",
    "band",
    "delta_logmag_db",
    "delta_logmag_db_mic_std",
    "line_gen_db",
    "line_real_db",
    "line_delta_db",
    "line_delta_db_mic_std",
    "ptf_gen_db",
    "ptf_real_db",
    "ptf_delta_db",
    "n_cells_gen",
    "n_cells_real",
    "gain_db",
    "mrstft",
]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_ROW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-(arm, rig, band) means over every chunk, mic-folded reading already inside.

    The rig is part of the key because the two rigs are different drones with
    different rotor speeds and different arrays — averaging them together would
    hide exactly the per-rig difference this readout exists to show. A one-rig
    run therefore keeps its historical rows and only gains a ``rig`` column.
    """
    metric_cols = [
        "delta_logmag_db",
        "line_gen_db",
        "line_real_db",
        "line_delta_db",
        "ptf_gen_db",
        "ptf_real_db",
        "ptf_delta_db",
        "gain_db",
        "mrstft",
    ]
    groups: dict[tuple[str, Any, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["arm"], row.get("rig", ""), row["band"]), []).append(row)
    out: list[dict[str, Any]] = []
    for (arm, rig, band), items in groups.items():
        rec: dict[str, Any] = {
            "arm": arm,
            "label_variant": items[0]["label_variant"],
            "rig": rig,
            "band": band,
            "n_rows": len(items),
        }
        for col in metric_cols:
            vals = [v for v in (item[col] for item in items) if v is not None and np.isfinite(v)]
            rec[col] = float(np.mean(vals)) if vals else float("nan")
        admitted = [item["n_cells_gen"] for item in items]
        total = [item["n_cells_real"] for item in items]
        rec["mean_cells_gen"] = float(np.mean(admitted)) if admitted else 0.0
        rec["mean_cells_real"] = float(np.mean(total)) if total else 0.0
        out.append(rec)
    order = list(K_BANDS)
    out.sort(
        key=lambda r: (
            r["arm"],
            str(r["rig"]),
            order.index(r["band"]) if r["band"] in order else 99,
        )
    )
    return out


def write_summary(path: Path, summary: list[dict[str, Any]]) -> None:
    if not summary:
        return
    fields = list(summary[0])
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def _per_k_curves(
    rows: list[dict[str, Any]], column: str
) -> dict[str, tuple[list[int], list[float]]]:
    per_arm: dict[str, dict[int, list[float]]] = {}
    rigs = {str(row.get("rig", "")) for row in rows}
    for row in rows:
        val = row[column]
        if val is None or not np.isfinite(val):
            continue
        # One curve per (arm, rig) when both rigs are in the run: the two drones
        # have different comb geometries, so a merged curve is not a curve.
        name = row["arm"] if len(rigs) < 2 else f"{row['arm']} [{row['rig']}]"
        per_arm.setdefault(name, {}).setdefault(int(row["k"]), []).append(float(val))
    out: dict[str, tuple[list[int], list[float]]] = {}
    for arm, per_k in per_arm.items():
        ks = sorted(per_k)
        out[arm] = (ks, [float(np.mean(per_k[k])) for k in ks])
    return out


def plot_per_k(
    rows: list[dict[str, Any]], out_dir: Path, *, null_db: float | None = None
) -> list[Path]:
    """One overlay per metric: mean over chunks against harmonic index."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (column, y label, title, horizontal reference, reference label)
    panels = [
        (
            "delta_logmag_db",
            "mean |Δ log-mag| (dB)",
            "Fidelity along the reference tracks",
            STOCHASTIC_FLOOR_DB,
            "stochastic floor (a perfect but random model)",
        ),
        ("line_delta_db", "line power, generated − real (dB)", "Line sharpness", 0.0, "no error"),
        ("ptf_gen_db", "peak-to-floor (dB)", "Peak-to-floor, generated", None, ""),
        ("ptf_real_db", "peak-to-floor (dB)", "Peak-to-floor, recording", None, ""),
    ]
    rigs = sorted({str(row.get("rig", "")) or "dregon" for row in rows})
    source = "DREGON" if rigs == ["dregon"] else "/".join(rigs)
    written: list[Path] = []
    for column, ylabel, title, hline, hlabel in panels:
        curves = _per_k_curves(rows, column)
        if not curves:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for arm in sorted(curves):
            ks, vals = curves[arm]
            ax.plot(ks, vals, marker="o", ms=3, lw=1.2, label=arm)
        if column.startswith("ptf_") and null_db is not None:
            ax.axhline(null_db, color="k", lw=1.0, ls="--", label="estimator null (no line)")
        if hline is not None:
            ax.axhline(hline, color="k", lw=0.8, ls="--", label=hlabel or None)
        for edge in (10, 25, 50):
            ax.axvline(edge, color="0.7", lw=0.6, ls=":")
        ax.set_xlabel("harmonic index k")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} — real {source} audio")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / f"per_k_{column}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


#: The harmonic stack of interest reaches k = 80 at 6.4 kHz, so the illustration
#: crops higher than the repo's usual 4 kHz spectrogram convention — the whole
#: point is to SEE whether the high teeth are there.
SPEC_FMAX = 7000.0
#: Zoom columns: two 1.5-2 kHz windows over a 1 s slice. A full 4 s by 7 kHz panel
#: puts 90 teeth into a couple of hundred pixels and shows nothing; a 2 kHz window
#: holds about 26 teeth at DREGON cruise, which does resolve. The pair is chosen to
#: straddle the campaign's question: the low window covers roughly k = 4-24, where
#: the recording's teeth are measurable, and the high window k = 45-70, where the
#: peak-to-floor column says they are not.
SPEC_ZOOMS = ((300.0, 1800.0), (3500.0, 5500.0))
SPEC_ZOOM_SECONDS = 1.0


def _log_spectrogram(
    audio: np.ndarray, sr: int, *, fmin: float = 0.0, fmax: float = SPEC_FMAX
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    power, freqs = stft_power(audio[:1], n_fft=N_FFT, hop=HOP, sr=sr)
    s_db = 10.0 * np.log10(power[0].T + _EPS)  # (freq, frame)
    keep = (freqs >= fmin) & (freqs <= fmax)
    times = np.arange(s_db.shape[1]) * HOP / sr
    return s_db[keep], freqs[keep], times


def plot_illustration(
    real: np.ndarray, generated: dict[str, np.ndarray], chunk: Chunk, path: Path
) -> None:
    """Real vs generated spectrogram of one chunk, one row per arm.

    Two columns. The left one is the plain dB log-STFT over the whole band. The
    right one zooms into :data:`SPEC_ZOOM_HZ` and subtracts each frequency's own
    median over time — the visual form of the peak-to-floor statistic, and the only
    view in which a mid-``k`` tooth is actually legible against the broadband floor
    that dominates a drone recording.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(generated)
    n_rows = len(names) + 1
    n_cols = 1 + len(SPEC_ZOOMS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 2.6 * n_rows), squeeze=False)
    sr = chunk.sample_rate

    def _draw(
        ax: Any, audio: np.ndarray, label: str, bottom: bool, zoom: tuple[float, float] | None
    ) -> None:
        if zoom is not None:
            s_db, freqs, times = _log_spectrogram(audio, sr, fmin=zoom[0], fmax=zoom[1])
            n_t = max(1, int(SPEC_ZOOM_SECONDS * sr / HOP))
            s_db, times = s_db[:, :n_t], times[:n_t]
            # Two-way median removal, because two nuisances hide the teeth. Along
            # time: the floor of a drone recording rises 40 dB from 6 kHz to DC.
            # Along frequency: the broadband level swings several dB frame to
            # frame, which on its own paints the panel in vertical stripes. What
            # survives both is the tooth-against-valley contrast — the picture
            # the peak-to-floor column puts a number on.
            s_db = s_db - np.median(s_db, axis=1, keepdims=True)
            s_db = s_db - np.median(s_db, axis=0, keepdims=True)
            lo, hi = -4.0, 8.0
        else:
            s_db, freqs, times = _log_spectrogram(audio, sr)
            lo, hi = (float(np.percentile(s_db, 2.0)), float(np.percentile(s_db, 99.8)))
        ax.pcolormesh(times, freqs, s_db, cmap="magma", vmin=lo, vmax=hi, shading="auto")
        if zoom is None:
            ax.set_ylabel(f"{label}\nFreq (Hz)", fontsize=8)
        if bottom:
            ax.set_xlabel("Time (s)")

    for col, zoom in enumerate([None, *SPEC_ZOOMS]):
        _draw(axes[0][col], real, "REAL", n_rows == 1, zoom)
        for i, name in enumerate(names, start=1):
            _draw(axes[i][col], generated[name], name, i == n_rows - 1, zoom)
    axes[0][0].set_title(f"full band, 0–{SPEC_FMAX:.0f} Hz", fontsize=9)
    for col, zoom in enumerate(SPEC_ZOOMS, start=1):
        axes[0][col].set_title(
            f"{zoom[0]:.0f}–{zoom[1]:.0f} Hz, first {SPEC_ZOOM_SECONDS:g} s, "
            "per-frequency median removed",
            fontsize=9,
        )
    fig.suptitle(
        f"{chunk.recording_id} t={chunk.t_rel:.0f}s  (mean RPS {chunk.mean_rps:.0f}, "
        f"{chunk.split})  —  mic 0, rows: REAL then each arm"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# self-test


def frequency_scaled(audio: np.ndarray, scale: float) -> np.ndarray:
    """``y(t) = x(scale * t)`` — every frequency in ``audio`` multiplied by ``scale``.

    A comb displacement with no other change: the teeth move, their relative
    amplitudes and the broadband floor do not. Linear interpolation is enough
    because the probe only has to be a KNOWN displacement, not a clean resampler;
    its interpolation loss is the same at every ``k`` band being compared. The tail
    that would need samples past the end is dropped, so the result is a little
    shorter than the input for ``scale > 1``.
    """
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    n = arr.shape[-1]
    n_out = min(n, int(np.floor((n - 1) / max(scale, 1e-9))) + 1)
    src = np.arange(n_out) * float(scale)
    grid = np.arange(n)
    return np.stack([np.interp(src, grid, row) for row in arr], axis=0)


def _band_mean(per_k: dict[int, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, band_ks in K_BANDS.items():
        vals = [per_k[k] for k in band_ks if k in per_k and np.isfinite(per_k[k])]
        out[name] = float(np.mean(vals)) if vals else float("nan")
    return out


def self_test(chunks: list[Chunk], *, ks: np.ndarray, stride: int, gate_frac: float) -> int:
    """Score the recording against ITSELF, then against a knowingly wrong label set.

    Two halves, and the second is the one that matters.

    **Null.** The readout has to survive its own floor slot, its band width and its
    gate before any arm's number means anything, so the recording is pushed through
    as its own "generated" audio: every ``|Delta log-mag|`` must be 0 and every line
    power must be identical.

    **Sensitivity.** A null of zero is worthless on its own — a statistic that
    always reads zero passes it. So the recording is scored a second time against a
    DISPLACED copy of itself: the waveform resampled by the phase-7 constant
    0.99458, which moves every tooth by exactly that fraction and leaves everything
    else alone. That is the smallest comb error this campaign has to resolve, and it
    stands in for "the generator placed its comb 0.542 % off". Both readouts must
    move on it. Two claims in this project were withdrawn because their statistic
    turned out to have no power; a proven sensitivity row is the cheapest guard
    against a third.

    Note that readout 1 is exactly zero on the NULL by construction (both sides are
    read at the same bin), so the displaced copy is the only probe that can give it
    a scale.
    """
    reference = "refined" if all("refined" in c.labels for c in chunks) else "orig"
    worst_delta = 0.0
    worst_line = 0.0
    sens_line: dict[int, list[float]] = {}
    sens_delta: dict[int, list[float]] = {}
    ptf_on: dict[int, list[float]] = {}
    null_on: dict[int, float] = {}
    for chunk in chunks:
        real_power, freqs = stft_power(chunk.audio, n_fft=N_FFT, hop=HOP, sr=chunk.sample_rate)
        n_frames = real_power.shape[1]
        ref_rates = rates_at_frames(chunk.labels[reference], n_frames, n_fft=N_FFT, hop=HOP)
        args = {"ks": ks, "sr": chunk.sample_rate, "stride": stride, "gate_frac": gate_frac}
        base = line_table(real_power, freqs, ref_rates, **args)  # type: ignore[arg-type]
        same = line_table(real_power, freqs, ref_rates, **args)  # type: ignore[arg-type]
        null_delta = track_delta_db(real_power, real_power, ref_rates, ks=ks, sr=chunk.sample_rate)
        # The displaced copy: y(t) = x(a t) scales every frequency by `a`, so the
        # whole comb moves by the phase-7 constant and nothing else changes.
        alt_audio = frequency_scaled(chunk.audio, PHASE7_SCALE)
        alt_power, _ = stft_power(alt_audio, n_fft=N_FFT, hop=HOP, sr=chunk.sample_rate)
        alt_rates = ref_rates[:, : alt_power.shape[1]]
        alt = line_table(alt_power, freqs, alt_rates, **args)  # type: ignore[arg-type]
        alt_delta = track_delta_db(alt_power, real_power, alt_rates, ks=ks, sr=chunk.sample_rate)
        for k in (int(v) for v in ks):
            if k in null_delta:
                worst_delta = max(worst_delta, float(np.max(np.abs(null_delta[k]))))
            if k in base.line and k in same.line:
                worst_line = max(
                    worst_line, float(np.max(np.abs(base.line_db(k) - same.line_db(k))))
                )
            if k in base.band:
                ptf_on.setdefault(k, []).append(float(np.mean(base.ptf_db(k))))
                null_on[k] = estimator_null_db(base.n_band_bins[k], base.n_floor_bins[k])
            if k in alt.line and k in base.line:
                sens_line.setdefault(k, []).append(float(np.mean(alt.line_db(k) - base.line_db(k))))
            if k in alt_delta:
                sens_delta.setdefault(k, []).append(float(np.mean(alt_delta[k])))
    print(f"self-test on {len(chunks)} chunks, reference labels = {reference}")
    print(f"  NULL, worst-band |Delta log-mag| bias : {worst_delta:.6f} dB")
    print(f"  NULL, worst-band line-power bias      : {worst_line:.6f} dB")
    line_bands = _band_mean({k: float(np.mean(v)) for k, v in sens_line.items()})
    delta_bands = _band_mean({k: float(np.mean(v)) for k, v in sens_delta.items()})
    print(f"  SENSITIVITY, the same audio with its comb displaced by x{PHASE7_SCALE:g}:")
    for name in K_BANDS:
        print(
            f"    {name:<8} line power {line_bands[name]:+7.3f} dB, "
            f"|Delta log-mag| {delta_bands[name]:7.3f} dB"
        )
    on_bands = _band_mean({k: float(np.mean(v)) for k, v in ptf_on.items()})
    null_bands = _band_mean(null_on)
    print("  SCALE, the recording's own peak-to-floor against the estimator's null:")
    for name in K_BANDS:
        print(
            f"    {name:<8} recording {on_bands[name]:+7.3f} dB, "
            f"null {null_bands[name]:+7.3f} dB "
            f"(head-room {on_bands[name] - null_bands[name]:+7.3f} dB)"
        )
    moved = max((abs(v) for v in line_bands.values() if np.isfinite(v)), default=0.0)
    if moved < 1e-3:
        print("  FAIL: the readout does not move on a wrong label set", file=sys.stderr)
    return 0 if max(worst_delta, worst_line) <= 1e-6 and moved >= 1e-3 else 1


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiments", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--checkpoint", default=None, help="override the checkpoint for every arm")
    ap.add_argument("--out", type=Path, default=_ROOT / "results" / "gen_comb_real")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda when available)")
    ap.add_argument("--chunk-seconds", type=float, default=4.0)
    ap.add_argument("--hop-seconds", type=float, default=4.0)
    ap.add_argument("--min-flight-rps", type=float, default=45.0)
    ap.add_argument(
        "--max-chunks", type=int, default=None, help="cap the kept chunks of EACH recording"
    )
    ap.add_argument(
        "--rigs",
        default="dregon",
        help="comma list of rigs to tile: dregon, michaels, or dregon,michaels. "
        "Michael's chunks are conditioned with meta.drone='michaels' and the "
        "michaels array geometry",
    )
    ap.add_argument(
        "--split-filter",
        default=None,
        help="comma list of split tags to keep (train, valid, boundary). Default: keep every chunk",
    )
    ap.add_argument("--k-max", type=int, default=80)
    ap.add_argument(
        "--line-stride",
        type=int,
        default=4,
        help="read the line power every N STFT frames (the readings are highly "
        "correlated between adjacent frames; 4 keeps ~30 reads per 4 s chunk)",
    )
    ap.add_argument(
        "--gate",
        type=float,
        default=0.0,
        help="strict admission gate: drop a cell whose band centre is within this "
        "fraction of the tooth spacing of a foreign rotor harmonic. 0 (the default) "
        "keeps every cell and protects only the floor slot, by excising foreign "
        "lines from it — see admit_cells for why a band gate empties the set on a "
        "four-rotor drone with near-equal rotor pairs",
    )
    ap.add_argument(
        "--refined-dir",
        default="src/data_processing/refined_labels",
        help="folder of refined-label sidecars (<recording id>.npz)",
    )
    ap.add_argument(
        "--skip-refined-labels",
        action="store_true",
        help="score against the original telemetry tracks when no sidecars exist",
    )
    ap.add_argument(
        "--illustrate",
        type=int,
        default=0,
        help="render a real-vs-generated spectrogram figure for the N highest-RPS chunks",
    )
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    override_dir: str | None = None
    if not args.skip_refined_labels:
        from data_processing.noise_rps_dataset import resolve_override_dir

        folder = resolve_override_dir(args.refined_dir)
        if folder.is_dir() and any(folder.glob("*.npz")):
            override_dir = str(args.refined_dir)
        else:
            print(
                f"WARNING: no refined-label sidecars in {folder}. Falling back to the "
                "ORIGINAL telemetry as the reference trajectory — readout 1 then "
                "measures fidelity against a known-biased track, and a 'refined' arm "
                "cannot be conditioned at all.",
                file=sys.stderr,
            )

    rigs = tuple(r.strip() for r in str(args.rigs).split(",") if r.strip())
    for rig in rigs:
        _rig_spec(rig)  # fail on a typo before any dataset is streamed
    keep_splits: set[str] | None = None
    if args.split_filter:
        keep_splits = {s.strip() for s in str(args.split_filter).split(",") if s.strip()}
        unknown = keep_splits - {"train", "valid", "boundary"}
        if unknown:
            print(f"unknown split tag(s) {sorted(unknown)}", file=sys.stderr)
            return 1

    ks = np.arange(1, int(args.k_max) + 1)
    t_chunks = time.time()
    chunks = build_chunks(
        seconds=args.chunk_seconds,
        hop_seconds=args.hop_seconds,
        min_flight_rps=args.min_flight_rps,
        override_dir=override_dir,
        max_chunks=args.max_chunks,
        rigs=rigs,
        keep_splits=keep_splits,
    )
    if not chunks:
        print("no eval chunks (check --min-flight-rps / --split-filter)", file=sys.stderr)
        return 1
    splits = {s: sum(1 for c in chunks if c.split == s) for s in ("train", "valid", "boundary")}
    per_rig = {
        rig: {s: sum(1 for c in chunks if c.rig == rig and c.split == s) for s in splits}
        for rig in rigs
    }
    print(
        f"{len(chunks)} chunks of {args.chunk_seconds:g}s from "
        f"{len({c.recording_id for c in chunks})} recording(s) in "
        f"{time.time() - t_chunks:.1f}s — {splits}"
    )
    if len(rigs) > 1:
        for rig in rigs:
            print(f"  {rig}: {sum(per_rig[rig].values())} chunks — {per_rig[rig]}")

    if args.self_test:
        return self_test(chunks, ks=ks, stride=args.line_stride, gate_frac=args.gate)

    import torch

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")  # noqa: SIM222
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The reference row: the RECORDING read along the refined trajectory. It is
    # arm-independent, so it is computed once and reused by every arm.
    ref_variant = "refined" if all("refined" in c.labels for c in chunks) else "orig"
    real_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, LineTable]] = {}
    for chunk in chunks:
        power, freqs = stft_power(chunk.audio, n_fft=N_FFT, hop=HOP, sr=chunk.sample_rate)
        rates = rates_at_frames(chunk.labels[ref_variant], power.shape[1], n_fft=N_FFT, hop=HOP)
        tab = line_table(
            power,
            freqs,
            rates,
            ks=ks,
            sr=chunk.sample_rate,
            stride=args.line_stride,
            gate_frac=args.gate,
        )
        real_cache[chunk.index] = (power, freqs, rates, tab)

    rows: list[dict[str, Any]] = []
    bin_counts: list[tuple[float, float]] = []
    illustrations: dict[int, dict[str, np.ndarray]] = {}
    # The highest-RPS chunks, not the first ones: the recording opens on the
    # takeoff ramp, whose comb is sweeping too fast to look at.
    illustrate_ids = [
        c.index for c in sorted(chunks, key=lambda c: -c.mean_rps)[: max(0, int(args.illustrate))]
    ]
    timings: dict[str, float] = {}

    for exp in [e for e in str(args.experiments).split(",") if e]:
        t_arm = time.time()
        print(f"=== {exp} ===", flush=True)
        try:
            # Compose first: an arm trained on labels this run cannot reproduce is
            # unscoreable, and that is worth knowing before an R2 download.
            ckpt = resolve_checkpoint(exp, args.checkpoint)
            cfg = _compose(exp, ckpt)
            unscoreable = [
                rig
                for rig in rigs
                if label_variant_of(cfg, rig)[0] == "refined"
                and any("refined" not in c.labels for c in chunks if c.rig == rig)
            ]
            if unscoreable:
                print(
                    f"  SKIPPED: {exp} trained on refined {'/'.join(unscoreable)} labels, "
                    "and no sidecars are present to reproduce them",
                    file=sys.stderr,
                )
                continue
            arm = load_arm(exp, device=device, checkpoint=ckpt, cfg=cfg)
        except Exception as exc:  # noqa: BLE001 - one broken arm must not kill the sweep
            print(f"  FAILED to load: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        shown = ", ".join(f"{r}={arm.labels_for(r)[0]}(x{arm.labels_for(r)[1]:g})" for r in rigs)
        print(f"  labels: {shown}  ckpt={arm.checkpoint}")
        for chunk in chunks:
            real_power, freqs, ref_rates, real_tab = real_cache[chunk.index]
            gen = render(arm, chunk, device=device, seed=args.seed + chunk.index)
            score = score_chunk(
                gen=gen,
                chunk=chunk,
                real_power=real_power,
                freqs=freqs,
                ref_rates=ref_rates,
                arm_rps=chunk.label(*arm.labels_for(chunk.rig)),
                ks=ks,
                stride=args.line_stride,
                gate_frac=args.gate,
            )
            mrstft = mrstft_of(arm, gen, chunk)
            if chunk.index in illustrate_ids:
                illustrations.setdefault(chunk.index, {})[exp] = gen
            for k in ks:
                k = int(k)
                band = band_of(k)
                if band is None:
                    continue
                delta = score.delta_logmag.get(k)
                gen_line = score.line_gen.line.get(k)
                real_line = real_tab.line.get(k)
                row: dict[str, Any] = {
                    "arm": exp,
                    "label_variant": arm.labels_for(chunk.rig)[0],
                    "rig": chunk.rig,
                    "chunk": chunk.index,
                    "recording_id": chunk.recording_id,
                    "t_rel": round(chunk.t_rel, 3),
                    "split": chunk.split,
                    "mean_rps": round(chunk.mean_rps, 3),
                    "k": k,
                    "band": band,
                    "delta_logmag_db": float(np.mean(delta)) if delta is not None else None,
                    "delta_logmag_db_mic_std": float(np.std(delta)) if delta is not None else None,
                    "line_gen_db": float(np.mean(score.line_gen.line_db(k)))
                    if gen_line is not None
                    else None,
                    "line_real_db": float(np.mean(real_tab.line_db(k)))
                    if real_line is not None
                    else None,
                    "line_delta_db": None,
                    "line_delta_db_mic_std": None,
                    "ptf_gen_db": float(np.mean(score.line_gen.ptf_db(k)))
                    if gen_line is not None
                    else None,
                    "ptf_real_db": float(np.mean(real_tab.ptf_db(k)))
                    if real_line is not None
                    else None,
                    "ptf_delta_db": None,
                    "n_cells_gen": score.line_gen.count.get(k, 0),
                    "n_cells_real": real_tab.count.get(k, 0),
                    "gain_db": round(score.gain_db, 4),
                    "mrstft": mrstft,
                }
                if gen_line is not None and real_line is not None:
                    per_mic = score.line_gen.line_db(k) - real_tab.line_db(k)
                    row["line_delta_db"] = float(np.mean(per_mic))
                    row["line_delta_db_mic_std"] = float(np.std(per_mic))
                    # The paired reading: same cells, same floor regions, so every
                    # bias of the estimator cancels and only the arm remains.
                    row["ptf_delta_db"] = float(
                        np.mean(score.line_gen.ptf_db(k) - real_tab.ptf_db(k))
                    )
                    bin_counts.append((real_tab.n_band_bins[k], real_tab.n_floor_bins[k]))
                rows.append(row)
        timings[exp] = time.time() - t_arm
        print(f"  {timings[exp]:.1f}s wall for {len(chunks)} chunks")

    if not rows:
        print("no arm produced any rows", file=sys.stderr)
        return 1

    # The zero of the peak-to-floor axis, at the bin counts this run actually read.
    null_db = estimator_null_db(
        float(np.mean([b for b, _ in bin_counts])), float(np.mean([f for _, f in bin_counts]))
    )

    write_rows(out_dir / "per_k.csv", rows)
    summary = summarize(rows)
    write_summary(out_dir / "summary.csv", summary)
    (out_dir / "run.json").write_text(
        json.dumps(
            {
                "reference_labels": ref_variant,
                "rigs": list(rigs),
                "split_filter": sorted(keep_splits) if keep_splits else None,
                "n_chunks": len(chunks),
                "splits": splits,
                "splits_per_rig": per_rig,
                "n_fft": N_FFT,
                "hop": HOP,
                "k_max": int(args.k_max),
                "line_stride": int(args.line_stride),
                "gate": float(args.gate),
                "stochastic_floor_db": STOCHASTIC_FLOOR_DB,
                "estimator_null_db": null_db,
                "seconds_per_arm": timings,
            },
            indent=2,
        )
    )
    plotted = plot_per_k(rows, out_dir, null_db=null_db)
    for idx, per_arm in illustrations.items():
        chunk = next(c for c in chunks if c.index == idx)
        plot_illustration(chunk.audio, per_arm, chunk, out_dir / f"illustration_chunk{idx:03d}.png")

    header = (
        f"\n{'arm':<26}{'rig':<10}{'band':<9}{'dLogMag':>9}{'dLine':>9}"
        f"{'PTFgen':>9}{'PTFreal':>9}{'dPTF':>9}{'mrstft':>9}"
    )
    print(header)
    print("-" * 99)
    for rec in summary:
        print(
            f"{rec['arm']:<26}{rec['rig']:<10}{rec['band']:<9}{rec['delta_logmag_db']:>9.3f}"
            f"{rec['line_delta_db']:>9.3f}{rec['ptf_gen_db']:>9.3f}"
            f"{rec['ptf_real_db']:>9.3f}{rec['ptf_delta_db']:>9.3f}{rec['mrstft']:>9.3f}"
        )
    print(
        f"\nread dLogMag against {STOCHASTIC_FLOOR_DB:.2f} dB (what a perfect but stochastic "
        f"model scores) and PTFgen/PTFreal against {null_db:+.2f} dB (the estimator's null on "
        "a line-free band): a peak-to-floor at the null means there is no measurable tooth. "
        "dPTF is the paired difference, in which the estimator's biases cancel."
    )
    if splits["valid"] == 0:
        print(
            "NOTE: no chunk falls in the held-out split — the recording's first "
            "val_pct is its pre-takeoff ramp, which --min-flight-rps removes. Every "
            "arm therefore scores on audio it saw in training; this is a comb-shape "
            "comparison between arms, not a generalization measurement."
        )
    print(f"\nwrote {out_dir}/per_k.csv, summary.csv, run.json, {len(plotted)} plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
