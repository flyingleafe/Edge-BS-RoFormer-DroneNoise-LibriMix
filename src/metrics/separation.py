"""Source-separation quality metrics.

``sdr``/``si_sdr``/``l1_freq``/``neg_log_wmse``/``aura_stft``/``aura_mrstft``/
``bleed_full`` (``bleedless``/``fullness``) ported from root ``metrics.py``;
``pesq``/``stoi``/``estoi`` ported from ``final_valid.py``
(``calculate_pesq``/``calculate_stoi``).
"""

from __future__ import annotations

import librosa
import numpy as np
import tdseries as td
import torch
import torch.nn.functional as F
from pesq import pesq as _pesq_fn
from pystoi import stoi as _stoi_fn

from framespec import FrameSpec
from metrics._common import AUDIO_RATE, audio_series_spec, get_array

# ─── Pure numpy/torch functions ─────────────────────────────────────────────


def sdr(references: np.ndarray, estimates: np.ndarray) -> np.ndarray:
    """Signal-to-Distortion Ratio (SDR), in dB.

    Args:
        references: (num_sources, num_channels, num_samples) reference signals.
        estimates: same shape, predicted signals.

    Returns:
        (num_sources,) SDR per source.
    """
    eps = 1e-8
    num = np.sum(np.square(references), axis=(1, 2))
    den = np.sum(np.square(references - estimates), axis=(1, 2))
    num += eps
    den += eps
    return 10 * np.log10(num / den)


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-Invariant Signal-to-Distortion Ratio (SI-SDR), in dB.

    Args:
        reference: (num_channels, num_samples) reference signal.
        estimate: same shape, predicted signal.

    The optimal scale and the SDR are both reduced over ``axis=(0, 1)`` —
    i.e. over the *whole* 2D signal at once, giving one global scalar. Unlike
    :func:`sdr`, there is no separate "num_sources" leading axis: pass a
    single (channels, samples) pair (this is how the pre-refactor
    ``get_metrics`` called it — note the docstring in the original root
    ``metrics.py`` claimed a 3D ``(sources, channels, samples)`` shape, but
    every real call site, and the ``axis=(0, 1)`` reduction itself, only
    make sense for 2D input; a 3D call would reduce per-time-step instead of
    globally, which is not scale-invariant SDR at all).
    """
    eps = 1e-8
    scale = np.sum(estimate * reference + eps, axis=(0, 1)) / np.sum(
        reference**2 + eps, axis=(0, 1)
    )
    scale = np.expand_dims(scale, axis=(0, 1))

    reference = reference * scale
    return float(
        np.mean(
            10
            * np.log10(
                np.sum(reference**2, axis=(0, 1))
                / (np.sum((reference - estimate) ** 2, axis=(0, 1)) + eps)
                + eps
            )
        )
    )


def l1_freq(
    reference: np.ndarray,
    estimate: np.ndarray,
    fft_size: int = 2048,
    hop_size: int = 1024,
    device: str | torch.device = "cpu",
) -> float:
    """L1 loss between magnitude STFTs, rescaled to [0, 100] (higher = better).

    Args:
        reference: (num_channels, num_samples) reference audio.
        estimate: (num_channels, num_samples) estimated audio.
    """
    reference_t = torch.from_numpy(reference).to(device)
    estimate_t = torch.from_numpy(estimate).to(device)

    reference_stft = torch.stft(reference_t, fft_size, hop_size, return_complex=True)
    estimated_stft = torch.stft(estimate_t, fft_size, hop_size, return_complex=True)

    reference_mag = torch.abs(reference_stft)
    estimate_mag = torch.abs(estimated_stft)

    loss = 10 * F.l1_loss(estimate_mag, reference_mag)
    return 100 / (1.0 + float(loss.cpu().numpy()))


def neg_log_wmse(
    reference: np.ndarray,
    estimate: np.ndarray,
    mixture: np.ndarray,
    sr: int = 44100,
    device: str | torch.device = "cpu",
) -> float:
    """Negative Log-Weighted MSE (higher = better separation).

    Args:
        reference: (num_channels, num_samples) reference audio.
        estimate: (num_channels, num_samples) estimated audio.
        mixture: (num_channels, num_samples) mixture audio.
        sr: sample rate of all three signals (the original hardcoded 44100;
            exposed here since this project runs natively at 16 kHz).
    """
    from torch_log_wmse import LogWMSE

    log_wmse = LogWMSE(
        audio_length=reference.shape[-1] / sr,
        sample_rate=sr,
        return_as_loss=False,
        bypass_filter=False,
    )

    reference_t = torch.from_numpy(reference).unsqueeze(0).unsqueeze(0).to(device)
    estimate_t = torch.from_numpy(estimate).unsqueeze(0).unsqueeze(0).to(device)
    mixture_t = torch.from_numpy(mixture).unsqueeze(0).to(device)

    res = log_wmse(mixture_t, reference_t, estimate_t)
    return -float(res.cpu().numpy())


def aura_stft(
    reference: np.ndarray,
    estimate: np.ndarray,
    device: str | torch.device = "cpu",
) -> float:
    """STFT loss (log + linear magnitude + spectral convergence), rescaled to
    [0, 100] (higher = better)."""
    from auraloss.freq import STFTLoss

    stft_loss = STFTLoss(w_log_mag=1.0, w_lin_mag=0.0, w_sc=1.0, device=device)

    reference_t = torch.from_numpy(reference).unsqueeze(0).to(device)
    estimate_t = torch.from_numpy(estimate).unsqueeze(0).to(device)

    res = 100 / (1.0 + 10 * stft_loss(reference_t, estimate_t))
    return float(res.cpu().numpy())


def aura_mrstft(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int = 44100,
    device: str | torch.device = "cpu",
) -> float:
    """Multi-Resolution STFT loss (mel-scaled, perceptually weighted), rescaled
    to [0, 100] (higher = better).

    ``sample_rate`` was hardcoded to 44100 in the original; exposed here since
    this project runs natively at 16 kHz.
    """
    from auraloss.freq import MultiResolutionSTFTLoss

    mrstft_loss = MultiResolutionSTFTLoss(
        fft_sizes=[1024, 2048, 4096],
        hop_sizes=[256, 512, 1024],
        win_lengths=[1024, 2048, 4096],
        scale="mel",
        n_bins=128,
        sample_rate=sample_rate,
        perceptual_weighting=True,
        device=device,
    )

    reference_t = torch.from_numpy(reference).unsqueeze(0).float().to(device)
    estimate_t = torch.from_numpy(estimate).unsqueeze(0).float().to(device)

    res = 100 / (1.0 + 10 * mrstft_loss(reference_t, estimate_t))
    return float(res.cpu().numpy())


def bleed_full(
    reference: np.ndarray,
    estimate: np.ndarray,
    sr: int = 44100,
    n_fft: int = 4096,
    hop_length: int = 1024,
    n_mels: int = 512,
    device: str | torch.device = "cpu",
) -> tuple[float, float]:
    """'Bleedless' and 'fullness' scores from mel-spectrogram dB differences.

    'bleedless' measures leakage from the estimate into the reference;
    'fullness' measures completeness of the estimate relative to the
    reference; both computed from mel spectrograms on a decibel scale (higher
    = better for both). Computes both at once (shared STFT/mel work) — use
    :func:`bleedless`/:func:`fullness` if only one is needed for a
    :class:`~metrics._common.Metric`.
    """
    from torchaudio.transforms import AmplitudeToDB

    ref_t = torch.from_numpy(reference).float().to(device)
    est_t = torch.from_numpy(estimate).float().to(device)

    window = torch.hann_window(n_fft).to(device)

    d1 = torch.abs(
        torch.stft(
            ref_t,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
            pad_mode="constant",
        )
    )
    d2 = torch.abs(
        torch.stft(
            est_t,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
            pad_mode="constant",
        )
    )

    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_filter_bank = torch.from_numpy(mel_basis).to(device)

    s1_mel = torch.matmul(mel_filter_bank, d1)
    s2_mel = torch.matmul(mel_filter_bank, d2)

    s1_db = AmplitudeToDB(stype="magnitude", top_db=80)(s1_mel)
    s2_db = AmplitudeToDB(stype="magnitude", top_db=80)(s2_mel)

    diff = s2_db - s1_db

    positive_diff = diff[diff > 0]
    negative_diff = diff[diff < 0]

    average_positive = (
        torch.mean(positive_diff) if positive_diff.numel() > 0 else torch.tensor(0.0).to(device)
    )
    average_negative = (
        torch.mean(negative_diff) if negative_diff.numel() > 0 else torch.tensor(0.0).to(device)
    )

    bleedless_score = 100 * 1 / (average_positive + 1)
    fullness_score = 100 * 1 / (-average_negative + 1)

    return float(bleedless_score.cpu().numpy()), float(fullness_score.cpu().numpy())


def bleedless(
    reference: np.ndarray,
    estimate: np.ndarray,
    sr: int = 44100,
    n_fft: int = 4096,
    hop_length: int = 1024,
    n_mels: int = 512,
    device: str | torch.device = "cpu",
) -> float:
    """'Bleedless' score only — see :func:`bleed_full`."""
    return bleed_full(
        reference, estimate, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, device=device
    )[0]


def fullness(
    reference: np.ndarray,
    estimate: np.ndarray,
    sr: int = 44100,
    n_fft: int = 4096,
    hop_length: int = 1024,
    n_mels: int = 512,
    device: str | torch.device = "cpu",
) -> float:
    """'Fullness' score only — see :func:`bleed_full`."""
    return bleed_full(
        reference, estimate, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, device=device
    )[1]


def pesq(ref: np.ndarray, est: np.ndarray, orig_sr: int) -> float:
    """Perceptual Evaluation of Speech Quality (PESQ). NaN if the pesq
    backend fails (e.g. near-silent input)."""
    if ref.ndim == 2:
        ref = librosa.to_mono(ref) if ref.shape[0] > 1 else ref.squeeze(0)
    if est.ndim == 2:
        est = librosa.to_mono(est) if est.shape[0] > 1 else est.squeeze(0)

    target_sr = 16000 if orig_sr >= 16000 else 8000
    ref = librosa.resample(ref, orig_sr=orig_sr, target_sr=target_sr)
    est = librosa.resample(est, orig_sr=orig_sr, target_sr=target_sr)

    try:
        return float(_pesq_fn(target_sr, ref, est, "wb" if target_sr == 16000 else "nb"))
    except Exception:
        return float("nan")


def stoi(ref: np.ndarray, est: np.ndarray, orig_sr: int, extended: bool = False) -> float:
    """Short-Time Objective Intelligibility (STOI), range [0, 1] (higher =
    better). NaN if the pystoi backend fails."""
    if ref.ndim == 2:
        ref = librosa.to_mono(ref) if ref.shape[0] > 1 else ref.squeeze(0)
    if est.ndim == 2:
        est = librosa.to_mono(est) if est.shape[0] > 1 else est.squeeze(0)

    target_sr = 10000
    ref = librosa.resample(ref, orig_sr=orig_sr, target_sr=target_sr)
    est = librosa.resample(est, orig_sr=orig_sr, target_sr=target_sr)

    try:
        return float(_stoi_fn(ref, est, target_sr, extended=extended))
    except Exception:
        return float("nan")


def estoi(ref: np.ndarray, est: np.ndarray, orig_sr: int) -> float:
    """Extended STOI — see :func:`stoi`."""
    return stoi(ref, est, orig_sr, extended=True)


# ─── Frame adapters ──────────────────────────────────────────────────────────


def _as_2d(x: np.ndarray) -> np.ndarray:
    """Promote mono ``(time,)`` Frame audio to ``(1, time)`` — every pure
    function in this module expects ``(channels, samples)``."""
    return x[None, :] if x.ndim == 1 else x


class _AudioPairMetric:
    """Base for metrics comparing ``pred[pred_key]`` against ``target[target_key]``.

    Subclasses implement :meth:`_compute` on the 2D ``(channels, samples)``
    reference/estimate pair. ``sample_rate`` (a plain int, independent of the
    ``(num, den)`` Frame rate constraint in ``requires_pred``/``requires_target``)
    is what the wrapped pure function needs for resampling/window-sizing —
    default 16000, this project's native rate.
    """

    pred_key: str
    target_key: str
    sample_rate: int

    def __init__(
        self,
        *,
        n_channels: int | None = None,
        sr: tuple[int, int] = AUDIO_RATE,
        sample_rate: int = 16000,
        pred_key: str = "enhanced",
        target_key: str = "target",
    ) -> None:
        self.pred_key = pred_key
        self.target_key = target_key
        self.sample_rate = sample_rate
        spec = audio_series_spec(n_channels, sr)
        self.requires_pred = FrameSpec({pred_key: spec})
        self.requires_target = FrameSpec({target_key: spec})

    def _compute(self, reference: np.ndarray, estimate: np.ndarray) -> float:
        raise NotImplementedError

    def __call__(self, pred: td.Frame, target: td.Frame) -> float:
        estimate = _as_2d(get_array(pred, self.pred_key))
        reference = _as_2d(get_array(target, self.target_key))
        return self._compute(reference, estimate)


class SDRMetric(_AudioPairMetric):
    """Frame adapter around :func:`sdr`."""

    def _compute(self, reference: np.ndarray, estimate: np.ndarray) -> float:
        return float(sdr(reference[None], estimate[None])[0])


class SISDRMetric(_AudioPairMetric):
    """Frame adapter around :func:`si_sdr`."""

    def _compute(self, reference: np.ndarray, estimate: np.ndarray) -> float:
        return si_sdr(reference, estimate)


class L1FreqMetric(_AudioPairMetric):
    """Frame adapter around :func:`l1_freq`."""

    def __init__(
        self,
        *,
        n_channels: int | None = None,
        sr: tuple[int, int] = AUDIO_RATE,
        pred_key: str = "enhanced",
        target_key: str = "target",
        fft_size: int = 2048,
        hop_size: int = 1024,
        device: str = "cpu",
    ) -> None:
        super().__init__(n_channels=n_channels, sr=sr, pred_key=pred_key, target_key=target_key)
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.device = device

    def _compute(self, reference: np.ndarray, estimate: np.ndarray) -> float:
        return l1_freq(reference, estimate, self.fft_size, self.hop_size, self.device)


class AuraSTFTMetric(_AudioPairMetric):
    """Frame adapter around :func:`aura_stft`."""

    def __init__(
        self,
        *,
        n_channels: int | None = None,
        sr: tuple[int, int] = AUDIO_RATE,
        pred_key: str = "enhanced",
        target_key: str = "target",
        device: str = "cpu",
    ) -> None:
        super().__init__(n_channels=n_channels, sr=sr, pred_key=pred_key, target_key=target_key)
        self.device = device

    def _compute(self, reference: np.ndarray, estimate: np.ndarray) -> float:
        return aura_stft(reference, estimate, self.device)


class AuraMRSTFTMetric(_AudioPairMetric):
    """Frame adapter around :func:`aura_mrstft`."""

    def __init__(
        self,
        *,
        n_channels: int | None = None,
        sr: tuple[int, int] = AUDIO_RATE,
        sample_rate: int = 16000,
        pred_key: str = "enhanced",
        target_key: str = "target",
        device: str = "cpu",
    ) -> None:
        super().__init__(
            n_channels=n_channels,
            sr=sr,
            sample_rate=sample_rate,
            pred_key=pred_key,
            target_key=target_key,
        )
        self.device = device

    def _compute(self, reference: np.ndarray, estimate: np.ndarray) -> float:
        return aura_mrstft(reference, estimate, self.sample_rate, self.device)


class BleedlessMetric(_AudioPairMetric):
    """Frame adapter around :func:`bleedless`."""

    def __init__(
        self,
        *,
        n_channels: int | None = None,
        sr: tuple[int, int] = AUDIO_RATE,
        sample_rate: int = 16000,
        pred_key: str = "enhanced",
        target_key: str = "target",
        device: str = "cpu",
    ) -> None:
        super().__init__(
            n_channels=n_channels,
            sr=sr,
            sample_rate=sample_rate,
            pred_key=pred_key,
            target_key=target_key,
        )
        self.device = device

    def _compute(self, reference: np.ndarray, estimate: np.ndarray) -> float:
        return bleedless(reference, estimate, sr=self.sample_rate, device=self.device)


class FullnessMetric(_AudioPairMetric):
    """Frame adapter around :func:`fullness`."""

    def __init__(
        self,
        *,
        n_channels: int | None = None,
        sr: tuple[int, int] = AUDIO_RATE,
        sample_rate: int = 16000,
        pred_key: str = "enhanced",
        target_key: str = "target",
        device: str = "cpu",
    ) -> None:
        super().__init__(
            n_channels=n_channels,
            sr=sr,
            sample_rate=sample_rate,
            pred_key=pred_key,
            target_key=target_key,
        )
        self.device = device

    def _compute(self, reference: np.ndarray, estimate: np.ndarray) -> float:
        return fullness(reference, estimate, sr=self.sample_rate, device=self.device)


class PESQMetric(_AudioPairMetric):
    """Frame adapter around :func:`pesq`."""

    def _compute(self, reference: np.ndarray, estimate: np.ndarray) -> float:
        return pesq(reference, estimate, self.sample_rate)


class STOIMetric(_AudioPairMetric):
    """Frame adapter around :func:`stoi`."""

    def __init__(
        self,
        *,
        n_channels: int | None = None,
        sr: tuple[int, int] = AUDIO_RATE,
        sample_rate: int = 16000,
        pred_key: str = "enhanced",
        target_key: str = "target",
        extended: bool = False,
    ) -> None:
        super().__init__(
            n_channels=n_channels,
            sr=sr,
            sample_rate=sample_rate,
            pred_key=pred_key,
            target_key=target_key,
        )
        self.extended = extended

    def _compute(self, reference: np.ndarray, estimate: np.ndarray) -> float:
        return stoi(reference, estimate, self.sample_rate, extended=self.extended)


class ESTOIMetric(STOIMetric):
    """Frame adapter around :func:`estoi` (``STOIMetric`` with ``extended=True``)."""

    def __init__(
        self,
        *,
        n_channels: int | None = None,
        sr: tuple[int, int] = AUDIO_RATE,
        sample_rate: int = 16000,
        pred_key: str = "enhanced",
        target_key: str = "target",
    ) -> None:
        super().__init__(
            n_channels=n_channels,
            sr=sr,
            sample_rate=sample_rate,
            pred_key=pred_key,
            target_key=target_key,
            extended=True,
        )


class NegLogWMSEMetric:
    """Frame adapter around :func:`neg_log_wmse`.

    Needs the mixture in addition to reference/estimate, so it does not
    subclass :class:`_AudioPairMetric`; ``target[mixture_key]`` (default
    ``"mixture"``) supplies it.
    """

    def __init__(
        self,
        *,
        n_channels: int | None = None,
        sr: tuple[int, int] = AUDIO_RATE,
        sample_rate: int = 16000,
        device: str = "cpu",
        pred_key: str = "enhanced",
        target_key: str = "target",
        mixture_key: str = "mixture",
    ) -> None:
        self.pred_key = pred_key
        self.target_key = target_key
        self.mixture_key = mixture_key
        self.sample_rate = sample_rate
        self.device = device
        spec = audio_series_spec(n_channels, sr)
        self.requires_pred = FrameSpec({pred_key: spec})
        self.requires_target = FrameSpec({target_key: spec, mixture_key: spec})

    def __call__(self, pred: td.Frame, target: td.Frame) -> float:
        estimate = _as_2d(get_array(pred, self.pred_key))
        reference = _as_2d(get_array(target, self.target_key))
        mixture = _as_2d(get_array(target, self.mixture_key))
        return neg_log_wmse(reference, estimate, mixture, sr=self.sample_rate, device=self.device)


__all__ = [
    "sdr",
    "si_sdr",
    "l1_freq",
    "neg_log_wmse",
    "aura_stft",
    "aura_mrstft",
    "bleed_full",
    "bleedless",
    "fullness",
    "pesq",
    "stoi",
    "estoi",
    "SDRMetric",
    "SISDRMetric",
    "L1FreqMetric",
    "NegLogWMSEMetric",
    "AuraSTFTMetric",
    "AuraMRSTFTMetric",
    "BleedlessMetric",
    "FullnessMetric",
    "PESQMetric",
    "STOIMetric",
    "ESTOIMetric",
]
