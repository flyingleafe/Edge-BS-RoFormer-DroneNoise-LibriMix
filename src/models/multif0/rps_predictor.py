"""
Multi-F0 CNN adapted for RPS prediction.

Wraps the LateDeep multi-F0 CNN (ISMIR 2020, Cuesta et al.) with:
    - Pluggable spectral front-end (default: HCQT magnitude+phase)
    - Differentiable "soft centroid" RPS extraction from the salience map
    - Time resampling from HCQT grid to STFT grid
    - Broadcasting to 4 rotors

Model follows the existing RPS prediction interface:
    forward(audio: (B, samples)) -> (B, 4, T_stft)

Training:  uses MSE/PIT-MSE loss (existing ``train_rps_predictor.py``).
Inference: the soft centroid + threshold peak-finding give refined RPS.

HCQT parameters (paper defaults):
    sr=22050, fmin=32.7 Hz, 6 octaves, 60 bins/octave (20¢) → 360 freq bins
    hop=256 samples, 5 harmonics [1..5]

Architecture summary:
    audio (16kHz) → frontend (HCQT or STFT) →
    LateDeep CNN → sigmoid salience (F, T_hcqt) →
    soft centroid → RPS (T_hcqt) → interpolate → (4, T_stft)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.multif0.hcqt import freq_grid
from models.multif0.model import LateDeep


def stft_time_frames(audio_length: int, hop_length: int = 512, n_fft: int = 2048) -> int:
    """Number of STFT time frames for given audio length (with center padding)."""
    return audio_length // hop_length + 1


class MultiF0RPSPredictor(nn.Module):
    """Multi-F0 CNN (Late/Deep) adapted for single-label RPS classification.

    Input:  (B, samples) raw mono waveform at 16 kHz
    Output: (B, 4, T_stft) predicted RPS (Hz) per STFT frame

    Parameters
    ----------
    n_fft, hop_length : int
        STFT n_fft / hop for the target time grid (output resampling).
    num_rotors : int
        Number of rotors to predict (broadcast to all if num_rotors > 1).
    n_harmonics : int
        Number of HCQT harmonics (only used if ``frontend`` is not provided).
    temperature : float
        Softmax temperature for the soft-centroid RPS extraction.
    frontend : SpectralFrontEnd | None
        Feature extractor.  If None, builds an HCQTFrontEnd with phase
        and ``n_harmonics`` harmonics.
    **frontend_kwargs
        Passed to the default HCQTFrontEnd constructor.
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        num_rotors: int = 4,
        n_harmonics: int = 5,
        temperature: float = 1.0,
        frontend: nn.Module | None = None,
        **frontend_kwargs,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length  # STFT hop (target time grid)
        self.num_rotors = num_rotors
        self.temperature = temperature

        # ── Front-end ──
        if frontend is None:
            from models.frontends import build_frontend

            frontend = build_frontend(
                "hcqt",
                phase=True,
                **frontend_kwargs,
            )
        self.frontend = frontend
        # Infer n_harmonics: if phase front-end → C=2H, else C=H.
        # Also read actual harmonics from the frontend for the freq grid.
        uses_phase = getattr(frontend, "use_phase", True)
        self.n_harmonics = frontend.out_channels // 2 if uses_phase else frontend.out_channels

        # ── CNN backbone ──
        self.cnn = LateDeep(n_harmonics=self.n_harmonics)

        # ── Frequency grid for soft-centroid ──
        # Build from frontend params if it's an HCQT frontend, else default.
        if (
            hasattr(frontend, "fmin")
            and hasattr(frontend, "n_octaves")
            and hasattr(frontend, "over_sample")
        ):
            fg = freq_grid(
                fmin=frontend.fmin,
                n_octaves=frontend.n_octaves,
                over_sample=frontend.over_sample,
            )
        else:
            fg = freq_grid()  # default paper params
        self.register_buffer("freq_grid", torch.from_numpy(fg).float())

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: (B, samples) or (B, 1, samples) at 16 kHz.

        Returns:
            rps:   (B, num_rotors, T_stft) predicted rotor speeds in Hz.
        """
        B = audio.shape[0]
        N = audio.shape[-1] if audio.dim() == 3 else audio.shape[-1]
        T_stft = stft_time_frames(N, self.hop_length, self.n_fft)

        # 1. Front-end → (B, 2H, F, T_hcqt)
        feats = self.frontend(audio)

        # 2. Split channels into mag + dphase for the CNN.
        #    If frontend provides phase (C=2H), split evenly.
        #    If no phase (C=H), dphase is zeros (CNN fallback).
        H = self.n_harmonics
        if getattr(self.frontend, "use_phase", True):
            mag = feats[:, :H, :, :]
            dphase = feats[:, H:, :, :]
        else:
            mag = feats
            dphase = torch.zeros_like(mag)

        # 3. CNN → salience map
        salience = self.cnn(mag, dphase)  # (B, 1, F, T)
        salience = salience.squeeze(1)  # (B, F, T)

        # 4. Soft centroid → scalar RPS per HCQT frame
        probs = F.softmax(salience / self.temperature, dim=1)  # (B, F, T)
        rps_hcqt = (probs * self.freq_grid.view(1, -1, 1)).sum(dim=1)  # (B, T)

        # 5. Resample HCQT time grid → STFT time grid
        if rps_hcqt.shape[-1] != T_stft:
            rps = F.interpolate(
                rps_hcqt.unsqueeze(1),
                size=T_stft,
                mode="linear",
                align_corners=False,
            ).squeeze(1)  # (B, T_stft)
        else:
            rps = rps_hcqt  # already aligned

        # 6. Broadcast to num_rotors
        rps = rps.unsqueeze(1).expand(-1, self.num_rotors, -1)  # (B, 4, T_stft)

        return rps

    # ── Inference helpers ────────────────────────────────────────────────

    @torch.no_grad()
    def predict_salience(self, audio: torch.Tensor) -> torch.Tensor:
        """Return the raw salience map (F, T_hcqt) for analysis/plotting."""
        self.eval()
        feats = self.frontend(audio)
        H = self.n_harmonics
        if getattr(self.frontend, "use_phase", True):
            mag = feats[:, :H, :, :]
            dphase = feats[:, H:, :, :]
        else:
            mag = feats
            dphase = torch.zeros_like(mag)
        salience = self.cnn(mag, dphase)  # (B, 1, F, T_hcqt)
        return salience.squeeze(1)  # (B, F, T_hcqt)

    @torch.no_grad()
    def predict_rps_peaks(
        self, audio: torch.Tensor, threshold: float = 0.5
    ) -> list[list[np.ndarray]]:
        """Extract RPS via peak-picking (original paper method).

        Returns:
            List over batch of list over time frames of RPS arrays (Hz).
        """
        import scipy.signal

        self.eval()
        feats = self.frontend(audio)
        H = self.n_harmonics
        if getattr(self.frontend, "use_phase", True):
            mag = feats[:, :H, :, :]
            dphase = feats[:, H:, :, :]
        else:
            mag = feats
            dphase = torch.zeros_like(mag)
        salience = self.cnn(mag, dphase).squeeze(1)  # (B, F, T)
        salience_np = salience.cpu().numpy()
        fg = self.freq_grid.cpu().numpy()

        results = []
        for b in range(salience_np.shape[0]):
            frame_rps = []
            for t in range(salience_np.shape[2]):
                col = salience_np[b, :, t]
                peaks = scipy.signal.argrelmax(col)[0]
                peak_vals = col[peaks]
                above = peaks[peak_vals >= threshold]
                frame_rps.append(fg[above])
            results.append(frame_rps)
        return results
