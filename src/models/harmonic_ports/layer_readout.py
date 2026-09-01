"""Per-rotor salience layers with the lossless CRF readout, for the three ports.

WHY THIS MODULE EXISTS. `models.salience_crf` measured the framework's shared
salience representation and found it is NOT an isomorphism: a PERFECT target,
encoded and decoded again on real training telemetry, comes back 8.24 rev/s away
on average, and 39-45% of frames land more than half a bin off. That is an
oracle floor — no model reading that representation can score better, however
well it predicts. The Gaussian-per-rotor-layer target with a CRF plus
log-parabolic readout comes back at 2.22e-16.

This module is the seam that puts the three harmonic ports on that pair. It
holds the two halves the framework contract does not:

1. ``split_maps`` — the ports keep returning ``(B, n_maps*G, T)``, because the
   ``salience_rps`` codec declares ``("batch", "freq", "time")`` and a 4-D model
   output would not type-check through it. The per-rotor layers are therefore
   STACKED ALONG THE OUTPUT AXIS and split back out here. The tensor a loss or a
   decoder sees is ``(B, R, G, T)``; only the wire format is flat.
2. ``LayerCRFReadout`` — ``predict_rps`` by one CRF best path per rotor layer,
   with NO threshold and NO Hungarian step. A stopped rotor is the path sitting
   at bin 0, which is a value; the old decoder had to call it an absence.

THE READOUT IS ``logsigmoid``, NOT THE RAW LOGIT, and the reason is the loss.
`losses.LayerPITSalienceBCELoss` drives ``sigmoid(z) -> exp(-d^2 / 2 sigma^2)``,
so it is ``log sigmoid(z)``, not ``z``, that converges to the log of a Gaussian
— and it is the log of a Gaussian that a three-point parabolic fit inverts
EXACTLY. Feeding the raw logit instead fits the vertex on
``log p - log(1 - p)``, which diverges at the peak. Measured on 40 clips of real
training telemetry (DREGON ``in_flight_noise`` minus the held-out room1, plus
FLY125) by pushing a PERFECT model's output — ``logit(target)`` — through this
decode, on the ports' own 0-150 rev/s grid at 0.5017 rev/s:

    readout        mean err        max err
    logsigmoid     3.55e-15        1.42e-14
    raw logit      5.93e-02        9.84e-01

``readout="raw"`` keeps the other branch available for that comparison.

THE BAND IS SIZED FROM THE SAME MEASUREMENT. At ``MAX_STEP_REV_S = 16`` the same
round trip returns 1.22e-05 mean and 0.203 max: exact almost everywhere, and
wrong on the frames whose slew the band does not make free, where the transition
penalty shaves the corner of a ramp. Those clips reach 23.8 rev/s in one 32 ms
frame, so 16 is too small for them and 25 is not — at 25 the error is the
``logsigmoid`` row above, everywhere.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F

from models.salience_crf import band_for_rev_s, crf_decode_layers

__all__ = ["split_maps", "LayerCRFReadout", "MAX_STEP_REV_S"]

# The per-frame change the transition band makes FREE, in rev/s. Sized from the
# data and not from a slew constant chosen on synthetic data: 40 clips of real
# training telemetry on the 32 ms frame grid move 0.476 rev/s per frame on
# average, 2.65 at the 99th percentile and 23.8 at the extreme. A band that
# forbids more than the extreme makes the true trajectory undecodable — at 16
# the round trip loses 0.203 rev/s on its worst frame, at 25 it loses 1.4e-14.
MAX_STEP_REV_S = 25.0


def split_maps(logits: torch.Tensor, n_maps: int) -> torch.Tensor:
    """``(B, n_maps*G, T)`` wire format -> ``(B, n_maps, G, T)`` layers."""
    b, fg, t = logits.shape
    if fg % n_maps:
        raise ValueError(f"output axis {fg} is not divisible by n_maps={n_maps}")
    return logits.reshape(b, n_maps, fg // n_maps, t)


class LayerCRFReadout:
    """Mixin: decode per-rotor salience layers by one CRF path per layer.

    Mix it in BEFORE :class:`models.salience_rps.SalienceRPSPredictor` so this
    ``predict_rps`` wins. With ``n_maps == 1`` every call falls through to the
    base class, so an old single-map config keeps its old behaviour exactly.
    """

    n_maps: int
    out_freqs: np.ndarray | None
    num_rotors: int
    hop_length: int
    readout: str = "logsigmoid"
    max_step_rev_s: float = MAX_STEP_REV_S
    crf_stiff: float = 40.0

    # ── the band ────────────────────────────────────────────────────────────

    def grid_step(self) -> float:
        g = np.asarray(cast(np.ndarray, self.out_freqs), dtype=np.float64)
        return float(np.median(np.diff(g)))

    def crf_band(self, dtype=torch.float32) -> tuple[int, torch.Tensor]:
        """``(span, pen)`` for :func:`models.salience_crf.crf_decode_layers`."""
        return band_for_rev_s(self.max_step_rev_s, self.grid_step(), self.crf_stiff, dtype=dtype)

    # ── decode ──────────────────────────────────────────────────────────────

    def decode_salience(self, logits: torch.Tensor) -> torch.Tensor:
        """``(B, R*G, T)`` raw model output -> ``(B, R, T)`` rev/s.

        No threshold and no assignment: rotor ``i`` is the best path through
        layer ``i``. ``crf_decode_layers(..., logits=True)`` wants a log-domain
        score, and under the layer BCE loss that is ``log sigmoid(z)``.
        """
        layers = split_maps(logits, int(self.n_maps))
        if self.readout == "logsigmoid":
            scores = F.logsigmoid(layers)
        elif self.readout == "raw":
            scores = layers
        else:
            raise ValueError(f"unknown readout {self.readout!r}")
        span, pen = self.crf_band(dtype=scores.dtype)
        return crf_decode_layers(
            scores, cast(np.ndarray, self.out_freqs), span, pen.to(scores.device), logits=True
        )

    @torch.no_grad()
    def predict_rps(
        self,
        audio: torch.Tensor,
        *,
        threshold: float | None = None,
        max_jump_bins: int | None = None,
        chunk_size: int = 8,
        **_: Any,
    ) -> torch.Tensor:
        """Audio -> ``(B, num_rotors, T_stft)`` rev/s.

        ``threshold`` and ``max_jump_bins`` are accepted and IGNORED: they are
        the single-map decoder's parameters, and this readout has neither a
        detection decision nor a per-frame jump cap (the CRF's transition band
        is the jump model, and it is sized from the data).
        """
        if int(self.n_maps) == 1:
            return super().predict_rps(  # type: ignore[misc]
                audio,
                threshold=0.3 if threshold is None else threshold,
                max_jump_bins=max_jump_bins,
                chunk_size=chunk_size,
            )
        if chunk_size and chunk_size > 0 and audio.shape[0] > chunk_size:
            logits = torch.cat(
                [
                    self.forward(audio[i : i + chunk_size])  # type: ignore[attr-defined]
                    for i in range(0, audio.shape[0], chunk_size)
                ],
                dim=0,
            )
        else:
            logits = self.forward(audio)  # type: ignore[attr-defined]
        rps_grid = self.decode_salience(logits)  # (B, R, T_grid)

        n_samples = audio.shape[-1]
        t_stft = int(n_samples) // int(self.hop_length) + 1
        if rps_grid.shape[-1] != t_stft:
            rps_grid = F.interpolate(
                rps_grid.float(), size=t_stft, mode="linear", align_corners=False
            )
        return rps_grid
