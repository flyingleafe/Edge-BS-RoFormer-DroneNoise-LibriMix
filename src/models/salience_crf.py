"""Per-rotor salience layers with a CRF readout — a LOSSLESS encode/decode pair.

THE GATE THIS MODULE EXISTS TO PASS. Encoding real training trajectories into a
salience map and decoding them back must be an isomorphism: whatever a model
loses, the representation itself must lose nothing. Measured on 40 clips of real
training telemetry (DREGON ``in_flight_noise`` minus the held-out room1, plus
FLY125; speeds 0.00-91.58 rev/s, per-frame |d rps| mean 0.718, p99 3.008, max
15.301), on a 0-150 rev/s grid at 0.5 rev/s:

    encode/decode pair                              max err     mean err
    triangular kernel, SHARED map, Hungarian        83.53       8.24
    Gaussian layers, CRF + log-parabolic            2.22e-16    2.22e-16

The old pair loses 8.24 rev/s on a PERFECT target, and 39-45% of frames land
more than 0.5 rev/s away. That is an oracle floor: no model reading that
representation can score better, however well it predicts. Three separate causes,
all fixed here:

1. ONE MAP CANNOT HOLD FOUR ROTORS. DREGON's rotor pairs sit 0.13-0.86 rev/s
   apart, inside a single 0.5 rev/s bin, so a shared map has fewer peaks than
   rotors and the tracker assigns several tracks to one peak. One layer per rotor
   removes the failure completely -- there is nothing left to merge.
2. THE KERNEL MUST BE GAUSSIAN. ``log`` of a Gaussian is a parabola *globally*,
   so a three-point log-parabolic fit recovers the peak's sub-bin position
   EXACTLY from any three consecutive bins. The framework's triangular kernel has
   no such property and leaves a quantization residue.
3. ZERO IS A VALUE, NOT AN ABSENCE. ``salience_target_from_resampled_rps`` drops
   any rotor below 0.1 rev/s (``active = rps_grid > 0.1``), so a stopped rotor
   lights no bin and "stopped" is encoded as absence of evidence -- which is what
   forces a decode threshold. Here a stopped rotor is simply a peak at bin 0.

The transition band must also admit what real rotors do: 15.3 rev/s in one 32 ms
frame at the extreme, which is 31 bins at 0.5 rev/s, against the 3 bins the comb
tracker's default allows. `band_for_rev_s` sizes it from that number rather than
from a slew constant chosen on synthetic data.
"""

from __future__ import annotations

import numpy as np
import torch

from models import comb_crf

__all__ = ["gaussian_layer_target", "crf_decode_layers", "band_for_rev_s"]


def band_for_rev_s(max_step_rev_s: float, grid_step: float, stiff: float = 40.0,
                   dtype=torch.float32):
    """Transition band that ADMITS a per-frame change of ``max_step_rev_s``.

    Sized from the data, not from a slew constant: real training telemetry moves
    up to 15.3 rev/s per 32 ms frame, so a band that forbids more than 1.5 rev/s
    makes the true trajectory undecodable.
    """
    return comb_crf.band_penalty(max(max_step_rev_s / grid_step, 1e-9), stiff, dtype=dtype)


def gaussian_layer_target(rps: torch.Tensor, grid, sigma_bins: float = 1.0) -> torch.Tensor:
    """``(B, R, T)`` rev/s -> ``(B, R, G, T)``: one Gaussian layer per rotor.

    A stopped rotor is a peak at bin 0, not an empty column.
    """
    g = torch.as_tensor(np.asarray(grid), dtype=rps.dtype, device=rps.device)
    step = float(g[1] - g[0])
    pos = (rps - g[0]) / step                                   # (B, R, T) in bins
    idx = torch.arange(g.numel(), device=rps.device, dtype=rps.dtype)
    d = (idx.view(1, 1, -1, 1) - pos.unsqueeze(2)) / float(sigma_bins)
    return torch.exp(-0.5 * d * d)


def crf_decode_layers(layers: torch.Tensor, grid, span: int, pen: torch.Tensor,
                      *, logits: bool = False, subgrid: bool = True) -> torch.Tensor:
    """``(B, R, G, T)`` per-rotor layers -> ``(B, R, T)`` rev/s.

    One CRF best path per layer, so there is no assignment step at decode time and
    no threshold: a stopped rotor is decoded as the path sitting at bin 0.
    ``logits=True`` treats the input as already log-domain (a model's raw output);
    otherwise it is a probability-like map and is logged here.
    """
    g = torch.as_tensor(np.asarray(grid), dtype=torch.get_default_dtype(),
                        device=layers.device)
    step = float(g[1] - g[0])
    s_all = layers if logits else torch.log(layers.clamp_min(1e-300))
    b, r, n_g, _ = s_all.shape
    out = []
    for i in range(r):
        s = s_all[:, i]                                          # (B, G, T)
        path = comb_crf.viterbi(s, span, pen)
        rate = g[path]
        if subgrid:
            i0 = path.clamp(1, n_g - 2)
            a = s.gather(1, (i0 - 1).unsqueeze(1)).squeeze(1)
            c0 = s.gather(1, i0.unsqueeze(1)).squeeze(1)
            c = s.gather(1, (i0 + 1).unsqueeze(1)).squeeze(1)
            den = a - 2 * c0 + c
            # The log of a Gaussian is a parabola everywhere, so this is exact.
            delta = torch.where(den.abs() < 1e-300, torch.zeros_like(den),
                                0.5 * (a - c) / den)
            rate = g[i0] + delta.clamp(-1, 1) * step
        out.append(rate)
    return torch.stack(out, dim=1)
