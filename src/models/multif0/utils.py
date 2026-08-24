"""
Utilities for converting between RPS trajectories and multi-F0 salience maps.

The original Cuesta et al. (ISMIR 2020) model outputs a single-channel
salience map (360, T_hcqt) with values in [0,1] per frequency bin.
Training uses per-bin binary cross-entropy (BCE).

This module provides:
- rps_to_salience:  continuous RPS (Hz) → binary salience map
- salience_to_rps:  salience map → RPS (Hz) via peak-to-bin mapping
- roundtrip_error:  measure irreducible quantization loss
"""

import numpy as np
import torch
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════════════
# CQT frequency grid
# ═══════════════════════════════════════════════════════════════════════════


def cqt_freq_grid(
    fmin: float = 32.7,
    n_octaves: int = 6,
    over_sample: int = 5,
    *,
    n_bins: int | None = None,
    bins_per_octave: int | None = None,
) -> np.ndarray:
    """CQT frequency grid (center frequency of each bin).

    By default the grid is derived from ``n_octaves`` and ``over_sample``
    (``bins_per_octave = 12 * over_sample``, ``n_bins = n_octaves * bins_per_octave``).
    Pass ``n_bins`` / ``bins_per_octave`` explicitly to describe grids whose
    octave count is non-integer — e.g. Basic Pitch's contour grid
    (264 bins, 36 bins/octave, fmin 27.5).

    Returns (n_bins,) float array in Hz.
    """
    import librosa

    if bins_per_octave is None:
        bins_per_octave = 12 * over_sample
    if n_bins is None:
        n_bins = n_octaves * 12 * over_sample
    return librosa.cqt_frequencies(n_bins, fmin=fmin, bins_per_octave=bins_per_octave)


def linear_freq_grid(fmin: float, fmax: float, n_bins: int) -> np.ndarray:
    """Uniform (linear-in-Hz) frequency grid, ``(n_bins,)`` from ``fmin`` to ``fmax``.

    Unlike :func:`cqt_freq_grid` (geometric / log-spaced), this places bins at a
    constant Hz spacing — useful as a fine *output* salience grid concentrated in
    a narrow band (e.g. 360 bins over 55–110 Hz ≈ 0.153 Hz/bin), decoupled from
    the model's log-spaced CQT input grid.
    """
    return np.linspace(float(fmin), float(fmax), int(n_bins), dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# Time grids
# ═══════════════════════════════════════════════════════════════════════════


def hcqt_time_grid(
    n_frames: int,
    sr: float = 22050,
    hop_length: int = 256,
) -> np.ndarray:
    """Time grid for HCQT frames (center of each frame, seconds)."""
    import librosa

    return librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)


def stft_time_grid(
    n_frames: int,
    sr: float = 16000,
    hop_length: int = 512,
) -> np.ndarray:
    """Time grid for STFT frames (center of each frame, seconds)."""
    import librosa

    return librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)


# ═══════════════════════════════════════════════════════════════════════════
# RPS ↔ salience conversion
# ═══════════════════════════════════════════════════════════════════════════


def salience_target_from_resampled_rps(
    rps_grid: torch.Tensor,
    freqs: np.ndarray,
    *,
    blur_bins: int = 0,
) -> torch.Tensor:
    """Nearest-bin BCE salience target from RPS already on the target time grid.

    The shared core of ``rps_to_salience`` (which additionally handles a
    ``rps_sr``-timed interpolation onto the output grid) and
    ``models.salience_rps.SalienceRPSPredictor.salience_target_from_frame_rps``
    (which additionally handles the STFT-grid -> model-grid resample and the
    model's own frequency grid). Both callers, and
    ``losses.salience.SalienceRPSBCELoss`` (which has no model instance to
    query for a frontend/frequency grid), funnel through this one
    implementation so the RPS->salience quantization never drifts between
    them — see docs/refactor-unified-framework.md § C7/C8.

    Args:
        rps_grid: ``(B, R, T)`` per-rotor speed (Hz), already resampled onto
            the target time grid (``T`` = the salience model's frame count).
        freqs: ``(n_bins,)`` frequency grid (Hz) — the salience bins.
        blur_bins: frequency-axis smoothing half-width (0 = strictly binary).

    Returns:
        ``(B, n_bins, T)`` binary (or soft, if ``blur_bins > 0``) target.
    """
    freqs_t = torch.as_tensor(freqs, dtype=rps_grid.dtype, device=rps_grid.device)
    b, r, t = rps_grid.shape
    n_bins = int(freqs_t.numel())

    dists = (rps_grid.unsqueeze(2) - freqs_t.view(1, 1, n_bins, 1)).abs()  # (B, R, n_bins, T)
    nearest_bin = dists.argmin(dim=2)  # (B, R, T)
    active = rps_grid > 0.1

    salience = torch.zeros(b, n_bins, t, device=rps_grid.device, dtype=rps_grid.dtype)
    b_idx = torch.arange(b, device=rps_grid.device).view(b, 1, 1).expand(-1, r, t)
    t_idx = torch.arange(t, device=rps_grid.device).view(1, 1, -1).expand(b, r, -1)
    salience[b_idx[active], nearest_bin[active], t_idx[active]] = 1.0
    salience = salience.clamp(0, 1)

    if blur_bins > 0:
        k = int(blur_bins)
        ramp = torch.arange(1, k + 1, device=salience.device, dtype=salience.dtype)
        kernel = torch.cat([ramp, torch.tensor([k + 1.0], device=salience.device), ramp.flip(0)])
        kernel = (kernel / kernel.max()).view(1, 1, -1)
        sal_f = salience.permute(0, 2, 1).reshape(b * t, 1, n_bins)
        sal_f = F.conv1d(sal_f, kernel, padding=k)
        salience = sal_f.reshape(b, t, n_bins).permute(0, 2, 1).clamp(0, 1)

    return salience


def rps_to_salience(
    rps: torch.Tensor,
    n_hcqt_frames: int,
    *,
    fmin: float = 32.7,
    n_octaves: int = 6,
    over_sample: int = 5,
    n_bins: int | None = None,
    bins_per_octave: int | None = None,
    freqs: np.ndarray | None = None,
    hcqt_sr: int = 22050,
    hcqt_hop: int = 256,
    rps_sr: float = 1000.0,
    rps_t_start: float = 0.0,
    blur_bins: int = 0,
) -> torch.Tensor:
    """Convert RPS trajectories to salience targets.

    Parameters
    ----------
    rps : (B, 4, T_rps) or (4, T_rps)
        RPS in Hz at sample rate ``rps_sr``.
    n_hcqt_frames : int
        Number of HCQT output frames.
    fmin, n_octaves, over_sample :
        CQT parameters. Ignored for the grid size if ``n_bins`` /
        ``bins_per_octave`` are given explicitly (non-integer-octave grids,
        e.g. Basic Pitch contour: ``n_bins=264, bins_per_octave=36, fmin=27.5``).
    hcqt_sr, hcqt_hop :
        Sample rate and hop for the HCQT time grid.
    rps_sr : float
        Sample rate of the RPS data (Hz).
    rps_t_start : float
        Start time of the RPS data relative to the audio start (seconds).
        If the RPS recording starts after the audio, this is > 0.
    blur_bins : int
        If > 0, spread each active bin over a triangular window of half-width
        ``blur_bins`` along the frequency axis (peak 1.0, linear falloff to 0).
        This turns the single-bin-per-rotor target into a soft target so BCE
        training has a non-degenerate gradient. ``blur_bins=0`` leaves the
        target strictly binary (used for round-trip / quantization analysis).

    Returns
    -------
    salience : (B, n_bins, n_hcqt_frames) or (n_bins, n_hcqt_frames)
        Binary (0/1) tensor, or soft [0, 1] tensor when ``blur_bins > 0``.
    """
    was_batched = rps.dim() == 3
    if not was_batched:
        rps = rps.unsqueeze(0)

    B, R, T_rps = rps.shape
    # Frequency grid: an explicit ``freqs`` array (e.g. a linear output grid)
    # overrides the geometric CQT construction.
    if freqs is None:
        freqs = cqt_freq_grid(
            fmin=fmin,
            n_octaves=n_octaves,
            over_sample=over_sample,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
        )
    else:
        freqs = np.asarray(freqs, dtype=np.float64)
    n_bins = len(freqs)
    freqs_t = torch.from_numpy(freqs).float().to(rps.device)

    # HCQT frame times (center of each frame)
    hcqt_times = (torch.arange(n_hcqt_frames, device=rps.device).float() + 0.5) * hcqt_hop / hcqt_sr

    # RPS sample times
    rps_times = torch.arange(T_rps, device=rps.device).float() / rps_sr + rps_t_start

    # For each HCQT frame, find the RPS value at that time via linear interpolation.
    # Use searchsorted to find the enclosing RPS samples.
    # hcqt_times: (T_hcqt,), rps_times: (T_rps,)
    # Find insertion index: idx such that rps_times[idx-1] <= hcqt_times[t] < rps_times[idx]
    idx = torch.searchsorted(rps_times, hcqt_times)  # (T_hcqt,)
    idx = idx.clamp(1, T_rps - 1)

    # Linear interpolation weights
    t_left = rps_times[idx - 1]
    t_right = rps_times[idx]
    w_right = (hcqt_times - t_left) / (t_right - t_left + 1e-10)
    w_left = 1.0 - w_right

    # Interpolate RPS for all rotors at HCQT times
    rps_left = rps[:, :, idx - 1]  # (B, 4, T_hcqt)
    rps_right = rps[:, :, idx]  # (B, 4, T_hcqt)
    rps_hcqt = w_left * rps_left + w_right * rps_right  # (B, 4, T_hcqt)

    # Mask: HCQT frames outside RPS time range get no active bins
    in_range = (hcqt_times >= rps_times[0]) & (hcqt_times <= rps_times[-1])
    in_range = in_range.view(1, 1, -1)  # (1, 1, T_hcqt)

    # For each active rotor at each HCQT frame, find nearest bin.
    # A rotor at ≤ 0 Hz or below fmin is "stopped" — no bin assigned.
    rps_expanded = rps_hcqt.unsqueeze(2)  # (B, 4, 1, T_hcqt)
    freqs_expanded = freqs_t.view(1, 1, n_bins, 1)  # (1, 1, n_bins, 1)

    dists = (rps_expanded - freqs_expanded).abs()  # (B, 4, n_bins, T_hcqt)

    # Active rotors: RPS > 0.1 Hz AND within time range
    active_rotor = (rps_hcqt > 0.1).float()  # (B, 4, T_hcqt)

    nearest_bin = dists.argmin(dim=2)  # (B, 4, T_hcqt)

    salience = torch.zeros(B, n_bins, n_hcqt_frames, device=rps.device)
    b_idx = torch.arange(B, device=rps.device).view(B, 1, 1).expand(-1, R, n_hcqt_frames)
    t_idx = torch.arange(n_hcqt_frames, device=rps.device).view(1, 1, -1).expand(B, R, -1)

    # Only set bins for active rotors AND in-range times
    active = (active_rotor * in_range) > 0.5
    salience[b_idx[active], nearest_bin[active], t_idx[active]] = 1.0

    # If two rotors map to the same bin, that bin stays 1 (unison is invisible)
    salience = salience.clamp(0, 1)

    # Optional frequency-axis smoothing: turn the single hard bin per rotor into
    # a soft triangular bump so per-bin BCE has a usable gradient.
    if blur_bins > 0:
        k = int(blur_bins)
        ramp = torch.arange(1, k + 1, device=salience.device, dtype=salience.dtype)
        kernel = torch.cat([ramp, torch.tensor([k + 1.0], device=salience.device), ramp.flip(0)])
        kernel = (kernel / kernel.max()).view(1, 1, -1)  # (1, 1, 2k+1), peak 1.0
        # Convolve along the frequency axis: treat each (B, frame) as a 1-D signal.
        sal_f = salience.permute(0, 2, 1).reshape(B * n_hcqt_frames, 1, n_bins)
        sal_f = F.conv1d(sal_f, kernel, padding=k)
        salience = sal_f.reshape(B, n_hcqt_frames, n_bins).permute(0, 2, 1).clamp(0, 1)

    if not was_batched:
        salience = salience.squeeze(0)

    return salience


def salience_to_rps(
    salience: torch.Tensor,
    num_rotors: int = 4,
    *,
    fmin: float = 32.7,
    n_octaves: int = 6,
    over_sample: int = 5,
    threshold: float = 0.0,
    tracking: bool = True,
    top_k_simple: bool = False,
) -> torch.Tensor:
    """Extract RPS trajectories from a salience map.

    Parameters
    ----------
    salience : (B, n_bins, T) or (n_bins, T)
        Salience/activation map (model output).
    num_rotors : int
        Number of rotors to track.
    fmin, n_octaves, over_sample :
        CQT parameters.
    threshold : float
        Minimum activation for a peak to be considered.
    tracking : bool
        If True, use left-to-right tracking (greedy nearest-neighbor
        continuation).  If False, take top-K independent peaks per frame.
    top_k_simple : bool
        If True and tracking=False, use simple top-K (may return the
        same bin for multiple rotors in unison frames).

    Returns
    -------
    rps : (B, num_rotors, T) or (num_rotors, T)
        Reconstructed RPS in Hz.
    """
    was_batched = salience.dim() == 3
    if not was_batched:
        salience = salience.unsqueeze(0)

    B, n_bins, T = salience.shape
    freqs = cqt_freq_grid(fmin=fmin, n_octaves=n_octaves, over_sample=over_sample)

    salience_np = salience.cpu().numpy()

    rps = torch.zeros(B, num_rotors, T, device=salience.device)

    for b in range(B):
        if tracking:
            rps[b] = _track_rotors(salience_np[b], num_rotors, freqs, threshold)
        else:
            rps[b] = _topk_per_frame(salience_np[b], num_rotors, freqs, threshold)

    if not was_batched:
        rps = rps.squeeze(0)

    return rps


def _track_rotors(
    sal: np.ndarray,
    num_rotors: int,
    freqs: np.ndarray,
    threshold: float,
) -> torch.Tensor:
    """Left-to-right greedy tracking of num_rotors peaks.

    Instead of find_peaks (which misses flat regions of equal-valued bins),
    treats any bin with activation > threshold as an active peak.
    """
    n_bins, T = sal.shape
    rps = np.zeros((num_rotors, T), dtype=np.float64)
    current_freqs = np.full(num_rotors, np.nan, dtype=np.float64)

    for t in range(T):
        col = sal[:, t]
        active_bins = np.where(col > threshold)[0]
        active_freqs = freqs[active_bins]
        active_vals = col[active_bins]

        if len(active_bins) == 0:
            # Dark frame: every rotor is stopped. Same convention as
            # ``_hungarian_tracking`` — 0.0 rev/s, no hold-over. ``rps`` is
            # already zero-initialized, thus nothing to write.
            continue

        if t == 0 or np.all(np.isnan(current_freqs)):
            # First frame: pick top-K by activation value, then by frequency order
            order = np.lexsort((active_freqs, -active_vals))
            for i in range(min(num_rotors, len(order))):
                current_freqs[i] = active_freqs[order[i]]
                rps[i, t] = current_freqs[i]
            # If fewer peaks than rotors, duplicate the lowest-frequency peak
            if len(order) < num_rotors and len(order) > 0:
                for i in range(len(order), num_rotors):
                    current_freqs[i] = active_freqs[order[0]]
                    rps[i, t] = current_freqs[i]
        else:
            # Match peaks to existing tracks (nearest-neighbor)
            assigned = np.zeros(len(active_bins), dtype=bool)
            # Sort rotors by frequency so matching is stable
            rotor_order = np.argsort(current_freqs)
            for r in rotor_order:
                if np.isnan(current_freqs[r]):
                    continue
                if not np.any(~assigned):
                    break
                unassigned_freqs = active_freqs[~assigned]
                unassigned_idx = np.where(~assigned)[0]
                dists = np.abs(unassigned_freqs - current_freqs[r])
                best_local = np.argmin(dists)
                best_global = unassigned_idx[best_local]
                current_freqs[r] = active_freqs[best_global]
                rps[r, t] = current_freqs[r]
                assigned[best_global] = True
            # Remaining unassigned peaks → assign to uninitialized rotors
            uninit = np.where(np.isnan(current_freqs))[0]
            unassigned_idx = np.where(~assigned)[0]
            for i, ua in enumerate(unassigned_idx[: len(uninit)]):
                current_freqs[uninit[i]] = active_freqs[ua]
                rps[uninit[i], t] = current_freqs[uninit[i]]
                assigned[ua] = True
            # Rotors that lost their peak: keep previous frequency
            for r in range(num_rotors):
                if np.isnan(current_freqs[r]):
                    continue
                if rps[r, t] == 0.0:
                    rps[r, t] = current_freqs[r]

    return torch.from_numpy(rps).float()


def _topk_per_frame(
    sal: np.ndarray,
    num_rotors: int,
    freqs: np.ndarray,
    threshold: float,
) -> torch.Tensor:
    """Independent top-K per frame (no tracking)."""
    n_bins, T = sal.shape
    rps = np.zeros((num_rotors, T), dtype=np.float64)
    for t in range(T):
        col = sal[:, t]
        top_bins = np.argsort(col)[::-1][:num_rotors]
        for i, b in enumerate(top_bins):
            rps[i, t] = freqs[b]
    return torch.from_numpy(rps).float()


# ═══════════════════════════════════════════════════════════════════════════
# Error analysis
# ═══════════════════════════════════════════════════════════════════════════


def roundtrip_error(
    rps_gt: torch.Tensor,
    n_hcqt_frames: int,
    *,
    num_rotors: int = 4,
    fmin: float = 32.7,
    n_octaves: int = 6,
    over_sample: int = 5,
    hcqt_sr: int = 22050,
    hcqt_hop: int = 256,
    rps_sr: float = 1000.0,
    rps_t_start: float = 0.0,
) -> dict:
    """Measure irreducible loss from RPS ↔ salience quantization.

    Returns dict with keys:
        mae_per_rotor: (num_rotors,) MAE per rotor in Hz
        rmse_per_rotor: (num_rotors,) RMSE per rotor in Hz
        mae_frame: mean absolute error across all frames/rotors
        rmse_frame: RMSE across all frames/rotors
        max_err: maximum single-frame error
        bin_spacing_mean: mean CQT bin spacing at the rotor frequency
    """
    salience = rps_to_salience(
        rps_gt,
        n_hcqt_frames,
        fmin=fmin,
        n_octaves=n_octaves,
        over_sample=over_sample,
        hcqt_sr=hcqt_sr,
        hcqt_hop=hcqt_hop,
        rps_sr=rps_sr,
        rps_t_start=rps_t_start,
    )

    rps_recon_hcqt = salience_to_rps(
        salience,
        num_rotors=num_rotors,
        fmin=fmin,
        n_octaves=n_octaves,
        over_sample=over_sample,
        tracking=True,
    )

    # Resample reconstructed RPS from HCQT grid back to RPS sample times for comparison.
    # Use time-based interpolation.
    was_2d = rps_gt.dim() == 2
    if was_2d:
        rps_gt = rps_gt.unsqueeze(0)
    B, R, T_rps = rps_gt.shape

    # RPS sample times
    rps_times = torch.arange(T_rps, device=rps_gt.device).float() / rps_sr + rps_t_start
    # HCQT frame times
    hcqt_times = (
        (torch.arange(rps_recon_hcqt.shape[-1], device=rps_recon_hcqt.device).float() + 0.5)
        * hcqt_hop
        / hcqt_sr
    )

    # Linear interpolation: HCQT → RPS times
    idx = torch.searchsorted(hcqt_times, rps_times).clamp(1, rps_recon_hcqt.shape[-1] - 1)
    t_left = hcqt_times[idx - 1]
    t_right = hcqt_times[idx]
    w_right = (rps_times - t_left) / (t_right - t_left + 1e-10)
    w_left = 1.0 - w_right

    # rps_recon_hcqt: (B, num_rotors, T_hcqt)
    rps_left = rps_recon_hcqt[:, :, idx - 1]
    rps_right = rps_recon_hcqt[:, :, idx]
    rps_recon = w_left * rps_left + w_right * rps_right  # (B, num_rotors, T_rps)

    err = (rps_recon - rps_gt).abs()
    se = (rps_recon - rps_gt) ** 2

    result = {
        "mae_per_rotor": err.mean(dim=(0, 2)).cpu().numpy(),
        "rmse_per_rotor": se.mean(dim=(0, 2)).sqrt().cpu().numpy(),
        "mae_frame": err.mean().item(),
        "rmse_frame": se.mean().sqrt().item(),
        "max_err": err.max().item(),
    }

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Segment-aware trajectory extraction
# ═══════════════════════════════════════════════════════════════════════════


def _extract_peaks_per_frame(
    salience: np.ndarray,
    freqs: np.ndarray,
    threshold: float = 0.0,
    min_spacing: int = 2,
) -> list:
    """Extract per-frame peaks from a salience map.

    For binary salience (ground truth), every active bin is a peak.
    For soft predictions, local maxima above threshold are used.

    Args:
        salience: (n_bins, T) array
        freqs: (n_bins,) frequency grid
        threshold: minimum activation
        min_spacing: minimum bin spacing between peaks (for soft salience)

    Returns:
        List of length T, each entry is (n_peaks_t,) float array of frequencies.
    """
    n_bins, T = salience.shape
    peaks = []

    for t in range(T):
        col = salience[:, t]
        active = col > threshold

        if not active.any():
            peaks.append(np.array([], dtype=np.float64))
            continue

        # Check if binary (all values are 0 or 1 at active positions)
        vals = col[active]
        is_binary = np.all((vals == 0.0) | (vals == 1.0))

        if is_binary:
            peaks.append(freqs[active])
        else:
            # Soft salience: use local maxima
            local_max = np.zeros(n_bins, dtype=bool)
            for b in range(1, n_bins - 1):
                if col[b] > threshold and col[b] >= col[b - 1] and col[b] >= col[b + 1]:
                    local_max[b] = True
            # Suppress weak neighbors within min_spacing
            lm_bins = np.where(local_max)[0]
            lm_vals = col[lm_bins]
            order = np.argsort(-lm_vals)
            keep = np.ones(len(lm_bins), dtype=bool)
            for i in range(len(order)):
                if not keep[order[i]]:
                    continue
                b_i = lm_bins[order[i]]
                for j in range(i + 1, len(order)):
                    if abs(lm_bins[order[j]] - b_i) < min_spacing:
                        keep[order[j]] = False
            peaks.append(freqs[lm_bins[keep]])

    return peaks


# ═══════════════════════════════════════════════════════════════════════════
# Robust Hungarian-based tracking
# ═══════════════════════════════════════════════════════════════════════════


def _hungarian_tracking(
    peaks_per_frame: list,
    num_rotors: int,
    freqs: np.ndarray,
    max_jump_bins: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Track rotors using optimal (Hungarian) frame-to-frame assignment.

    This replaces the greedy nearest-neighbour assignment with minimum-cost
    bipartite matching, which avoids unnecessary identity swaps that the
    greedy algorithm can introduce when two trajectories are close in
    frequency.

    Parameters
    ----------
    peaks_per_frame :
        List of (n_peaks_t,) frequency arrays per frame.
    num_rotors :
        Number of rotor tracks to maintain.
    freqs :
        (n_bins,) full frequency grid (used for merge detection).
    max_jump_bins :
        Maximum number of CQT bins a rotor is allowed to "jump" between
        consecutive frames.  Assignments beyond this distance are rejected
        and the previous frequency is carried forward.  This suppresses
        tracking jitter caused by adjacent-bin ambiguity.

    Returns
    -------
    rps : (num_rotors, T)
        A frame with no peak above the threshold decodes to 0.0 rev/s for
        every rotor (silence == zero rotor speed), not to a hold-over.
    merge_mask : (T,) bool  — True at frames where two or more active
                 trajectories share the same bin.
    """
    from scipy.optimize import linear_sum_assignment

    K = num_rotors
    T = len(peaks_per_frame)
    rps = np.full((K, T), np.nan, dtype=np.float64)
    current = np.full(K, np.nan, dtype=np.float64)
    merge_mask = np.zeros(T, dtype=bool)

    for t in range(T):
        p_freqs = np.asarray(peaks_per_frame[t], dtype=np.float64)
        n_peaks = len(p_freqs)

        # ── No peaks: the frame is dark, thus every rotor is STOPPED ──
        # Project convention (docs/experiments/honest-base-frontends.md):
        # silence == zero rotor speed. A dark frame decodes to 0.0 rev/s for
        # every rotor — never a hold-over of the last speed and never NaN.
        # ``current`` is NOT cleared, thus a momentary dropout does not reset
        # track identity; only the emitted value is zero.
        if n_peaks == 0:
            rps[:, t] = 0.0
            continue

        # ── First frame or all tracks dead ──
        if t == 0 or np.all(np.isnan(current)):
            order = np.argsort(p_freqs)
            n_use = min(K, n_peaks)
            for i in range(n_use):
                current[i] = p_freqs[order[i]]
                rps[i, t] = current[i]
            for i in range(n_use, K):
                current[i] = p_freqs[order[0]]
                rps[i, t] = current[i]
            continue

        # ── Build cost matrix ──
        active = ~np.isnan(current)
        n_active = active.sum()

        # Cost matrix: (n_active, n_peaks)
        act_freqs = current[active]
        cost = np.full((n_active, n_peaks), 1e12, dtype=np.float64)
        for i in range(n_active):
            for j in range(n_peaks):
                cost[i, j] = abs(act_freqs[i] - p_freqs[j])

        row_ind, col_ind = linear_sum_assignment(cost)

        # ── Apply assignments (with max-jump rejection) ──
        active_indices = np.where(active)[0]

        # Track which peaks are claimed by multiple active rotors
        peak_claimed_by = {}  # peak_idx → [rotor_indices]

        for ri, pj in zip(row_ind, col_ind):
            if cost[ri, pj] >= 1e11:
                continue  # dummy assignment
            orig_r = active_indices[ri]

            # Max-jump check
            dist_bins = abs(freqs - p_freqs[pj]).argmin()
            prev_bin = (
                abs(freqs - current[orig_r]).argmin() if not np.isnan(current[orig_r]) else -1
            )
            if prev_bin >= 0 and abs(dist_bins - prev_bin) > max_jump_bins:
                # Reject: carry forward previous value
                rps[orig_r, t] = current[orig_r]
                continue

            peak_claimed_by.setdefault(int(pj), []).append(orig_r)
            current[orig_r] = p_freqs[pj]
            rps[orig_r, t] = p_freqs[pj]

        # Detect merges: any peak claimed by ≥ 2 rotors, OR
        # more active rotors than available peaks (unison/stripe).
        for _pj, rotors in peak_claimed_by.items():
            if len(rotors) >= 2:
                merge_mask[t] = True
        # Also merge when we have fewer peaks than active rotors
        if n_peaks < n_active and n_peaks > 0:
            merge_mask[t] = True

        # Carry forward unassigned active rotors
        for r in range(K):
            if not np.isnan(current[r]) and np.isnan(rps[r, t]):
                rps[r, t] = current[r]

        # Assign unclaimed peaks to dead rotors
        dead = np.where(np.isnan(current))[0]
        assigned_pj = set(col_ind[cost[row_ind, col_ind] < 1e11])
        unassigned = [j for j in range(n_peaks) if j not in assigned_pj]
        for i, pj in enumerate(unassigned[: len(dead)]):
            current[dead[i]] = p_freqs[pj]
            rps[dead[i], t] = p_freqs[pj]

    return rps, merge_mask


def _detect_adjacent_merges(rps: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Post-hoc detection: frames where two trajectories are in adjacent bins.

    This catches "stripe" merges — trajectories that are close but not
    identical after tracking.  These are ambiguous identity points.
    """
    K, T = rps.shape
    mask = np.zeros(T, dtype=bool)
    for t in range(T):
        f = rps[:, t]
        valid = ~np.isnan(f)
        if valid.sum() < 2:
            continue
        fv = f[valid]
        bins = np.array([abs(freqs - fv[i]).argmin() for i in range(len(fv))], dtype=int)
        for i in range(len(bins)):
            for j in range(i + 1, len(bins)):
                if abs(bins[i] - bins[j]) <= 1:
                    mask[t] = True
                    break
            if mask[t]:
                break
    return mask


def salience_to_rps_segmented(
    salience: torch.Tensor,
    num_rotors: int = 4,
    *,
    fmin: float = 32.7,
    n_octaves: int = 6,
    over_sample: int = 5,
    n_bins: int | None = None,
    bins_per_octave: int | None = None,
    freqs: np.ndarray | None = None,
    threshold: float = 0.0,
    max_jump_bins: int = 3,
    merge_mode: str = "same_bin",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract RPS trajectories with merge-point detection via Hungarian tracking.

    Args:
        salience: (B, n_bins, T) or (n_bins, T) salience map.
        num_rotors: Number of rotor trajectories to track.
        fmin, n_octaves, over_sample: CQT parameters.
        n_bins, bins_per_octave: explicit grid override (non-integer-octave
            grids, e.g. Basic Pitch contour). When given, the frequency grid is
            taken from these and ``n_octaves``/``over_sample`` are ignored.
        threshold: Minimum activation for peak detection (0.0 for binary).
        max_jump_bins: Max CQT bins a rotor can jump between frames.
        merge_mode: ``"same_bin"`` (default) — only exact same-bin collisions
                    are merge points.  ``"adjacent"`` — same or adjacent bins.

    Returns:
        rps: (B, num_rotors, T) or (num_rotors, T) reconstructed RPS (Hz).
            Frames with no peak above ``threshold`` decode to 0.0 for every
            rotor — the project-wide silence == zero-rotor-speed convention.
        merge_mask: (B, T) or (T,) bool — merge-point frames.
    """
    was_batched = salience.dim() == 3
    if not was_batched:
        salience = salience.unsqueeze(0)

    B, _n_bins, T = salience.shape
    # Explicit ``freqs`` (e.g. a linear output grid) overrides the geometric grid.
    if freqs is None:
        freqs = cqt_freq_grid(
            fmin=fmin,
            n_octaves=n_octaves,
            over_sample=over_sample,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
        )
    else:
        freqs = np.asarray(freqs, dtype=np.float64)

    salience_np = salience.cpu().numpy()

    rps_list, mask_list = [], []
    for b in range(B):
        peaks = _extract_peaks_per_frame(salience_np[b], freqs, threshold)
        rps_np, merge_np = _hungarian_tracking(
            peaks, num_rotors, freqs, max_jump_bins=max_jump_bins
        )

        if merge_mode == "adjacent":
            adj_merge = _detect_adjacent_merges(rps_np, freqs)
            merge_np = merge_np | adj_merge

        rps_list.append(torch.from_numpy(rps_np).float().to(salience.device))
        mask_list.append(torch.from_numpy(merge_np).to(salience.device))

    rps = torch.stack(rps_list, dim=0)
    merge_mask = torch.stack(mask_list, dim=0)

    if not was_batched:
        rps = rps.squeeze(0)
        merge_mask = merge_mask.squeeze(0)

    return rps, merge_mask


# ═══════════════════════════════════════════════════════════════════════════
# Segment definitions and segmented PIT loss
# ═══════════════════════════════════════════════════════════════════════════


def _segment_boundaries(merge_mask: np.ndarray) -> list[tuple[int, int]]:
    """Convert a frame-level merge mask into (start, end) segment indices.

    A segment is a contiguous range of frames that contains NO merge points.
    Merge points themselves form 1-frame "segments" where identity is lost.

    Returns list of (start, end) inclusive.
    """
    T = len(merge_mask)
    if T == 0:
        return []

    segments = []
    start = 0
    for t in range(T):
        if merge_mask[t]:
            if t > start:
                segments.append((start, t - 1))
            segments.append((t, t))
            start = t + 1
    if start < T:
        segments.append((start, T - 1))

    return segments


def segmented_pit_mse(
    rps_pred: torch.Tensor,
    rps_gt: torch.Tensor,
    merge_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Segment-based PIT-MSE loss.

    Unlike ``pit_mse_loss`` in ``train_rps_predictor.py``, which finds
    a single global permutation, this function splits the timeline at merge
    points and finds the best permutation **independently within each
    segment**.

    Args:
        rps_pred: (B, K, T) predicted RPS.
        rps_gt: (B, K, T) ground-truth RPS.
        merge_mask: (B, T) bool, or None.  If None, degenerates to global
            PIT (one segment covering all frames).

    Returns:
        Scalar MSE loss.
    """
    from itertools import permutations

    B, K, T = rps_pred.shape
    device = rps_pred.device
    perms = list(permutations(range(K)))

    if merge_mask is None:
        merge_mask = torch.zeros(B, T, dtype=torch.bool, device=device)

    total_se = torch.tensor(0.0, device=device)
    total_count = 0

    for b in range(B):
        merge_np = merge_mask[b].cpu().numpy()
        segments = _segment_boundaries(merge_np)

        for start, end in segments:
            pred_seg = rps_pred[b, :, start : end + 1]
            gt_seg = rps_gt[b, :, start : end + 1]

            best_err = float("inf")
            for perm in perms:
                err = ((pred_seg - gt_seg[perm, :]) ** 2).sum()
                if err < best_err:
                    best_err = err

            total_se = total_se + best_err
            total_count = total_count + (end - start + 1)

    return total_se / (total_count * K)


def roundtrip_error_segmented(
    rps_gt: torch.Tensor,
    n_hcqt_frames: int,
    *,
    num_rotors: int = 4,
    fmin: float = 32.7,
    n_octaves: int = 6,
    over_sample: int = 5,
    hcqt_sr: int = 22050,
    hcqt_hop: int = 256,
    rps_sr: float = 1000.0,
    rps_t_start: float = 0.0,
) -> dict:
    """Round-trip error using *segment-based* PIT matching.

    Same pipeline as :func:`roundtrip_error` but the RPS reconstruction
    uses the Hungarian-based tracking and the error is computed via
    :func:`segmented_pit_mse`.
    """
    salience = rps_to_salience(
        rps_gt,
        n_hcqt_frames,
        fmin=fmin,
        n_octaves=n_octaves,
        over_sample=over_sample,
        hcqt_sr=hcqt_sr,
        hcqt_hop=hcqt_hop,
        rps_sr=rps_sr,
        rps_t_start=rps_t_start,
    )

    rps_recon_hcqt, merge_mask = salience_to_rps_segmented(
        salience,
        num_rotors=num_rotors,
        fmin=fmin,
        n_octaves=n_octaves,
        over_sample=over_sample,
        threshold=0.0,
    )

    was_2d = rps_gt.dim() == 2
    if was_2d:
        rps_gt = rps_gt.unsqueeze(0)
    B, R, T_rps = rps_gt.shape

    rps_times = torch.arange(T_rps, device=rps_gt.device).float() / rps_sr + rps_t_start
    hcqt_times = (
        (torch.arange(rps_recon_hcqt.shape[-1], device=rps_recon_hcqt.device).float() + 0.5)
        * hcqt_hop
        / hcqt_sr
    )

    idx = torch.searchsorted(hcqt_times, rps_times).clamp(1, rps_recon_hcqt.shape[-1] - 1)
    t_left = hcqt_times[idx - 1]
    t_right = hcqt_times[idx]
    w_right = (rps_times - t_left) / (t_right - t_left + 1e-10)

    rps_left = rps_recon_hcqt[:, :, idx - 1]
    rps_right = rps_recon_hcqt[:, :, idx]
    rps_recon = (1.0 - w_right) * rps_left + w_right * rps_right

    seg_mse = segmented_pit_mse(rps_recon, rps_gt, merge_mask)

    err = (rps_recon - rps_gt).abs()
    se = (rps_recon - rps_gt) ** 2

    return {
        "rmse_segmented": seg_mse.sqrt().item(),
        "rmse_global": se.mean().sqrt().item(),
        "mae_frame": err.mean().item(),
        "max_err": err.max().item(),
        "n_merge_frames": merge_mask.sum().item(),
    }
