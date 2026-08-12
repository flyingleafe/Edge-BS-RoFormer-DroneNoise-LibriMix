"""One place to drive every drone-noise generator we have.

Replaces the scattered per-model notebooks (`drone_embedding_explorer`,
`noise_gen_real_vs_generated`, `noise_four_way_comparison`,
`jasa_gp_interactive`), several of which no longer import — they still
reach for `data_processing.dregon` / `.michaels`, which the data-layer refactor
moved into `data_processing.sources`.

Three things a generator needs, and this module supplies each once:

1. **An excitation** — either a real recording's RPS (with its audio alongside,
   so you can compare) or a synthetic trajectory. :func:`real_slice` and
   :func:`synth_slice` both return the same :class:`Excitation`, so everything
   downstream is indifferent to where it came from.
2. **A model** — :data:`VARIANTS` names every generator family and checkpoint.
   :func:`load_variant` caches, because most of these are slow to build.
3. **A render** — :func:`render` dispatches on family and returns ``(M, T)``
   audio at :data:`SR`, whatever the model's native rate or interface.

The families are genuinely different kinds of object, and it is worth keeping
that straight:

- ``deep`` — our learned generator (`PositionalHarmonicNoiseGen`, optionally
  with the additive wind channel). Conditioned on a per-drone code, so it is the
  only family with an interpolatable embedding.
- ``gp`` — the JASA Gaussian-process rotor model, a fitted statistical field.
- ``cona`` — constant-RPS auralized cases from the published `drone-egonoise`
  set. Not a model we fit; a reference synthesis.
- ``real`` — the recording itself.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

SR = 16_000
Family = Literal["deep", "gp", "cona", "real"]


# ─── Variant registry ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Variant:
    """One selectable generator."""

    name: str
    family: Family
    note: str
    experiment: str | None = None
    checkpoint: str | None = None
    build_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def conditioned(self) -> bool:
        """Whether it takes a per-drone code (and so has an embedding to move)."""
        return bool(self.build_kwargs.get("cond_dim", 0))


_DEEP_PERDRONE = dict(
    model_name="positional_harmonic_gen",
    sample_rate=SR,
    n_harmonics=100,
    cond_dim=16,
    drone_names=["dregon", "michaels"],
    rps_jitter_sigma=0.6,
    rps_jitter_tau=0.016,
    learn_rps_jitter_sigma=True,
    z_noise_std=0.1,
    film_spectral_norm=True,
)
_DEEP_WIND = {**_DEEP_PERDRONE, "model_name": "positional_harmonic_wind_gen"}
#: The DREGON-only label-A/B arms (docs/experiments/generator-refined-labels.md)
#: train with a one-name codebook; r2 adds the per-rotor deltas.
_DEEP_DREGON = {**_DEEP_PERDRONE, "drone_names": ["dregon"]}
_DEEP_DREGON_PERROTOR = {**_DEEP_DREGON, "per_rotor_deltas": True, "n_rotors": 4}


def _r2(exp: str) -> str:
    return f"r2://ml-data/artifacts/{exp}/checkpoints/best.ckpt"


def _r2_file(exp: str, filename: str) -> str:
    """A SPECIFIC checkpoint file — for comb-aware epoch picks, where
    ``best.ckpt`` (best-by-mrstft) is the wrong draw."""
    return f"r2://ml-data/artifacts/{exp}/checkpoints/{filename}"


#: Every generator worth comparing. Keys are what the notebook's picker shows.
VARIANTS: dict[str, Variant] = {
    "real": Variant("real", "real", "the recording itself"),
    # ── learned generators, in the order they were produced ──────────────────
    "deep/e6-perdrone (old geometry)": Variant(
        "deep/e6-perdrone (old geometry)",
        "deep",
        "E6 winner: per-drone code + learned jitter sigma. Trained BEFORE the "
        "180-degree DREGON mic-frame fix, so its geometry is wrong.",
        "e6_noisegen_jitter_latreg_perdrone",
        _r2("e6_noisegen_jitter_latreg_perdrone"),
        _DEEP_PERDRONE,
    ),
    "deep/v1 corrected-geometry": Variant(
        "deep/v1 corrected-geometry",
        "deep",
        "Same architecture on corrected geometry. SINGLE-MIC trained.",
        "gen_v1_corrected",
        _r2("gen_v1_corrected"),
        _DEEP_PERDRONE,
    ),
    "deep/v1 recalibrated labels": Variant(
        "deep/v1 recalibrated labels",
        "deep",
        "v1 on the measured Michael's telemetry calibration. Single-mic.",
        "gen_v1_recal",
        _r2("gen_v1_recal"),
        _DEEP_PERDRONE,
    ),
    "deep/v1 recal, 8 mics": Variant(
        "deep/v1 recal, 8 mics",
        "deep",
        "First multi-observer training run; magnitude loss.",
        "gen_v1_recal_mm",
        _r2("gen_v1_recal_mm"),
        _DEEP_PERDRONE,
    ),
    "deep/likelihood, 8 mics": Variant(
        "deep/likelihood, 8 mics",
        "deep",
        "Rice/Whittle spectral likelihood instead of the magnitude loss. Best held-out fit so far.",
        "gen_w3_lik_nowind_mm",
        _r2("gen_w3_lik_nowind_mm"),
        _DEEP_PERDRONE,
    ),
    "deep/likelihood + wind": Variant(
        "deep/likelihood + wind",
        "deep",
        "Adds the additive wind-wake channel. Under the MARGINAL likelihood it "
        "trains to inertness (~0.1% of predicted power).",
        "gen_w4_lik_wind_mm",
        _r2("gen_w4_lik_wind_mm"),
        _DEEP_WIND,
    ),
    "deep/spatial + wind": Variant(
        "deep/spatial + wind",
        "deep",
        "Wind channel under the cross-microphone (array-covariance) likelihood, "
        "the objective that can actually see incoherence.",
        "gen_s2_spatial_wind",
        _r2("gen_s2_spatial_wind"),
        _DEEP_WIND,
    ),
    "deep/spatial + wind, uniform": Variant(
        "deep/spatial + wind, uniform",
        "deep",
        "Wake-model control: same capacity, geometry gate removed (every mic "
        "gets the array-mean flow speed).",
        "gen_s3_spatial_uniform",
        _r2("gen_s3_spatial_uniform"),
        {**_DEEP_WIND, "wind_uniform_exposure": True},
    ),
    # ── the DREGON-only label A/B (refined-telemetry campaign, 2026-08-12) ──
    "deep/r1 DREGON-only, orig labels": Variant(
        "deep/r1 DREGON-only, orig labels",
        "deep",
        "Label-A/B control: v1-recal-mm setup, DREGON only, raw telemetry. "
        "No measurable tooth above k~22.",
        "gen_r1_orig",
        _r2("gen_r1_orig"),
        _DEEP_DREGON,
    ),
    "deep/r1 DREGON-only, scaled labels": Variant(
        "deep/r1 DREGON-only, scaled labels",
        "deep",
        "Telemetry x 0.99458 (the phase-7 constant fix). Sharp mid-k lines, "
        "but off the true comb; dies above k~50. Drive with labels='scaled'.",
        "gen_r1_scaled",
        _r2("gen_r1_scaled"),
        _DEEP_DREGON,
    ),
    "deep/r1 DREGON-only, refined labels": Variant(
        "deep/r1 DREGON-only, refined labels",
        "deep",
        "L-BFGS-refined trajectories (sidecar labels). The only arm with teeth "
        "above the estimator null through k=80. Drive with labels='refined'.",
        "gen_r1_refined",
        _r2("gen_r1_refined"),
        _DEEP_DREGON,
    ),
    "deep/r2 + per-rotor dz, orig labels": Variant(
        "deep/r2 + per-rotor dz, orig labels",
        "deep",
        "r1-orig + per-rotor sub-embeddings z_r = z_drone + dz_r. The deltas "
        "HURT line sharpness in both label conditions.",
        "gen_r2_orig_perrotor",
        _r2("gen_r2_orig_perrotor"),
        _DEEP_DREGON_PERROTOR,
    ),
    "deep/r2 + per-rotor dz, refined labels": Variant(
        "deep/r2 + per-rotor dz, refined labels",
        "deep",
        "r1-refined + per-rotor sub-embeddings. Weaker combs than r1-refined. "
        "Drive with labels='refined'.",
        "gen_r2_refined_perrotor",
        _r2("gen_r2_refined_perrotor"),
        _DEEP_DREGON_PERROTOR,
    ),
    # ── full-dataset refined-label arms (comb-aware checkpoint selection) ───
    "deep/m1 full data, refined labels": Variant(
        "deep/m1 full data, refined labels",
        "deep",
        "DREGON+Michael's, refined DREGON labels, per-drone codebook. "
        "Comb-best epoch (ep0); mixed-drone training still costs DREGON "
        "mid-k sharpness. Drive with labels='refined' on DREGON.",
        "gen_m1_refined",
        _r2_file("gen_m1_refined", "ep0_mrstft_3.0321.ckpt"),
        _DEEP_PERDRONE,
    ),
    "deep/m2 full data + per-rotor dz, refined": Variant(
        "deep/m2 full data + per-rotor dz, refined",
        "deep",
        "m1 + per-rotor deltas, comb-best epoch (ep14): the best high-k combs "
        "of any arm (k50-80 +1.05 dB) — with two drones the deltas earn their "
        "keep. Drive with labels='refined' on DREGON.",
        "gen_m2_refined_perrotor",
        _r2_file("gen_m2_refined_perrotor", "ep14_mrstft_2.1149.ckpt"),
        {**_DEEP_PERDRONE, "per_rotor_deltas": True, "n_rotors": 4},
    ),
    # ── non-learned references ──────────────────────────────────────────────
    "gp/JASA rotor field": Variant(
        "gp/JASA rotor field", "gp", "Fitted Gaussian-process rotor-noise field."
    ),
    "cona/constant-RPS auralization": Variant(
        "cona/constant-RPS auralization",
        "cona",
        "Published drone-egonoise constant-RPS cases. Nearest case to the "
        "requested mean RPS is used.",
    ),
}


# ─── Excitation: real slice or synthetic ──────────────────────────────────────


@dataclass
class Excitation:
    """What every generator is driven by, plus the reference audio if there is one."""

    rps: np.ndarray  # (R, T) rev/s at audio rate
    mic_pos: np.ndarray  # (M, 3) metres, body frame
    rotor_pos: np.ndarray  # (R, 3)
    drone: str  # codebook key: "dregon" | "michaels"
    audio: np.ndarray | None = None  # (M, T) real recording, when there is one
    label: str = ""

    @property
    def duration_s(self) -> float:
        return self.rps.shape[-1] / SR

    @property
    def mean_rps(self) -> float:
        return float(self.rps.mean())


@lru_cache(maxsize=4)
def _frames(dataset: str) -> list:
    from data_processing.streams import iter_published_frames

    return list(iter_published_frames(dataset, None))


def recordings(dataset: str, usable_only: bool = True) -> list[str]:
    """Recording ids in a published frames dataset.

    ``usable_only`` (the default) drops recordings that cannot drive a
    generator — DREGON ships clean-source captures like
    ``clean_chirps_45_-15_1.2`` that carry audio but **no rotor track**. Listing
    them would put entries in the notebook's dropdown that fail on selection.
    """
    from data_processing.frames import PUBLISHED_RPS_KEYS, get_meta

    out = []
    for f in _frames(dataset):
        if usable_only and not ("audio" in f and any(k in f for k in PUBLISHED_RPS_KEYS)):
            continue
        out.append(str(get_meta(f, "recording_id", "?")))
    return out


DATASETS = {"DREGON-frames": "dregon", "michaels-frames": "michaels"}


def real_slice(
    dataset: str,
    recording_id: str,
    start_s: float,
    dur_s: float = 4.0,
    labels: str = "telemetry",
) -> Excitation:
    """A window of a real recording: audio, RPS and geometry, all aligned.

    The rotor track is resolved through :func:`data_processing.frames.
    adapt_recording_frame`, which is the project's canonical path: DREGON frames
    carry `motors_command` / `motors_measured` while Michael's carry `rps`, and
    that helper picks the right one *in preference order* and resamples the audio
    in the same step. Reaching for a fixed entry name here — as an earlier
    version of this function did — silently breaks on one of the two rigs.

    ``labels`` selects the conditioning source, matching the training arms of
    the label A/B (docs/experiments/generator-refined-labels.md):
    ``"telemetry"`` (raw), ``"scaled"`` (x 0.99458), or ``"refined"`` (the
    L-BFGS sidecar in ``src/data_processing/refined_labels/`` — DREGON only;
    samples outside the sidecar span keep telemetry). Drive each generator
    with the labels it trained on.
    """
    from data_processing.frames import adapt_recording_frame, get_meta

    for frame in _frames(dataset):
        if str(get_meta(frame, "recording_id", "?")) != recording_id:
            continue
        # Geometry lives on the RAW frame; adapt_recording_frame drops it.
        mic_pos = np.asarray(frame["mic_pos"].data, dtype=np.float32)
        rotor_pos = np.asarray(frame["rotor_pos"].data, dtype=np.float32)

        adapted = adapt_recording_frame(frame, sample_rate=SR)
        if adapted is None:
            raise ValueError(f"{recording_id} has no audio or no rotor track")

        audio_s = adapted["audio"]
        total = audio_s.data.shape[-1] / SR
        dur = float(min(dur_s, total))
        start = float(np.clip(start_s, 0.0, max(total - dur, 0.0)))
        t0 = audio_s.t_start + start
        window = adapted.time[t0 : t0 + dur]

        audio = np.asarray(window["audio"].data, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[None, :]
        rps = np.asarray(window["rps"].data, dtype=np.float32)
        if rps.ndim == 1:
            rps = rps[None, :]

        # The rotor track is event-sampled telemetry; lift it onto the audio grid.
        n = audio.shape[-1]
        src = np.linspace(0.0, 1.0, rps.shape[-1])
        dst = np.linspace(0.0, 1.0, n)
        rps_audio = np.stack([np.interp(dst, src, row) for row in rps]).astype(np.float32)

        if labels == "scaled":
            rps_audio = rps_audio * np.float32(0.99458)
        elif labels == "refined":
            import tdseries as td

            sidecar = _ROOT / "src" / "data_processing" / "refined_labels" / f"{recording_id}.npz"
            if not sidecar.is_file():
                raise FileNotFoundError(
                    f"no refined sidecar for {recording_id!r} at {sidecar} — "
                    "refined labels exist only for the DREGON generator recordings"
                )
            with np.load(sidecar) as z:
                ft, r_refined = np.asarray(z["ft"]), np.asarray(z["r_refined"])
            # Offsets from the FULL published frame's audio t_start — the
            # sidecar's time reference — computed tick-exactly.
            base = (window["audio"].t_start_ticks - frame["audio"].t_start_ticks) / float(
                td.TICKS_PER_SECOND
            )
            offs = base + np.arange(n, dtype=np.float64) / SR
            inside = (offs >= ft[0]) & (offs <= ft[-1])
            for r in range(min(rps_audio.shape[0], r_refined.shape[0])):
                rps_audio[r, inside] = np.maximum(
                    np.interp(offs[inside], ft, r_refined[r]), 0.0
                ).astype(np.float32)
        elif labels != "telemetry":
            raise ValueError(f"labels must be telemetry|scaled|refined, got {labels!r}")

        tag = "" if labels == "telemetry" else f", {labels} labels"
        return Excitation(
            rps=rps_audio,
            mic_pos=mic_pos,
            rotor_pos=rotor_pos,
            drone=DATASETS[dataset],
            audio=audio,
            label=f"{recording_id} @ {start:.1f}s +{dur:.1f}s{tag}",
        )
    raise KeyError(f"recording {recording_id!r} not in {dataset}")


def synth_slice(
    drone: str = "dregon",
    kind: Literal["intermittent", "full_flight"] = "intermittent",
    dur_s: float = 4.0,
    seed: int = 0,
    aggressiveness: float = 1.0,
) -> Excitation:
    """A synthetic RPS trajectory with a real rig's geometry and no reference audio."""
    from data_processing import rps_synthesis

    rng = np.random.default_rng(seed)
    if kind == "full_flight":
        # The full-flight trajectory walks a fixed phase sequence (ground ->
        # warm-up -> takeoff -> cruise -> landing -> ground) with a hard minimum
        # duration. Ask for something shorter and it raises; so synthesize at
        # the natural length and return the requested window from the middle,
        # which is where cruise lives.
        full = np.asarray(rps_synthesis.generate_full_flight(None, SR, rng=rng), np.float32)
        want = int(round(dur_s * SR))
        if full.shape[-1] > want:
            start = (full.shape[-1] - want) // 2
            full = full[..., start : start + want]
        rps = full
    else:
        rps = rps_synthesis.generate_intermittent(dur_s, SR, rng=rng, aggressiveness=aggressiveness)
    rps = np.asarray(rps, dtype=np.float32)
    if rps.ndim == 1:
        rps = np.repeat(rps[None, :], 4, axis=0)
    mic, rotor = geometry_for(drone)
    return Excitation(
        rps=rps,
        mic_pos=mic,
        rotor_pos=rotor,
        drone=drone,
        audio=None,
        label=f"synthetic {kind} (seed {seed})",
    )


@lru_cache(maxsize=4)
def geometry_for(drone: str) -> tuple[np.ndarray, np.ndarray]:
    """Published rig geometry — corrected DREGON frame, horizontal Michael's ring."""
    from data_processing import sources
    from data_processing.frame_datasets import _frames_spec_geometry

    if drone == "dregon":
        mic, rotor = _frames_spec_geometry("frames:DREGON-frames")
    else:
        mic, rotor = sources.geometry("michaels")
    return np.asarray(mic, np.float32), np.asarray(rotor, np.float32)


# ─── Model loading + rendering ────────────────────────────────────────────────


@lru_cache(maxsize=8)
def load_variant(name: str):
    """Build a variant's model. Cached — most of these are slow to construct."""
    import torch

    from models.registry import build_noise_gen_model
    from training.artifacts import resolve_checkpoint_uri

    spec = VARIANTS[name]
    if spec.family != "deep":
        raise ValueError(f"{name} is family {spec.family!r}; use render() directly")
    kwargs = dict(spec.build_kwargs)
    model = build_noise_gen_model(**kwargs)
    if spec.checkpoint:
        state = torch.load(resolve_checkpoint_uri(spec.checkpoint), map_location="cpu")
        for key in ("state_dict", "model"):
            if isinstance(state, dict) and key in state:
                state = state[key]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(missing) > 20:
            raise RuntimeError(f"{name}: {len(missing)} missing keys — wrong build kwargs?")
    model.eval()
    return model


def drone_code(model, drone: str):
    """The learned embedding for a drone, or None if the model is unconditioned."""
    book = getattr(model, "codebook", None)
    if book is None:
        return None
    import torch

    with torch.no_grad():
        return book([drone])[0].clone()


def interpolate_code(model, alpha: float, offset: float = 0.0):
    """A point on the DREGON->Michael's embedding axis.

    ``alpha`` 0 is DREGON, 1 is Michael's, and values outside [0, 1] extrapolate.
    ``offset`` pushes off that axis along a fixed orthogonal direction, so you
    can ask what the decoder does *around* the two codes it was trained on —
    the region nothing in the data constrains.
    """
    import torch

    z0, z1 = drone_code(model, "dregon"), drone_code(model, "michaels")
    if z0 is None:
        return None
    axis = z1 - z0
    z = z0 + float(alpha) * axis
    if offset:
        perp = torch.zeros_like(axis)
        perp[0] = axis[1]
        perp[1] = -axis[0]
        n = torch.linalg.vector_norm(perp).clamp_min(1e-8)
        z = z + float(offset) * axis.norm() * perp / n
    return z


def render(
    name: str,
    exc: Excitation,
    *,
    alpha: float | None = None,
    offset: float = 0.0,
    jitter_sigma: float | None = None,
    wind: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Render one variant against an excitation. Returns ``(M, T)`` at :data:`SR`.

    ``alpha``/``offset`` move the per-drone embedding (deep family only);
    ``alpha=None`` uses the drone's own learned code. ``jitter_sigma`` overrides
    the learned RPS-jitter linewidth. ``wind=False`` zeroes the wind channel on
    variants that have one, which is the cheapest way to see what it contributes.
    """
    spec = VARIANTS[name]
    if spec.family == "real":
        if exc.audio is None:
            raise ValueError("this excitation is synthetic — there is no real audio")
        return exc.audio
    if spec.family == "deep":
        return _render_deep(
            spec, exc, alpha=alpha, offset=offset, jitter_sigma=jitter_sigma, wind=wind, seed=seed
        )
    if spec.family == "gp":
        return _render_gp(exc)
    if spec.family == "cona":
        return _render_cona(exc)
    raise ValueError(f"unknown family {spec.family!r}")


def _render_deep(spec, exc, *, alpha, offset, jitter_sigma, wind, seed):
    from typing import cast

    import torch
    from torch import nn

    from models.generative.codebook import geometry_to_rel_pos

    model = load_variant(spec.name)
    torch.manual_seed(seed)
    rps = torch.from_numpy(exc.rps).unsqueeze(0)
    mic = torch.from_numpy(exc.mic_pos).unsqueeze(0)
    rotor = torch.from_numpy(exc.rotor_pos).unsqueeze(0)
    rel = geometry_to_rel_pos(mic, rotor)

    gen = cast(nn.Module, getattr(model, "generator", model))
    level: nn.Parameter | None = None
    saved = None
    if not wind and hasattr(gen, "wind"):
        level = cast(nn.Parameter, gen.get_submodule("wind.transduction").raw_level)
        saved = level.detach().clone()
        with torch.no_grad():
            level.fill_(-30.0)  # softplus -> ~0
    try:
        with torch.no_grad():
            kwargs: dict[str, Any] = {}
            if jitter_sigma is not None:
                kwargs["rps_jitter"] = True
                kwargs["rps_jitter_sigma"] = torch.tensor([float(jitter_sigma)])
            z = interpolate_code(model, alpha, offset) if alpha is not None else None
            if spec.conditioned and z is not None:
                out = gen(rps, rel, z=z.unsqueeze(0), **kwargs)
            elif spec.conditioned:
                out = model(rps, rel, [exc.drone], **kwargs)
            else:
                out = gen(rps, rel, **kwargs)
    finally:
        if saved is not None and level is not None:
            with torch.no_grad():
                level.copy_(saved)
    audio = out["audio"] if isinstance(out, dict) else out
    return audio.squeeze(0).cpu().numpy().astype(np.float32)


def _render_gp(exc: Excitation) -> np.ndarray:
    """JASA GP field. Takes body-frame mic positions and a rotor-MEAN trajectory
    at SR, and returns ``(M, T)`` itself — it does its own 44.1 kHz round trip."""
    from four_way_lib import load_gp, render_gp

    ckpt = _ROOT / "results" / "jasa_gp" / "gp.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"no GP checkpoint at {ckpt} — see the GP report to fit one")
    return np.asarray(render_gp(load_gp(ckpt), exc.mic_pos, exc.rps.mean(axis=0)), dtype=np.float32)


def _render_cona(exc: Excitation) -> np.ndarray:
    """Nearest published constant-RPS auralization, tiled to the clip length.

    CONA cases are single-microphone constant-RPS renders, so this is a
    reference synthesis rather than a per-mic prediction: the same waveform is
    repeated across the array.
    """
    from four_way_lib import cona_inventory, fetch_cona_case, nearest_cona_key, resample_to_sr

    inv = cona_inventory()
    drones = sorted({m["drone"] for m in inv.values()})
    if exc.drone not in drones:
        raise LookupError(
            f"the published CONA set has no cases for drone={exc.drone!r} "
            f"(available: {drones}) — it is a reference synthesis, not a model "
            f"that can be driven from arbitrary geometry"
        )
    key = nearest_cona_key(inv, exc.drone, float(exc.mean_rps), 0)
    cache = _ROOT / ".cache" / "cona"
    cache.mkdir(parents=True, exist_ok=True)
    case = fetch_cona_case(key, cache)
    mono = np.asarray(case["audio"] if isinstance(case, dict) else case, np.float32).reshape(-1)
    mono = resample_to_sr(mono)
    n = exc.rps.shape[-1]
    mono = np.resize(mono, n) if mono.size else np.zeros(n, np.float32)
    return np.repeat(mono[None, :].astype(np.float32), exc.mic_pos.shape[0], axis=0)


# ─── Analysis helpers ─────────────────────────────────────────────────────────


def spectrogram(x: np.ndarray, n_fft: int = 1024, hop: int = 256) -> tuple:
    """``(dB spectrogram (F, N), freqs (F,), times (N,))`` of one channel."""
    import librosa

    x = np.asarray(x, np.float32).reshape(-1)
    spec = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop))
    db = librosa.amplitude_to_db(spec, ref=1.0, top_db=100.0)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(db.shape[1]), sr=SR, hop_length=hop)
    return db, freqs, times


def spectrum_at(db: np.ndarray, times: np.ndarray, t: float) -> np.ndarray:
    """The single spectrum column nearest time ``t`` — what the slider reads out."""
    return db[:, int(np.argmin(np.abs(times - float(t))))]


def match_rms(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Scale ``x`` to ``ref``'s RMS.

    Level and spectral shape are different claims, and most of these families do
    not agree on absolute gain at all (CONA and the GP carry their own
    calibration). Matching level first makes the *shape* comparison honest;
    when you care about level, compare the numbers this returns instead.
    """
    rx = float(np.sqrt(np.mean(np.square(x)))) or 1.0
    rr = float(np.sqrt(np.mean(np.square(ref))))
    return (x * (rr / rx)).astype(np.float32)


def player(x: np.ndarray, normalize: bool = True):
    """An audio widget for one channel."""
    from IPython.display import Audio

    return Audio(np.asarray(x, np.float32).reshape(-1), rate=SR, normalize=normalize)
