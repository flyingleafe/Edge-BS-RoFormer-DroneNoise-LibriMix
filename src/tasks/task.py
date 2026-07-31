"""Task = the function type of a model over :class:`tdseries.Frame`s.

A task names the entries a model consumes and produces, as
:class:`~tasks.spec.FrameSpec`s over *batched* Frames (leading ``"batch"``
dim). Because concrete dims vary per experiment (mono vs multichannel
audio, rotor count, frame rate), tasks are built by **factories** that take
those parameters from the model config; the factory name is the task name
recorded in configs and wandb tags.

See docs/refactor-unified-framework.md § "Task typing: FrameSpec".
"""

from __future__ import annotations

from dataclasses import dataclass

from tasks.spec import FrameSpec, SeriesSpec

AUDIO_RATE = (16000, 1)


@dataclass(frozen=True)
class Task:
    name: str
    input_spec: FrameSpec
    output_spec: FrameSpec


def _audio_dims(n_channels: int | None) -> tuple[str | None, ...]:
    """Batched audio dims: mono ``(batch, time)``; else ``(batch, mic, time)``."""
    return ("batch", "time") if n_channels is None else ("batch", "mic", "time")


def _rps_spec(rate: tuple[int, int] | None) -> SeriesSpec:
    return SeriesSpec(dims=("batch", "rotor", "time"), time="grid", rate=rate)


def speech_enhancement(
    *,
    n_channels: int | None = None,
    use_rps: bool = False,
    predict_rps: bool = False,
    sr: tuple[int, int] = AUDIO_RATE,
    rps_rate: tuple[int, int] | None = None,
) -> Task:
    """Noisy mixture in, enhanced speech out; optional RPS conditioning
    input and auxiliary RPS prediction output."""
    audio = SeriesSpec(dims=_audio_dims(n_channels), time="grid", rate=sr)
    inputs: dict[str, SeriesSpec | FrameSpec] = {"mixture": audio}
    in_optional: set[str] = set()
    if use_rps:
        inputs["rps"] = _rps_spec(rps_rate)
    outputs: dict[str, SeriesSpec | FrameSpec] = {"enhanced": audio}
    out_optional: set[str] = set()
    if predict_rps:
        outputs["rps_pred"] = _rps_spec(rps_rate)
    return Task(
        name="speech_enhancement",
        input_spec=FrameSpec(inputs, frozenset(in_optional)),
        output_spec=FrameSpec(outputs, frozenset(out_optional)),
    )


def rps_prediction(
    *,
    n_channels: int | None = None,
    sr: tuple[int, int] = AUDIO_RATE,
    frame_rate: tuple[int, int] | None = None,
    use_cond: bool = False,
) -> Task:
    """Drone-noise mixture in, per-rotor rotation speeds out (on the STFT
    frame grid — pass ``frame_rate=(sr, hop)`` for the exact rate).

    ``use_cond=True`` is the conditional-*refiner* variant: the model
    additionally consumes ``rps_cond`` — a coarse/corrupted RPS track on the
    same frame grid — and its output rows follow the conditioning's rotor
    order (so the training loss is plain, non-PIT — ``losses.RPSMSELoss``).
    """
    inputs: dict[str, SeriesSpec | FrameSpec] = {
        "mixture": SeriesSpec(dims=_audio_dims(n_channels), time="grid", rate=sr)
    }
    if use_cond:
        inputs["rps_cond"] = _rps_spec(frame_rate)
    return Task(
        name="rps_prediction",
        input_spec=FrameSpec(inputs),
        output_spec=FrameSpec({"rps_pred": _rps_spec(frame_rate)}),
    )


def salience_rps(
    *,
    n_channels: int | None = None,
    sr: tuple[int, int] = AUDIO_RATE,
    frame_rate: tuple[int, int] | None = None,
) -> Task:
    """Mixture in, frequency-salience logits out; RPS is derived from the
    salience map at evaluation time (Hungarian tracking)."""
    return Task(
        name="salience_rps",
        input_spec=FrameSpec(
            {"mixture": SeriesSpec(dims=_audio_dims(n_channels), time="grid", rate=sr)}
        ),
        output_spec=FrameSpec(
            {"salience": SeriesSpec(dims=("batch", "freq", "time"), time="grid", rate=frame_rate)}
        ),
    )


def noise_generation(
    *,
    sr: tuple[int, int] = AUDIO_RATE,
    conditioned: bool = False,
    return_dict: bool = False,
    distributional: bool = False,
) -> Task:
    """RPS trajectories + array geometry in, synthesized drone noise at
    each microphone out.

    ``conditioned``/``return_dict`` don't change this Frame contract (per-
    drone identity travels as ``meta.drone`` string metadata, not a Series
    entry, and the emitter's internal control curves are extra — always-
    allowed — pred entries, not part of the required output spec); they are
    accepted here only so a model config's single ``task_params`` dict can
    build both this Task and the paired
    :class:`tasks.codecs.NoiseGenerationCodec` from the same kwargs (see
    module docstring of ``tasks.codecs``). ``conditioned`` must match the
    model's own ``cond_dim > 0``; ``return_dict`` must match whether the
    loss needs the emitter's ``harm_amps``/``noise_amps`` (E3's smoothness
    regularisers) — see ``src/tasks/noise-generation/AGENTS.md``.

    ``distributional`` likewise does not change this Frame contract: it makes
    the codec ask the model for a *distribution* (a coherent mean plus a
    stochastic spectral envelope) instead of a single realization, adding the
    extra ``coherent``/``noise_mags`` pred entries that
    :class:`losses.SpectralLikelihoodLoss` consumes. ``audio`` is still emitted,
    so metrics are unaffected. See :mod:`losses.spectral_likelihood` for why
    fitting the stochastic branches needs it.
    """
    del conditioned, return_dict, distributional
    return Task(
        name="noise_generation",
        input_spec=FrameSpec(
            {
                "rps": _rps_spec(None),
                "mic_pos": SeriesSpec(dims=("batch", "mic", None), time=None),
                "rotor_pos": SeriesSpec(dims=("batch", "rotor", None), time=None),
            }
        ),
        output_spec=FrameSpec(
            {"audio": SeriesSpec(dims=("batch", "mic", "time"), time="grid", rate=sr)}
        ),
    )


TASK_FACTORIES = {
    "speech_enhancement": speech_enhancement,
    "rps_prediction": rps_prediction,
    "salience_rps": salience_rps,
    "noise_generation": noise_generation,
}
