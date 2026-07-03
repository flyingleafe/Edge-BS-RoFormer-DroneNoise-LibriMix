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
) -> Task:
    """Drone-noise mixture in, per-rotor rotation speeds out (on the STFT
    frame grid — pass ``frame_rate=(sr, hop)`` for the exact rate)."""
    return Task(
        name="rps_prediction",
        input_spec=FrameSpec(
            {"mixture": SeriesSpec(dims=_audio_dims(n_channels), time="grid", rate=sr)}
        ),
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
) -> Task:
    """RPS trajectories + array geometry in, synthesized drone noise at
    each microphone out."""
    return Task(
        name="noise_generation",
        input_spec=FrameSpec(
            {
                "rps": _rps_spec(None),
                "mic_pos": SeriesSpec(dims=("batch", "mic", None), time=None),
                "rotor_pos": SeriesSpec(dims=("batch", "rotor", None), time=None),
                "drone_id": SeriesSpec(dims=("batch",), time=None, dtype="integer"),
            },
            frozenset({"drone_id"}),
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
