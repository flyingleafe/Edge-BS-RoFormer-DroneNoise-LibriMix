"""Per-task codecs — the Frame boundary around tensor-only models.

Models stay plain tensor ``nn.Module``s (docs/refactor-unified-framework.md §
"Task typing: FrameSpec"): a codec is the thin adapter that pulls the tensors
a model's ``forward`` needs out of a batched ``td.Frame`` (:meth:`Codec.to_inputs`)
and wraps whatever the model returns back into a ``td.Frame`` matching the
task's ``output_spec`` (:meth:`Codec.to_frame`). One codec class per task in
``tasks.task.TASK_FACTORIES``, built with the *same* keyword parameters as the
matching task factory — ``build_codec(name, **params)`` and
``tasks.task.TASK_FACTORIES[name](**params)`` are meant to be called with an
identical ``params`` dict, so a model config's ``task_params`` builds both the
spec and the codec in one shot.

Every codec is deliberately thin: no normalization, no device juggling, no
padding logic — those stay in the model or the training loop. A codec only
knows entry names, dim order, and sample rate.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import tdseries as td
import torch

from losses._common import get_tensor
from tasks.task import AUDIO_RATE

__all__ = [
    "Codec",
    "SpeechEnhancementCodec",
    "RPSPredictionCodec",
    "SalienceRPSCodec",
    "NoiseGenerationCodec",
    "CODEC_FACTORIES",
    "build_codec",
]


def _audio_dims(n_channels: int | None) -> tuple[str | None, ...]:
    """Batched audio dims: mono ``(batch, time)``; multi-mic ``(batch, mic, time)``.

    Mirrors ``tasks.task._audio_dims`` (kept private there) so codecs build
    the exact same dim tuples as the task specs they pair with.
    """
    return ("batch", "time") if n_channels is None else ("batch", "mic", "time")


def _squeeze_extra_dims(data: torch.Tensor, n_dims: int) -> torch.Tensor:
    """Best-effort normalization for models that return extra singleton axes.

    E.g. ``DCUNetRefactored`` returns mono audio as ``(B, 1, 1, T)`` (a
    leftover STFT-processor channel axis) where the task spec wants
    ``(batch, time)``. Squeezes size-1 axes strictly between the leading
    (batch) and trailing (time) axis until ``data.ndim == n_dims``, or gives
    up and lets the mismatch surface as a spec-validation error — codecs
    normalize *shape*, not silently reinterpret a model that is actually
    producing the wrong number of non-singleton dims.
    """
    while data.ndim > n_dims:
        squeezed = False
        for ax in range(1, data.ndim - 1):
            if data.shape[ax] == 1:
                data = data.squeeze(ax)
                squeezed = True
                break
        if not squeezed:
            break
    return data


def _batched_series(
    data: torch.Tensor, dims: tuple[str | None, ...], rate: tuple[int, int] | None
) -> td.Series:
    """Wrap a model-output tensor as a batched ``td.Series`` with ``dims``.

    A ``"time"`` dim gets a fresh phase-0 ``GridIndex`` at ``rate`` sized to
    the tensor's actual length (robust to a model changing frame count via
    padding/centering); non-temporal dims (e.g. noise-generation's mic/rotor
    geometry) are just ``td.wrap``-ed. ``data`` is first squeezed to
    ``len(dims)`` axes (see :func:`_squeeze_extra_dims`).
    """
    data = _squeeze_extra_dims(data, len(dims))
    if "time" in dims:
        if rate is None:
            raise ValueError(f"a 'time' entry with dims {dims} needs an exact rate, got None")
        n = int(data.shape[dims.index("time")])
        idx = td.GridIndex.create(rate, n, t_start=0)
        return td.Series(data, dims, {"time": idx})
    return td.wrap(data, dims=dims)


def _split_model_output(outputs: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Normalize a model's return value to ``(primary, auxiliary)``.

    Mirrors ``train.py``/``train_rps_predictor.py``'s convention: a model
    either returns a bare tensor, or a ``(primary, aux)`` tuple (e.g.
    ``(enhanced, rps_pred)`` for DCUNet/DCCRN with ``predict_rps=True``).
    """
    if isinstance(outputs, tuple):
        primary = outputs[0]
        aux = outputs[1] if len(outputs) > 1 else None
        return primary, aux
    return outputs, None


@runtime_checkable
class Codec(Protocol):
    """Structural protocol every task codec in this module satisfies.

    ``call_model`` is the third leg every concrete codec adds beyond the two
    Frame-boundary methods: legacy model ``forward`` signatures don't agree
    on parameter names or input rank with the task's canonical entry names
    (e.g. ``SimpleConvV2.forward(audio)`` wants ``(B, T)``,
    ``DCUNetRefactored.forward(x, rps=None)`` wants ``(B, 1, T)``), so the
    codec — which already owns the entry-name/shape contract — also owns the
    one-line adapter that turns ``to_inputs``'s dict into the model's actual
    call convention.
    """

    def to_inputs(self, batch: td.Frame) -> dict[str, torch.Tensor]: ...
    def to_frame(self, outputs: Any, batch: td.Frame) -> td.Frame: ...
    def call_model(self, model: Any, inputs: dict[str, torch.Tensor]) -> Any: ...


class SpeechEnhancementCodec:
    """Codec for ``tasks.task.speech_enhancement``.

    ``to_inputs`` pulls ``"mixture"`` (and ``"rps"`` when ``use_rps``);
    ``to_frame`` wraps the model's primary output as ``"enhanced"`` and, when
    ``predict_rps`` and the model returned an auxiliary tensor, also emits
    ``"rps_pred"``.
    """

    def __init__(
        self,
        *,
        n_channels: int | None = None,
        use_rps: bool = False,
        predict_rps: bool = False,
        sr: tuple[int, int] = AUDIO_RATE,
        rps_rate: tuple[int, int] | None = None,
    ) -> None:
        self.n_channels = n_channels
        self.use_rps = use_rps
        self.predict_rps = predict_rps
        self.sr = sr
        self.rps_rate = rps_rate
        self._audio_dims = _audio_dims(n_channels)

    def to_inputs(self, batch: td.Frame) -> dict[str, torch.Tensor]:
        inputs = {"mixture": get_tensor(batch, "mixture")}
        if self.use_rps:
            inputs["rps"] = get_tensor(batch, "rps")
        return inputs

    def to_frame(self, outputs: Any, batch: td.Frame) -> td.Frame:
        enhanced, rps_pred = _split_model_output(outputs)
        entries: dict[str, Any] = {"enhanced": _batched_series(enhanced, self._audio_dims, self.sr)}
        if self.predict_rps and rps_pred is not None:
            entries["rps_pred"] = _batched_series(
                rps_pred, ("batch", "rotor", "time"), self.rps_rate
            )
        return td.Frame(entries)

    def call_model(self, model: Any, inputs: dict[str, torch.Tensor]) -> Any:
        x = inputs["mixture"]
        if self.n_channels is None:
            # Mono task spec is (B, T); DCUNet/DCCRN-family models want an
            # explicit (B, 1, T) channel axis (train.py's convention).
            x = x.unsqueeze(1)
        if self.use_rps:
            return model(x, rps=inputs["rps"])
        return model(x)


class RPSPredictionCodec:
    """Codec for ``tasks.task.rps_prediction``: mixture in, ``rps_pred`` out."""

    def __init__(
        self,
        *,
        n_channels: int | None = None,
        sr: tuple[int, int] = AUDIO_RATE,
        frame_rate: tuple[int, int] | None = None,
    ) -> None:
        self.sr = sr
        self.frame_rate = frame_rate
        self._audio_dims = _audio_dims(n_channels)

    def to_inputs(self, batch: td.Frame) -> dict[str, torch.Tensor]:
        return {"mixture": get_tensor(batch, "mixture")}

    def to_frame(self, outputs: Any, batch: td.Frame) -> td.Frame:
        rps_pred, _aux = _split_model_output(outputs)
        return td.Frame(
            {"rps_pred": _batched_series(rps_pred, ("batch", "rotor", "time"), self.frame_rate)}
        )

    def call_model(self, model: Any, inputs: dict[str, torch.Tensor]) -> Any:
        return model(inputs["mixture"])


class SalienceRPSCodec:
    """Codec for ``tasks.task.salience_rps``: mixture in, ``salience`` out."""

    def __init__(
        self,
        *,
        n_channels: int | None = None,
        sr: tuple[int, int] = AUDIO_RATE,
        frame_rate: tuple[int, int] | None = None,
    ) -> None:
        self.sr = sr
        self.frame_rate = frame_rate
        self._audio_dims = _audio_dims(n_channels)

    def to_inputs(self, batch: td.Frame) -> dict[str, torch.Tensor]:
        return {"mixture": get_tensor(batch, "mixture")}

    def to_frame(self, outputs: Any, batch: td.Frame) -> td.Frame:
        salience, _aux = _split_model_output(outputs)
        return td.Frame(
            {"salience": _batched_series(salience, ("batch", "freq", "time"), self.frame_rate)}
        )

    def call_model(self, model: Any, inputs: dict[str, torch.Tensor]) -> Any:
        return model(inputs["mixture"])


class NoiseGenerationCodec:
    """Codec for ``tasks.task.noise_generation``: RPS + geometry in, ``audio`` out."""

    def __init__(self, *, sr: tuple[int, int] = AUDIO_RATE) -> None:
        self.sr = sr

    def to_inputs(self, batch: td.Frame) -> dict[str, torch.Tensor]:
        inputs = {
            "rps": get_tensor(batch, "rps"),
            "mic_pos": get_tensor(batch, "mic_pos"),
            "rotor_pos": get_tensor(batch, "rotor_pos"),
        }
        if "drone_id" in batch:
            inputs["drone_id"] = get_tensor(batch, "drone_id")
        return inputs

    def to_frame(self, outputs: Any, batch: td.Frame) -> td.Frame:
        audio, _aux = _split_model_output(outputs)
        return td.Frame({"audio": _batched_series(audio, ("batch", "mic", "time"), self.sr)})

    def call_model(self, model: Any, inputs: dict[str, torch.Tensor]) -> Any:
        kwargs = {"mic_pos": inputs["mic_pos"], "rotor_pos": inputs["rotor_pos"]}
        if "drone_id" in inputs:
            kwargs["drone_id"] = inputs["drone_id"]
        return model(inputs["rps"], **kwargs)


CODEC_FACTORIES = {
    "speech_enhancement": SpeechEnhancementCodec,
    "rps_prediction": RPSPredictionCodec,
    "salience_rps": SalienceRPSCodec,
    "noise_generation": NoiseGenerationCodec,
}


def build_codec(name: str, **params: Any) -> Codec:
    """Build the codec for task ``name`` — call with the same ``params`` dict
    passed to ``tasks.task.TASK_FACTORIES[name]`` to build the matching spec."""
    if name not in CODEC_FACTORIES:
        raise ValueError(f"Unknown task {name!r}; choose one of {sorted(CODEC_FACTORIES)}")
    return CODEC_FACTORIES[name](**params)
