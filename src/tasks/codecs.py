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

from data_processing.frames import meta_dict
from losses._common import get_tensor
from tasks.noise_generation import geometry_to_rel_pos
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
    """Codec for ``tasks.task.rps_prediction``: mixture in, ``rps_pred`` out.

    ``use_cond=True`` (the conditional-refiner variant — see
    ``tasks.task.rps_prediction``) additionally pulls ``"rps_cond"`` and calls
    ``model(mixture, cond)``.
    """

    def __init__(
        self,
        *,
        n_channels: int | None = None,
        sr: tuple[int, int] = AUDIO_RATE,
        frame_rate: tuple[int, int] | None = None,
        use_cond: bool = False,
    ) -> None:
        self.sr = sr
        self.frame_rate = frame_rate
        self.use_cond = use_cond
        self._audio_dims = _audio_dims(n_channels)

    def to_inputs(self, batch: td.Frame) -> dict[str, torch.Tensor]:
        inputs = {"mixture": get_tensor(batch, "mixture")}
        if self.use_cond:
            inputs["rps_cond"] = get_tensor(batch, "rps_cond")
        return inputs

    def to_frame(self, outputs: Any, batch: td.Frame) -> td.Frame:
        rps_pred, _aux = _split_model_output(outputs)
        return td.Frame(
            {"rps_pred": _batched_series(rps_pred, ("batch", "rotor", "time"), self.frame_rate)}
        )

    def call_model(self, model: Any, inputs: dict[str, torch.Tensor]) -> Any:
        if self.use_cond:
            return model(inputs["mixture"], inputs["rps_cond"])
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


def _sample_from_stats(coherent: torch.Tensor, noise_psd: torch.Tensor) -> torch.Tensor:
    """Draw one realization from a predicted (mean, envelope) pair.

    Used *only* to give the magnitude metrics something to score in
    distributional mode — the loss never sees it. Each microphone gets its own
    excitation, which is right for the incoherent branches (wind is incoherent
    by construction, and the per-rotor broadband residuals are independent).

    Args:
        coherent: ``[B, M, T]`` deterministic component.
        noise_psd: ``[B, M, n_frames, n_freqs]`` stochastic power envelope.
    """
    from models.generative.dsp import frequency_filter

    b, m, t = coherent.shape
    # Detached + eps-floored: this is a metrics-only draw, and sqrt at exactly
    # zero has an infinite derivative (the trap the power interface exists to
    # avoid), so it must never be on the training graph.
    mags = noise_psd.detach().clamp_min(1e-20).sqrt()
    mags = mags.reshape(b * m, mags.shape[-2], mags.shape[-1])
    excitation = torch.randn(b * m, t, device=coherent.device, dtype=coherent.dtype)
    noise = frequency_filter(excitation, mags).reshape(b, m, t)
    return coherent + noise


class NoiseGenerationCodec:
    """Codec for ``tasks.task.noise_generation``: RPS + geometry in, ``audio`` out.

    Fixes the codec/model signature mismatch documented in
    REPLICATION.md § "E1/E2/E3 — noise-generation training": ``mic_pos``/
    ``rotor_pos`` (the Frame's geometry entries) are turned into the
    ``rel_pos`` tensor :class:`~models.generative.PositionalHarmonicNoiseGen.forward`
    actually wants via :func:`tasks.noise_generation.geometry_to_rel_pos`
    (batched-tensor path). ``conditioned=True`` additionally resolves each
    sample's per-drone identity from ``batch["meta"]["drone"]`` (a string
    per sample, e.g. ``"dregon"``/``"michaels"`` — see
    :class:`~data_processing.frame_datasets.NoiseGenFrameDataset`) and
    passes it through as ``drone_names`` for the model to resolve a
    conditioning code ``z`` via its own (trainable, checkpoint-persisted)
    ``tasks.noise_generation.DroneCodebook`` submodule — see
    ``models.registry.build_noise_gen_model``'s ``drone_names`` argument and
    ``src/tasks/noise-generation/AGENTS.md`` for why the codebook lives
    inside "the model" rather than the codec (the training loop's
    checkpoint/optimizer contract is exactly one ``model.state_dict()``).
    ``return_dict=True`` requests the emitter's internal control curves
    (``harm_amps``/``noise_amps``) as extra pred entries, for E3's
    smoothness regularisers (``losses.SmoothnessPenalty``).
    """

    def __init__(
        self,
        *,
        sr: tuple[int, int] = AUDIO_RATE,
        conditioned: bool = False,
        return_dict: bool = False,
        distributional: bool = False,
        spatial: bool = False,
        default_drone: str = "dregon",
    ) -> None:
        self.sr = sr
        self.conditioned = conditioned
        self.return_dict = return_dict
        self.distributional = distributional
        self.spatial = spatial
        self.default_drone = default_drone

    def to_inputs(self, batch: td.Frame) -> dict[str, Any]:
        mic_pos = get_tensor(batch, "mic_pos")  # (B, M, 3)
        rotor_pos = get_tensor(batch, "rotor_pos")  # (B, R, 3)
        inputs: dict[str, Any] = {
            "rps": get_tensor(batch, "rps"),
            "rel_pos": geometry_to_rel_pos(mic_pos, rotor_pos),  # (B, M, R, 3)
        }
        if self.conditioned:
            names = meta_dict(batch).get("drone")
            if names is None:
                names = [self.default_drone] * int(inputs["rps"].shape[0])
            inputs["drone_names"] = list(names)
        return inputs

    def to_frame(self, outputs: Any, batch: td.Frame) -> td.Frame:
        if isinstance(outputs, dict):
            entries: dict[str, Any] = {
                "audio": _batched_series(outputs["audio"], ("batch", "mic", "time"), self.sr)
            }
            if "harm_amps" in outputs:
                entries["harm_amps"] = td.wrap(
                    outputs["harm_amps"], dims=("batch", "rotor", None, None, None)
                )
            if "noise_amps" in outputs:
                entries["noise_amps"] = td.wrap(
                    outputs["noise_amps"], dims=("batch", "rotor", None, None)
                )
            # Distributional mode: the predicted mean and the stochastic
            # branches' spectral envelope, which SpectralLikelihoodLoss scores
            # instead of `audio`. `coherent` is a waveform (so it carries the
            # audio rate); `noise_psd` is a (frame, freq) envelope on its own
            # grid, hence untyped trailing dims.
            if "coherent" in outputs:
                entries["coherent"] = _batched_series(
                    outputs["coherent"], ("batch", "mic", "time"), self.sr
                )
            for key, dims in (
                ("source_psd", ("batch", "rotor", None, None)),
                ("wind_psd", ("batch", "mic", None, None)),
                ("rel_pos", ("batch", "mic", "rotor", None)),
            ):
                if key in outputs:
                    entries[key] = td.wrap(outputs[key], dims=dims)
            if "noise_psd" in outputs:
                entries["noise_psd"] = td.wrap(
                    outputs["noise_psd"], dims=("batch", "mic", None, None)
                )
            return td.Frame(entries)
        audio, _aux = _split_model_output(outputs)
        return td.Frame({"audio": _batched_series(audio, ("batch", "mic", "time"), self.sr)})

    def call_model(self, model: Any, inputs: dict[str, torch.Tensor]) -> Any:
        kwargs: dict[str, Any] = {}
        if self.spatial:
            fn = getattr(model, "spatial_stats", None)
            if fn is None:
                raise TypeError(f"{type(model).__name__} has no `spatial_stats`")
            args = (inputs["rps"], inputs["rel_pos"])
            out = dict(
                fn(*args, inputs["drone_names"]) if self.conditioned else fn(*args)
            )
            if "wind_psd" not in out:
                # A coherent-only generator has no diagonal term; the loss floors
                # it anyway, so an explicit zero keeps the control arm well-posed.
                src = out["source_psd"]
                m = out["rel_pos"].shape[1]
                out["wind_psd"] = src.new_zeros((src.shape[0], m, src.shape[2], src.shape[3]))
            return out
        if self.distributional:
            # Ask for the predicted DISTRIBUTION rather than one realization:
            # a coherent mean plus the stochastic branches' spectral envelope,
            # with nothing sampled. `losses.SpectralLikelihoodLoss` scores that
            # directly; see its module docstring for why sampling here would
            # bias the fitted noise level low and swamp its gradient.
            fn = getattr(model, "spectral_stats", None)
            if fn is None:
                raise TypeError(
                    f"{type(model).__name__} has no `spectral_stats`; a distributional "
                    "codec needs a model that can predict a mean and a variance "
                    "(see models.generative.PositionalHarmonicNoiseGen.spectral_stats)"
                )
            if self.conditioned:
                out = fn(inputs["rps"], inputs["rel_pos"], inputs["drone_names"])
            else:
                out = fn(inputs["rps"], inputs["rel_pos"])
            # A realization for the magnitude metrics only — never for the loss.
            out = dict(out)
            out["audio"] = _sample_from_stats(out["coherent"], out["noise_psd"])
            return out
        if self.return_dict:
            kwargs["return_dict"] = True
        if self.conditioned:
            return model(inputs["rps"], inputs["rel_pos"], inputs["drone_names"], **kwargs)
        return model(inputs["rps"], inputs["rel_pos"], **kwargs)


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
