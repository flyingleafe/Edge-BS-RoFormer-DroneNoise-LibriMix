"""Driver for the stochastic rotor-noise lab notebook.

The model is :mod:`data_processing.stochastic_rotor_noise`: a colored floor
plus Lorentzian harmonic lines, every amplitude drifting as a Gaussian process
in time. This module holds the notebook's controls and plots and nothing else,
so the notebook stays a two-line driver and the logic keeps its tests.

The one class is :class:`Lab`. It keeps the current parameter set, draws a new
one on request, and renders a clip on request. A slider move changes a number
and leaves the static random parts alone, so the same drone can be heard under
different amplitude levels and different wander rates.

Rotor-speed trajectories come from three places, which is what the sliders
switch between:

* ``ou`` --- the Ornstein-Uhlenbeck model in quadrotor control-mode space
  (:func:`data_processing.rps_synthesis.generate`). Continuous wander.
* ``intermittent`` --- the pilot-and-airframe model
  (:func:`data_processing.rps_synthesis.generate_intermittent`). Mostly steady,
  with occasional maneuvers, which is what a real flight looks like.
* ``full_flight`` --- ground, warm-up, takeoff, cruise, landing and ground
  again, so the clip visits zero speed.
* ``real`` --- the telemetry of a real recording, through
  :func:`generator_lab.real_slice`.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

SR = 16000

#: Rate of the rotor-speed trajectory before it is interpolated to the audio
#: grid. Rotor speed is slow, and the full-flight generator runs a per-sample
#: filter in Python, so generating it at the audio rate wastes seconds.
RPS_FS = 400.0

RPS_SOURCES = ("intermittent", "ou", "full_flight", "real")

DATASETS = ("DREGON-frames", "michaels-frames")


def rps_window(
    source: str,
    duration_s: float,
    *,
    rng: np.random.Generator,
    aggressiveness: float = 1.0,
    dataset: str = "DREGON-frames",
    recording: str | None = None,
    start_s: float = 10.0,
) -> tuple[np.ndarray, str]:
    """``((4, T) rotor speeds at the audio rate, a label)`` for one clip."""
    from data_processing import rps_synthesis

    if source == "real":
        from generator_lab import real_slice

        if not recording:
            raise ValueError("the real source needs a recording id")
        exc = real_slice(dataset, recording, start_s, duration_s)
        rps = np.asarray(exc.rps, dtype=np.float64)
        return rps, f"{recording} at {start_s:.1f} s"

    if source == "ou":
        low = rps_synthesis.generate(duration_s, RPS_FS, aggressiveness=aggressiveness, rng=rng)
        label = f"OU, aggressiveness {aggressiveness:.2f}"
    elif source == "full_flight":
        low = rps_synthesis.generate_full_flight(
            None, RPS_FS, aggressiveness=aggressiveness, rng=rng
        )
        label = "full flight"
    else:
        low = rps_synthesis.generate_intermittent(
            duration_s, RPS_FS, aggressiveness=aggressiveness, rng=rng
        )
        label = f"intermittent, aggressiveness {aggressiveness:.2f}"

    low = np.atleast_2d(np.asarray(low, dtype=np.float64))
    t_low = np.arange(low.shape[1]) / RPS_FS
    if source == "full_flight":
        # A full flight is minutes long and the clip is seconds. Take a random
        # window, so successive renders visit the ground, the ramps and cruise.
        start = float(rng.uniform(0.0, max(float(t_low[-1]) - duration_s, 0.0)))
    else:
        start = 0.0
    n_samples = int(round(duration_s * SR))
    t_audio = start + np.arange(n_samples) / SR
    rps = np.stack([np.interp(t_audio, t_low, low[r]) for r in range(low.shape[0])])
    if source == "full_flight":
        label = f"full flight, window at {start:.1f} s"
    return rps, label


def recordings(dataset: str) -> list[str]:
    """Recording ids of a published frames dataset that carry a rotor track."""
    from generator_lab import recordings as _recordings

    return _recordings(dataset)


class Lab:
    """The notebook's state: one parameter set, one rendered clip.

    ``resample()`` draws a new parameter set from the family. ``render()``
    applies the current slider values on top of it and synthesizes.
    """

    #: Slider fields, with their ranges. The first two are the amplitude means
    #: and the rest are the covariance parameters of the Gaussian processes.
    SLIDERS: dict[str, tuple[float, float, float, str]] = {
        "harm_mean_db": (-20.0, 20.0, 0.5, "harmonic level (dB)"),
        "floor_mean_db": (-40.0, 10.0, 0.5, "broadband level (dB)"),
        "harm_gp_std_db": (0.0, 12.0, 0.25, "harmonic wander (dB)"),
        "harm_gp_tau_s": (0.1, 12.0, 0.1, "harmonic wander time (s)"),
        "harm_coherence": (0.0, 1.0, 0.05, "harmonic coherence"),
        "floor_gp_std_db": (0.0, 10.0, 0.25, "floor wander (dB)"),
        "floor_gp_tau_s": (0.2, 20.0, 0.2, "floor wander time (s)"),
        "floor_tilt_gp_std": (0.0, 3.0, 0.05, "floor color wander (dB/oct)"),
        "floor_tilt_gp_tau_s": (1.0, 30.0, 0.5, "floor color time (s)"),
    }

    def __init__(
        self,
        *,
        seed: int = 0,
        n_harmonics: int = 80,
        n_rotors: int = 4,
        n_fft: int = 2048,
    ):
        from data_processing import stochastic_rotor_noise as srn

        self.srn = srn
        self.n_harmonics = int(n_harmonics)
        self.n_rotors = int(n_rotors)
        self.n_fft = int(n_fft)
        self.seed = int(seed)
        self.params = srn.sample_params(
            np.random.default_rng(self.seed),
            n_rotors=self.n_rotors,
            n_harmonics=self.n_harmonics,
            sample_rate=SR,
        )
        self.last: dict[str, Any] = {}

    def resample(self, seed: int | None = None) -> Any:
        """Draw a new parameter set — a new drone, a new floor color."""
        if seed is not None:
            self.seed = int(seed)
        else:
            self.seed += 1
        self.params = self.srn.sample_params(
            np.random.default_rng(self.seed),
            n_rotors=self.n_rotors,
            n_harmonics=self.n_harmonics,
            sample_rate=SR,
        )
        return self.params

    def slider_values(self) -> dict[str, float]:
        """The current value of every slider field, read off the parameters."""
        return {key: float(getattr(self.params, key)) for key in self.SLIDERS}

    def render(
        self,
        *,
        duration_s: float = 8.0,
        rps_source: str = "intermittent",
        aggressiveness: float = 1.0,
        dataset: str = "DREGON-frames",
        recording: str | None = None,
        start_s: float = 10.0,
        render_seed: int | None = None,
        **overrides: float,
    ) -> dict[str, Any]:
        """Synthesize one clip and keep everything the plots need."""
        rng = np.random.default_rng(self.seed if render_seed is None else int(render_seed))
        rps, rps_label = rps_window(
            rps_source,
            duration_s,
            rng=rng,
            aggressiveness=aggressiveness,
            dataset=dataset,
            recording=recording,
            start_s=start_s,
        )
        params = self.params.with_(**{k: float(v) for k, v in overrides.items()})
        audio, diag = self.srn.synthesize(params, rps, rng=rng, n_mics=1, n_fft=self.n_fft)
        self.last = {
            "audio": audio,
            "rps": rps,
            "params": params,
            "diag": diag,
            "rps_label": rps_label,
            "duration_s": float(duration_s),
        }
        return self.last

    # ── Plots ───────────────────────────────────────────────────────────────

    def figure(self, *, f_max: float = 8000.0, dyn_range: float = 70.0):
        """Three panels: the spectrogram, the rotor speeds, the spectrum."""
        import matplotlib.pyplot as plt

        if not self.last:
            raise RuntimeError("call render() first")
        audio = self.last["audio"][0]
        rps = self.last["rps"]
        diag = self.last["diag"]

        n_fft, hop = self.n_fft, self.n_fft // 4
        window = np.hanning(n_fft + 1)[:n_fft]
        n_frames = max((audio.size - n_fft) // hop, 1)
        frames = np.stack([audio[i * hop : i * hop + n_fft] * window for i in range(n_frames)])
        power = np.abs(np.fft.rfft(frames, axis=-1)) ** 2
        db = 10.0 * np.log10(np.maximum(power, 1e-20)).T
        freqs = np.fft.rfftfreq(n_fft, 1.0 / SR)
        times = np.arange(n_frames) * hop / SR
        top = float(np.percentile(db, 99.8))

        fig, axes = plt.subplots(
            3, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [2.2, 1.0, 1.3]}
        )
        keep = freqs <= f_max
        axes[0].pcolormesh(
            times,
            freqs[keep],
            db[keep],
            vmin=top - dyn_range,
            vmax=top,
            shading="auto",
            cmap="magma",
        )
        axes[0].set_ylabel("frequency (Hz)")
        axes[0].set_title(f"realized clip — {self.last['rps_label']}")

        t_rps = np.arange(rps.shape[1]) / SR
        for r in range(rps.shape[0]):
            axes[1].plot(t_rps, rps[r], lw=1.2, label=f"rotor {r + 1}")
        axes[1].set_ylabel("rev/s")
        axes[1].set_xlabel("time (s)")
        axes[1].legend(fontsize=7, ncol=4, loc="upper right")
        axes[1].grid(alpha=0.2)

        model_db = self.srn.model_psd_db(diag, 0).mean(axis=0)
        realized = 10.0 * np.log10(np.maximum(power.mean(axis=0), 1e-20))
        model_db = model_db - model_db.max()
        realized = realized - realized.max()
        axes[2].plot(freqs[keep], realized[keep], lw=0.8, color="#888888", label="realized")
        axes[2].plot(freqs[keep], model_db[keep], lw=1.2, color="#c0392b", label="model")
        axes[2].set_xlabel("frequency (Hz)")
        axes[2].set_ylabel("dB")
        axes[2].set_ylim(-dyn_range, 3)
        axes[2].legend(fontsize=8)
        axes[2].grid(alpha=0.2)
        fig.tight_layout()
        return fig

    def player(self):
        """An audio widget for the rendered clip."""
        from IPython.display import Audio

        return Audio(self.last["audio"][0], rate=SR, normalize=True)

    def summary(self) -> dict[str, Any]:
        """The scalar parameters of the current clip, for a printed readout."""
        params = self.last.get("params", self.params)
        out = {
            k: (round(float(v), 3) if isinstance(v, (int, float)) else None)
            for k, v in asdict(params).items()
            if not isinstance(v, np.ndarray)
        }
        return {k: v for k, v in out.items() if v is not None}

    # ── The panel ───────────────────────────────────────────────────────────

    def panel(self):
        """The whole interactive panel: controls, plots and a player."""
        import ipywidgets as widgets
        from IPython.display import clear_output, display

        sliders = {
            key: widgets.FloatSlider(
                value=float(getattr(self.params, key)),
                min=lo,
                max=hi,
                step=step,
                description=label,
                continuous_update=False,
                readout_format=".2f",
                style={"description_width": "170px"},
                layout=widgets.Layout(width="430px"),
            )
            for key, (lo, hi, step, label) in self.SLIDERS.items()
        }
        duration = widgets.FloatSlider(
            value=8.0,
            min=1.0,
            max=20.0,
            step=0.5,
            description="duration (s)",
            continuous_update=False,
            style={"description_width": "170px"},
            layout=widgets.Layout(width="430px"),
        )
        source = widgets.Dropdown(
            options=list(RPS_SOURCES),
            value="intermittent",
            description="rotor speeds",
            style={"description_width": "170px"},
            layout=widgets.Layout(width="430px"),
        )
        aggressiveness = widgets.FloatSlider(
            value=1.0,
            min=0.0,
            max=3.0,
            step=0.1,
            description="aggressiveness",
            continuous_update=False,
            style={"description_width": "170px"},
            layout=widgets.Layout(width="430px"),
        )
        dataset = widgets.Dropdown(
            options=list(DATASETS),
            description="dataset",
            style={"description_width": "170px"},
            layout=widgets.Layout(width="430px"),
        )
        recording = widgets.Dropdown(
            options=[],
            description="recording",
            style={"description_width": "170px"},
            layout=widgets.Layout(width="430px"),
        )
        start = widgets.FloatSlider(
            value=12.0,
            min=0.0,
            max=60.0,
            step=0.5,
            description="start (s)",
            continuous_update=False,
            style={"description_width": "170px"},
            layout=widgets.Layout(width="430px"),
        )
        real_box = widgets.VBox([dataset, recording, start])
        real_box.layout.display = "none"

        new_params = widgets.Button(description="New random parameters", button_style="info")
        regenerate = widgets.Button(description="Regenerate", button_style="success")
        status = widgets.HTML()
        out = widgets.Output()

        def on_source(change: Any) -> None:
            is_real = change["new"] == "real"
            real_box.layout.display = "flex" if is_real else "none"
            aggressiveness.disabled = is_real
            if is_real and not recording.options:
                fill_recordings(None)

        def fill_recordings(_: Any) -> None:
            status.value = "loading recording list…"
            try:
                recording.options = recordings(dataset.value)
                if recording.options:
                    recording.value = recording.options[0]
                status.value = ""
            except Exception as exc:  # noqa: BLE001 — surfaced in the panel
                status.value = f"<span style='color:#c0392b'>{exc}</span>"

        def push_sliders() -> None:
            for key, slider in sliders.items():
                value = float(getattr(self.params, key))
                slider.value = float(np.clip(value, slider.min, slider.max))

        def draw(_: Any = None) -> None:
            status.value = "rendering…"
            overrides = {key: slider.value for key, slider in sliders.items()}
            try:
                self.render(
                    duration_s=duration.value,
                    rps_source=source.value,
                    aggressiveness=aggressiveness.value,
                    dataset=dataset.value,
                    recording=recording.value or None,
                    start_s=start.value,
                    **overrides,
                )
            except Exception as exc:  # noqa: BLE001 — surfaced in the panel
                status.value = f"<span style='color:#c0392b'>{exc}</span>"
                return
            with out:
                clear_output(wait=True)
                import matplotlib.pyplot as plt

                fig = self.figure()
                plt.show()
                plt.close(fig)
                display(self.player())
            status.value = f"seed {self.seed}"

        def on_new(_: Any) -> None:
            self.resample()
            push_sliders()
            draw()

        source.observe(on_source, names="value")
        dataset.observe(fill_recordings, names="value")
        new_params.on_click(on_new)
        regenerate.on_click(draw)

        controls = widgets.HBox(
            [
                widgets.VBox(
                    [
                        sliders["harm_mean_db"],
                        sliders["floor_mean_db"],
                        sliders["harm_gp_std_db"],
                        sliders["harm_gp_tau_s"],
                        sliders["harm_coherence"],
                    ]
                ),
                widgets.VBox(
                    [
                        sliders["floor_gp_std_db"],
                        sliders["floor_gp_tau_s"],
                        sliders["floor_tilt_gp_std"],
                        sliders["floor_tilt_gp_tau_s"],
                        duration,
                    ]
                ),
                widgets.VBox([source, aggressiveness, real_box]),
            ]
        )
        draw()
        return widgets.VBox([controls, widgets.HBox([new_params, regenerate, status]), out])
