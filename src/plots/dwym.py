"""``plots.dwym`` — the do-what-you-mean plotting front door.

One call covers the common figures::

    from plots import dwym
    dwym(frame)                          # dispatch on entry names
    dwym({"model A": fa, "model B": fb}) # aligned comparison per label
    dwym(frame, rps="motor_speed")       # explicit entry remap (silences
                                         # the coercion warning)
    dwym(frame, renderer="timeframe")    # force a dispatch route

Frame-level dispatch (on the coerced entry names, see :mod:`plots.coerce`):

========================  =============================================
Route (``result.route``)  Trigger / figure
========================  =============================================
``"se"``                  >= 2 of ``mixture``/``target``/``enhanced`` —
                          spectrogram rows on a shared time axis
``"salience"``            a ``salience`` entry — salience heatmap (+
                          input spectrogram and GT ``rps`` row if present)
``"noise_gen"``           ``audio`` + ``generated`` in one frame, or a
                          dict of exactly two bare-audio frames —
                          real-vs-generated spectrogram grid
``"rps"``                 ``audio`` + ``rps`` (and/or ``rps_pred``) —
                          spectrogram + RPS rows via ``plot_timeframe``
``"audio"``               a frame whose only temporal entry is ``audio``
                          — spectrogram + waveform
``"timeframe"``           anything else — ``plot_timeframe``'s existing
                          per-track dispatch over every temporal entry
========================  =============================================

Multi-frame input (list or ``{label: Frame}`` dict) renders one aligned
figure with a row block per label when every frame dispatches to the same
route (``route`` gains a ``"multi:"`` prefix); heterogeneous dicts fall
back to one figure per frame (``route="mixed"``).

Hints: ``renderer=<route>`` forces a path; ``<canonical>="entry"`` string
hints are entry-name remaps passed to :func:`plots.coerce.coerce_frame`;
``fmax``/``freqs`` shape the spectrogram/salience tracks; everything else
flows into ``plot_timeframe`` (and from there into the existing
``PlotTrack.hints``/style channel of the track renderers).

The result is environment-aware: inside IPython, ``DwymResult`` displays
its figures AND one ``IPython.display.Audio`` player per collected audio
entry; outside, it is a plain figure holder with ``.figures``/``.save()``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import tdseries as td

from plots.audio import first_channel, sample_rate_of, to_mono, to_numpy
from plots.coerce import CANONICAL_ENTRIES, coerce_frame
from plots.noise_gen import noise_gen_comparison_tracks
from plots.se import se_comparison_tracks
from plots.timeframe import PlotTrack, plot_timeframe
from plots.timeframe.renderers import make_spectrogram_series

__all__ = ["DwymResult", "dwym"]

#: Entries collected as playable audio (mono + sample rate) on the result.
AUDIO_ENTRIES = ("audio", "mixture", "target", "enhanced", "generated")

_ROUTES = ("se", "salience", "noise_gen", "rps", "audio", "timeframe")


def _in_ipython() -> bool:
    try:
        from IPython.core.getipython import get_ipython
    except ImportError:
        return False
    return get_ipython() is not None


@dataclass
class DwymResult:
    """Figures + audio arrays returned by :func:`dwym`.

    * ``figures`` — the matplotlib figure(s); ``figure`` is the first.
    * ``audio`` — ``{entry_or_label/entry: (mono waveform, sample_rate)}``.
    * ``route`` — the dispatch route taken (see the module table), for
      tests and for checking a guess.

    In IPython the object displays itself: figures first, then one audio
    player per entry. Outside IPython use ``.save(path)`` / ``.figures``.
    """

    figures: list[matplotlib.figure.Figure]
    audio: dict[str, tuple[np.ndarray, int]] = field(default_factory=dict)
    route: str = "timeframe"

    @property
    def figure(self) -> matplotlib.figure.Figure:
        """The first (usually only) figure."""
        return self.figures[0]

    def save(self, path: str | Path, **savefig_kwargs: Any) -> list[Path]:
        """Save the figure(s); multiple figures get ``-<i>`` suffixes."""
        path = Path(path)
        if len(self.figures) == 1:
            self.figures[0].savefig(path, **savefig_kwargs)
            return [path]
        paths = []
        for i, fig in enumerate(self.figures):
            p = path.with_name(f"{path.stem}-{i}{path.suffix}")
            fig.savefig(p, **savefig_kwargs)
            paths.append(p)
        return paths

    def close(self) -> None:
        """Close all held figures."""
        for fig in self.figures:
            plt.close(fig)

    def _ipython_display_(self) -> None:
        from IPython.display import Audio, display

        for fig in self.figures:
            display(fig)
        for name, (waveform, sr) in self.audio.items():
            print(name)
            display(Audio(waveform, rate=sr, normalize=True))


# ---------------------------------------------------------------------------
# Route decision + per-route track builders
# ---------------------------------------------------------------------------


def _temporal_entries(frame: td.Frame) -> list[str]:
    return [k for k, v in frame.items() if isinstance(v, td.Series) and v.has_time]


def _route_for(frame: td.Frame) -> str:
    if sum(k in frame for k in ("mixture", "target", "enhanced")) >= 2:
        return "se"
    if "salience" in frame:
        return "salience"
    if "audio" in frame and "generated" in frame:
        return "noise_gen"
    if "audio" in frame and ("rps" in frame or "rps_pred" in frame):
        return "rps"
    if _temporal_entries(frame) == ["audio"]:
        return "audio"
    return "timeframe"


def _retitle(track: PlotTrack, title: str) -> PlotTrack:
    return PlotTrack(
        series=track.series, renderer=track.renderer, hints={**track.hints, "title": title}
    )


def _spectrogram_track(frame: td.Frame, entry: str, hints: dict[str, Any]) -> PlotTrack:
    spec = make_spectrogram_series(first_channel(frame[entry]), fmax=hints.get("fmax"))
    return _retitle(spec, f"{entry} spectrogram")


def _aligned_rps_pred_track(frame: td.Frame) -> PlotTrack:
    """The ``rps_pred`` entry as a track, PIT-aligned to GT ``rps`` if present."""
    from tasks.rps_prediction import align_rps_to_gt

    pred_series = frame["rps_pred"]
    pred = to_numpy(pred_series.data)
    if "rps" in frame and pred.ndim == 2:
        gt_series = frame["rps"]
        if isinstance(gt_series, td.Series) and gt_series.data is not None:
            tindex = pred_series.tindex
            if isinstance(tindex, td.GridIndex):
                times = tindex.sample_times()
            else:
                times = np.linspace(frame.t_start, frame.t_end, pred.shape[-1])
            gt = np.asarray(gt_series.interpolate(times))
            pred = align_rps_to_gt(pred, gt)
            # Rebuild on a uniform grid spanning the prediction window so the
            # rps renderer draws the aligned data (StampIndex stays as-is).
            dur = float(times[-1] - times[0]) if len(times) > 1 else 1.0
            rate = Fraction(pred.shape[-1] * td.TICKS_PER_SECOND, int(dur * td.TICKS_PER_SECOND))
            dims = ("rotor", "time")
            series = td.uniform(pred, rate, dims=dims, t_start=float(times[0]))
            return PlotTrack(series=series, renderer="rps", hints={"title": "rps_pred"})
    return PlotTrack(series=pred_series, renderer="rps", hints={"title": "rps_pred"})


def _salience_track(frame: td.Frame, hints: dict[str, Any]) -> PlotTrack:
    series = frame["salience"]
    freqs = hints.get("freqs")
    if freqs is None:
        n_bins = int(np.shape(series.data)[0])
        freqs = np.arange(1, n_bins + 1, dtype=np.float64)
        warnings.warn(
            "plots.dwym: salience entry has no 'freqs' hint — using bin indices "
            "for the y-axis; pass freqs=<bin centre frequencies> for real units",
            stacklevel=4,
        )
    track_hints: dict[str, Any] = {
        "freqs": np.asarray(freqs, dtype=np.float64),
        "title": "salience",
    }
    if "rps_pred" in hints:
        track_hints["rps_pred"] = hints["rps_pred"]
    return PlotTrack(series=series, renderer="salience", hints=track_hints)


def _tracks_for(frame: td.Frame, route: str, hints: dict[str, Any]) -> list[PlotTrack] | None:
    """Track list for one frame under ``route`` (``None`` = plot_timeframe default)."""
    if route == "se":
        return se_comparison_tracks(frame, fmax=hints.get("fmax"))
    if route == "salience":
        tracks: list[PlotTrack] = []
        if "audio" in frame:
            tracks.append(_spectrogram_track(frame, "audio", hints))
        tracks.append(_salience_track(frame, hints))
        if "rps" in frame:
            tracks.append(PlotTrack(series=frame["rps"], hints={"title": "rps (GT)"}))
        return tracks
    if route == "noise_gen":
        labeled = {
            "real": first_channel(frame["audio"]),
            "generated": first_channel(frame["generated"]),
        }
        tracks = list(noise_gen_comparison_tracks(labeled, fmax=hints.get("fmax")))
        if "rps" in frame:
            tracks.append(PlotTrack(series=frame["rps"], hints={"title": "rps"}))
        return tracks
    if route == "rps":
        tracks = [_spectrogram_track(frame, "audio", hints)]
        if "rps" in frame:
            tracks.append(PlotTrack(series=frame["rps"], hints={"title": "rps"}))
        if "rps_pred" in frame:
            tracks.append(_aligned_rps_pred_track(frame))
        return tracks
    if route == "audio":
        return [
            _spectrogram_track(frame, "audio", hints),
            PlotTrack(series=frame["audio"], hints={"title": "audio"}),
        ]
    return None  # "timeframe": let plot_timeframe pick its default tracks


def _explicit_tracks(frame: td.Frame, route: str, hints: dict[str, Any]) -> list[PlotTrack]:
    """Like :func:`_tracks_for` but never ``None`` (for multi-frame merging)."""
    tracks = _tracks_for(frame, route, hints)
    if tracks is not None:
        return tracks
    out = []
    for name in _temporal_entries(frame):
        entry = frame[name]
        assert isinstance(entry, td.Series)
        out.append(PlotTrack(series=entry, hints={"title": name}))
    return out


def _collect_audio(frames: dict[str | None, td.Frame]) -> dict[str, tuple[np.ndarray, int]]:
    out: dict[str, tuple[np.ndarray, int]] = {}
    for label, frame in frames.items():
        for name in AUDIO_ENTRIES:
            if name not in frame:
                continue
            series = frame[name]
            if not (isinstance(series, td.Series) and isinstance(series.tindex, td.GridIndex)):
                continue
            key = name if label is None else f"{label}/{name}"
            out[key] = (to_mono(series), sample_rate_of(series))
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def dwym(obj: td.Frame | Sequence[td.Frame] | dict[str, td.Frame], **hints: Any) -> DwymResult:
    """Plot ``obj`` by what it looks like. See the module docstring.

    Parameters
    ----------
    obj
        One ``td.Frame``, a list of Frames, or a ``{label: Frame}`` dict.
    **hints
        ``renderer=<route>`` forces a dispatch route; canonical-entry
        string hints (``rps="motor_speed"``) remap entry names via
        :func:`plots.coerce.coerce_frame` (and silence its warning);
        everything else flows to ``plot_timeframe``/the renderers.
    """
    forced = hints.pop("renderer", None)
    if forced is not None and forced not in _ROUTES:
        raise ValueError(f"Unknown renderer {forced!r}; known routes: {list(_ROUTES)}")
    remaps = {
        k: hints.pop(k) for k in list(hints) if k in CANONICAL_ENTRIES and isinstance(hints[k], str)
    }

    frames: dict[str | None, td.Frame]
    if isinstance(obj, td.Frame):
        frames = {None: obj}
    elif isinstance(obj, dict):
        frames = {str(label): f for label, f in obj.items()}
    elif isinstance(obj, (list, tuple)):
        frames = {f"frame {i}": f for i, f in enumerate(obj)}
    else:
        raise TypeError(
            f"dwym expects a td.Frame, a list of Frames, or a dict of Frames, "
            f"got {type(obj).__name__}"
        )
    if not frames:
        raise ValueError("dwym got no frames")
    frames = {
        label: coerce_frame(f, **{k: v for k, v in remaps.items() if v in dict(f.items())})
        for label, f in frames.items()
    }

    # Consumed by track builders, not by plot_timeframe.
    build_hints = {k: hints.pop(k) for k in ("fmax", "freqs") if k in hints}

    if len(frames) == 1:
        frame = next(iter(frames.values()))
        route = forced or _route_for(frame)
        tracks = _tracks_for(frame, route, build_hints)
        fig = plot_timeframe(frame, tracks=tracks, **hints)
        figures = [fig]
    else:
        routes = {label: (forced or _route_for(f)) for label, f in frames.items()}
        unique = sorted(set(routes.values()))
        if forced is None and len(frames) == 2 and unique == ["audio"]:
            # A two-frame dict of bare audio: real-vs-generated grid.
            route = "noise_gen"
            labeled = {
                str(label): first_channel(f["audio"])  # type: ignore[index]
                for label, f in frames.items()
            }
            base = next(iter(frames.values()))
            tracks = list(noise_gen_comparison_tracks(labeled, fmax=build_hints.get("fmax")))
            figures = [plot_timeframe(base, tracks=tracks, **hints)]
        elif len(unique) == 1:
            # Homogeneous: one aligned figure, a row block per label.
            sub_route = unique[0]
            route = f"multi:{sub_route}"
            merged: list[PlotTrack] = []
            for label, f in frames.items():
                for track in _explicit_tracks(f, sub_route, build_hints):
                    title = track.hints.get("title") or "track"
                    merged.append(_retitle(track, f"{label} — {title}"))
            base = next(iter(frames.values()))
            figures = [plot_timeframe(base, tracks=merged, **hints)]
        else:
            # Heterogeneous mix: one figure per frame.
            route = "mixed"
            figures = []
            for label, f in frames.items():
                fig = plot_timeframe(f, tracks=_tracks_for(f, routes[label], build_hints), **hints)
                fig.suptitle(str(label))
                figures.append(fig)

    result = DwymResult(figures=figures, audio=_collect_audio(frames), route=route)
    if _in_ipython():
        # Prevent the inline backend from auto-displaying the figures a second
        # time — closed figures still render through `display(fig)`.
        for fig in figures:
            plt.close(fig)
    return result
