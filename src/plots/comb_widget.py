"""The comb explorer as a notebook widget over a plain ``tdseries`` Frame.

    import tdseries as td
    from plots.comb_widget import comb_explorer

    comb_explorer(frame, t0=22.5, dur=16.0)

``frame`` is whatever Frame the caller already has — a DREGON recording, one
of Michael's flights, a Frame they built themselves — sliced however they like.
Nothing about the recording is looked up from disk, and nothing assumes 4
rotors or 8 microphones.

Two things are discovered from the Frame instead of being configured:

* **the audio**: the uniformly-sampled entry with a ``mic`` / ``channel`` axis
  (or a bare mono ``("time",)`` entry) at the highest sample rate
* **every rotor-speed track**: each ``(rotor, time)`` entry, uniform or
  event-sampled, whose values are plausible rev/s.  On a DREGON frame that is
  ``motors_measured``, ``motors_command`` and ``motors_command_raw``; on
  Michael's it is ``rps``; a refined track the caller added with
  ``frame.with_entry("rps_refined", ...)`` appears by itself.

**Every discovered track ships into the page as a selectable carrier.**  Which
one drives the combs and the strip demodulation is a dropdown in the page, so
telemetry vs command vs a refined track is a by-eye comparison, not a rebuild.

**EVERY microphone of the array ships into the SAME page** and the channel
selector is in-page state: a spectrogram costs about a hundred kilobytes, so
all eight mics of a DREGON array are selectable without a rebuild.  The
demodulated STRIPS are the megabyte-scale product, so they are built for
``strip_channels`` only (default: the average plus the loudest mic); ask for
more with ``strip_channels="all"``.  The CLI splits channels across sibling
files because a page it writes has a 9 MB budget; a notebook has none, so the
only cost is the size of the cell output — which is reported, and warned about
above ``warn_mb`` because the payload is saved into the ``.ipynb``.

The page itself (payload builder + the one HTML/JS template) is
:mod:`plots.comb_page`, shared with ``scripts/displacement/comb_explorer.py``.
"""

from __future__ import annotations

import html as _html
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tdseries as td

from plots.comb_page import (
    DECIM,
    Carrier,
    PageOptions,
    Slice,
    build_payload,
    channel_label,
    parse_channels,
    payload_bytes,
    render_html,
)

#: Dim names that mark a per-rotor track.  A Frame that uses none of them falls
#: back to a name-and-magnitude heuristic (see :func:`discover_rps`), but the
#: dim name is the strong signal: every adapter in this repo builds rotor
#: tracks with ``dims=("rotor", "time")``.
ROTOR_DIMS = ("rotor", "motor", "prop", "propeller")

#: Dim names that mark a microphone axis.
MIC_DIMS = ("mic", "channel", "mics", "ch")

#: Entry-name pattern for the fallback rotor-track heuristic.
RPS_NAME_RE = re.compile(r"(?i)\brps\b|rps|rotor|motor(?!_)|rpm|refin")


@dataclass
class Found:
    """What :func:`discover` read out of a Frame — printed, not hidden."""

    audio_key: str
    audio_sr: float
    n_mics: int
    rps_keys: list[str]
    n_rotors: int
    rejected: dict[str, str]

    def describe(self) -> str:
        lines = [
            f"[frame] audio: {self.audio_key!r} at {self.audio_sr:g} Hz, {self.n_mics} mic",
            f"[frame] rotor-speed tracks ({self.n_rotors} rotors): "
            + ", ".join(repr(k) for k in self.rps_keys),
        ]
        if self.rejected:
            lines.append(
                "[frame] not used: "
                + ", ".join(f"{k} ({why})" for k, why in sorted(self.rejected.items()))
            )
        return "\n".join(lines)


# ─── Frame introspection ──────────────────────────────────────────────────────


def _series_entries(frame: td.Frame) -> list[tuple[str, td.Series]]:
    """``(name, Series)`` for every TEMPORAL Series entry, in frame order.

    Nested Frames (``meta``) and atemporal entries (``mic_pos``,
    ``rotor_pos``, ``audio_timestamps``) are skipped.
    """
    out: list[tuple[str, td.Series]] = []
    for key in frame:
        entry = frame[key]
        if not isinstance(entry, td.Series):
            continue
        if "time" not in entry.dims:
            continue
        out.append((key, entry))
    return out


def _rate_hz(s: td.Series) -> float:
    """Samples (or events) per second — the one comparable number across a
    uniform and an event-sampled series."""
    idx = s.tindex
    if isinstance(idx, td.GridIndex):
        return float(idx.sr)
    n = int(s.shape[s.dims.index("time")])
    return n / max(float(s.duration), 1e-9)


def discover_audio(frame: td.Frame) -> str:
    """The name of the audio entry: uniform, with a mic axis, highest rate."""
    best: tuple[float, int, str] | None = None
    for key, s in _series_entries(frame):
        if not isinstance(s.tindex, td.GridIndex):
            continue
        others = [d for d in s.dims if d != "time"]
        if len(others) > 1:
            continue
        if others and others[0] not in MIC_DIMS:
            continue
        n_ch = 1 if not others else int(s.shape[s.dims.index(others[0])])
        cand = (float(s.tindex.sr), n_ch, key)
        if best is None or cand > best:
            best = cand
    if best is None:
        raise ValueError(
            "no audio entry in this frame: expected a uniformly-sampled Series with a "
            f"{'/'.join(MIC_DIMS[:2])} axis (or a mono ('time',) Series); "
            f"entries are {[k for k, _ in _series_entries(frame)]}"
        )
    return best[2]


def discover_rps(
    frame: td.Frame,
    audio_key: str,
    *,
    rps_range: tuple[float, float] = (0.0, 150.0),
) -> tuple[list[str], dict[str, str]]:
    """``(rotor-speed entry names, {rejected name: why})``.

    A candidate is a 2-D ``(D, "time")`` Series, uniform or event-sampled,
    sampled slower than the audio, whose median value is a plausible rev/s.
    ``D`` must be a rotor dim; only if the frame has NO such entry at all does
    the search fall back to any named non-mic axis whose entry name looks like
    a rotor track, which is what keeps Michael's ``motor_volts`` /
    ``motor_esctemp`` / ``motorctrl_pwm`` blocks — all of them ``(channel,
    time)`` with a median inside the rev/s range — out of the carrier list.
    """
    lo, hi = rps_range
    audio = frame[audio_key]
    audio_rate = _rate_hz(audio)
    strong: list[str] = []
    weak: list[str] = []
    rejected: dict[str, str] = {}

    for key, s in _series_entries(frame):
        if key == audio_key:
            continue
        others = [d for d in s.dims if d != "time"]
        if len(s.dims) != 2 or len(others) != 1 or others[0] is None:
            rejected[key] = f"dims {s.dims} are not (rotor, time)"
            continue
        dim = others[0]
        if dim in MIC_DIMS and dim not in ROTOR_DIMS:
            name_ok = bool(RPS_NAME_RE.search(key))
        else:
            name_ok = True
        if _rate_hz(s) >= audio_rate:
            rejected[key] = "sampled at or above the audio rate"
            continue
        data = np.asarray(s.data, dtype=np.float64)
        if not data.size or not np.isfinite(data).any():
            rejected[key] = "no finite values"
            continue
        med = float(np.nanmedian(data))
        if not (lo <= med <= hi):
            rejected[key] = f"median {med:.3g} outside {lo:g}..{hi:g} rev/s"
            continue
        if dim in ROTOR_DIMS:
            strong.append(key)
        elif name_ok:
            weak.append(key)
        else:
            rejected[key] = f"axis {dim!r} is not a rotor axis"

    if strong:
        for k in weak:
            rejected.setdefault(k, "a rotor-dim track was found, so this is not needed")
        return strong, rejected
    return weak, rejected


def discover(
    frame: td.Frame,
    *,
    rps_keys: list[str] | None = None,
    rps_range: tuple[float, float] = (0.0, 150.0),
) -> Found:
    """Everything the widget reads out of the Frame, as one reportable record."""
    audio_key = discover_audio(frame)
    audio = frame[audio_key]
    others = [d for d in audio.dims if d != "time"]
    n_mics = 1 if not others else int(audio.shape[audio.dims.index(others[0])])
    found, rejected = discover_rps(frame, audio_key, rps_range=rps_range)
    if rps_keys is not None:
        missing = [k for k in rps_keys if k not in frame]
        if missing:
            raise ValueError(f"rps_keys {missing} are not entries of this frame")
        for k in found:
            if k not in rps_keys:
                rejected[k] = "not in rps_keys"
        found = list(rps_keys)
    if not found:
        raise ValueError(
            "no rotor-speed track in this frame: expected a (rotor, time) Series with "
            f"a plausible rev/s median. Candidates examined: {rejected or 'none'}"
        )
    n_rotors = int(np.asarray(frame[found[0]].data).shape[0])
    return Found(audio_key, _rate_hz(audio), n_mics, found, n_rotors, rejected)


# ─── Frame -> Slice + Carriers ────────────────────────────────────────────────


def _audio_block(
    frame: td.Frame, audio_key: str, t_lo: float, t_hi: float
) -> tuple[np.ndarray, int]:
    """``((C, N) float64, sr)`` for the requested absolute window."""
    s = frame[audio_key].time[t_lo:t_hi]
    data = np.asarray(s.data, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    elif s.dims.index("time") != data.ndim - 1:
        data = np.moveaxis(data, s.dims.index("time"), -1)
    if data.shape[1] < 4:
        raise ValueError(f"the requested window holds {data.shape[1]} audio samples")
    idx = s.tindex
    if not isinstance(idx, td.GridIndex):
        raise ValueError(f"{audio_key!r} is not uniformly sampled")
    sr = float(idx.sr)
    if abs(sr - round(sr)) > 1e-9:
        raise ValueError(f"{audio_key!r} has a non-integral sample rate {sr}")
    return np.ascontiguousarray(data), int(round(sr))


def _stamps_and_values(s: td.Series) -> tuple[np.ndarray, np.ndarray]:
    """``(absolute stamps (M,), values (R, M))`` for a rotor-speed Series."""
    data = np.asarray(s.data, dtype=np.float64)
    ti = s.dims.index("time")
    if data.ndim == 1:
        data = data[None, :]
    elif ti != data.ndim - 1:
        data = np.moveaxis(data, ti, -1)
    idx = s.tindex
    if isinstance(idx, td.GridIndex):
        stamps = s.t_start + np.arange(data.shape[-1]) / float(idx.sr)
    elif isinstance(idx, td.StampIndex):
        stamps = np.asarray(idx.abs_stamps, dtype=np.float64)
    else:
        raise ValueError(
            f"a rotor-speed track must be uniform or event-sampled, not {type(idx).__name__}"
        )
    order = np.argsort(stamps, kind="stable")
    return stamps[order], data[:, order]


def carrier_from_series(
    key: str,
    s: td.Series,
    t_abs: np.ndarray,
    *,
    cid: str,
    gap_tol: float = 0.5,
) -> Carrier:
    """One Carrier on the audio grid ``t_abs`` (absolute seconds).

    Event-sampled telemetry is linearly interpolated, but NOT across a hole:
    a gap longer than ``gap_tol`` inside the window, or a window edge outside
    the track's coverage by more than ``gap_tol``, raises instead of drawing a
    straight line the data does not support.  Gaps below the tolerance are
    interpolated and named in the page's provenance panel.
    """
    stamps, vals = _stamps_and_values(s)
    if stamps.size < 2:
        raise ValueError(f"{key!r} has {stamps.size} samples — not a trajectory")
    lo, hi = float(t_abs[0]), float(t_abs[-1])
    left = stamps[0] - lo
    right = hi - stamps[-1]
    if max(left, right) > gap_tol:
        raise ValueError(
            f"{key!r} does not cover the requested window: it spans "
            f"[{stamps[0]:.3f}, {stamps[-1]:.3f}] s and the window is [{lo:.3f}, {hi:.3f}] s "
            f"(short by {max(left, right):.3f} s > gap_tol={gap_tol:g}). Move the window, "
            f"raise gap_tol, or drop this track with rps_keys=."
        )
    inside = (stamps >= lo - gap_tol) & (stamps <= hi + gap_tol)
    span = stamps[inside]
    max_gap = float(np.max(np.diff(span))) if span.size > 1 else 0.0
    if max_gap > gap_tol:
        j = int(np.argmax(np.diff(span)))
        raise ValueError(
            f"{key!r} has a {max_gap:.3f} s hole at [{span[j]:.3f}, {span[j + 1]:.3f}] s inside "
            f"the requested window (gap_tol={gap_tol:g}). Interpolating across it would draw a "
            "comb the telemetry does not support — move the window, raise gap_tol, or drop "
            "this track with rps_keys=."
        )
    g = np.stack([np.interp(t_abs, stamps, vals[r]) for r in range(vals.shape[0])])
    kind = "uniform" if isinstance(s.tindex, td.GridIndex) else "events"
    note = (
        f"{key} ({kind}, {_rate_hz(s):.1f} Hz, {vals.shape[0]} rotor), "
        f"linearly resampled onto the audio grid; largest in-window sample gap "
        f"{max_gap * 1000:.1f} ms"
    )
    if max(left, right) > 0:
        note += f"; edge held constant for {max(max(left, right), 0.0) * 1000:.1f} ms"
    return Carrier(id=cid, label=key, g=g, note=note)


def _auto_channels(audio: np.ndarray) -> list[str]:
    """``avg`` plus the loudest single microphone — the two viewpoints that are
    worth the megabytes of a demodulated strip stack."""
    n = int(audio.shape[0])
    if n == 1:
        return ["mic00"]
    rms = np.sqrt((audio**2).mean(axis=1))
    return ["avg", f"mic{int(np.argmax(rms)):02d}"]


# ─── The widget ───────────────────────────────────────────────────────────────


def build_widget_payload(
    frame: td.Frame,
    *,
    t0: float = 0.0,
    dur: float | None = None,
    absolute: bool = False,
    channels: str | int | list | tuple = "all",
    strip_channels: str | int | list | tuple = "auto",
    rps_keys: list[str] | None = None,
    rps_range: tuple[float, float] = (0.0, 150.0),
    gap_tol: float = 0.5,
    ks: str = "1-100",
    k_max: int = 100,
    segs: tuple[float, ...] = (0.1, 0.5, 2.0),
    decim: int = DECIM,
    ylim: float = 6.0,
    strip_rows: int = 80,
    strip_cols: int = 110,
    strip_floor: float = 60.0,
    strip_top: float = 99.5,
    nfft: int = 2048,
    spec_cols: int = 600,
    fmax: float = 10000.0,
    jobs: int | None = None,
    cache: bool = False,
    cache_dir: str | Path | None = None,
    max_mb: float | None = None,
    verbose: bool = True,
) -> tuple[dict, Found]:
    """``(payload, discovery record)`` — everything except the HTML."""
    found = discover(frame, rps_keys=rps_keys, rps_range=rps_range)
    if verbose:
        print(found.describe(), flush=True)

    audio_series = frame[found.audio_key]
    origin = 0.0 if absolute else float(audio_series.t_start)
    t_lo = origin + float(t0)
    audio_end = float(audio_series.t_start) + float(audio_series.duration)
    t_hi = audio_end if dur is None else t_lo + float(dur)
    if t_hi > audio_end + 1e-9:
        raise ValueError(
            f"window [{t0:g}, {t0 + (t_hi - t_lo):g}] s runs past the audio, which is "
            f"{audio_series.duration:.3f} s long"
        )
    if t_hi - t_lo > 40.0 and verbose:
        print(
            f"[warn] {t_hi - t_lo:.1f} s window: the demodulation is O(window x k x mic) and "
            "the payload grows with it — 16 s is the working length",
            file=sys.stderr,
        )

    audio, sr = _audio_block(frame, found.audio_key, t_lo, t_hi)
    t_abs = t_lo + np.arange(audio.shape[1]) / sr
    rid = "recording"
    if "meta" in frame:
        meta = frame["meta"]
        if "recording_id" in meta:
            rid = str(meta["recording_id"]) or rid
    sl = Slice(
        rid=rid,
        dataset="frame",
        t0=float(t0),
        dur=float(audio.shape[1] / sr),
        sr=sr,
        audio=audio,
        note=f"{found.audio_key} of a tdseries Frame",
    )

    carriers = [
        carrier_from_series(key, frame[key], t_abs, cid=f"c{i}", gap_tol=gap_tol)
        for i, key in enumerate(found.rps_keys)
    ]
    chans = _auto_channels(audio) if channels == "auto" else parse_channels(channels, sl.n_mics)
    strips_on = (
        _auto_channels(audio)
        if strip_channels == "auto"
        else parse_channels(strip_channels, sl.n_mics)
    )
    strips_on = [c for c in strips_on if c in chans] or [chans[0]]

    opts = PageOptions(
        channels=chans,
        strip_channels=strips_on,
        ks=ks,
        k_max=k_max,
        segs=tuple(segs),
        strip_rows=strip_rows,
        strip_cols=strip_cols,
        strip_floor=strip_floor,
        strip_top=strip_top,
        ylim=ylim,
        decim=decim,
        nfft=nfft,
        spec_cols=spec_cols,
        fmax=fmax,
        max_mb=max_mb,
        cache=cache,
        cache_dir=Path(cache_dir) if cache_dir else None,
        **({} if jobs is None else {"jobs": jobs}),
    )
    payload = build_payload(
        sl,
        carriers,
        opts,
        meta_extra={
            "rps_channel": found.rps_keys[0],
            "rps_note": carriers[0].note,
            "channel_label": channel_label(chans[0], sl.n_mics),
            "channel_trade": (
                f"every one of the {len(chans)} microphone channel(s) is selectable in ONE "
                f"page; demodulated strips for {', '.join(strips_on)} (a strip stack costs "
                "megabytes, a spectrogram does not — pass strip_channels= for more); every "
                f"discovered rotor-speed track ({', '.join(found.rps_keys)}) is a selectable "
                "carrier"
            ),
        },
        verbose=verbose,
    )
    return payload, found


def widget_html(
    payload: dict,
    *,
    height: int = 900,
    max_height: int = 2200,
    uid: str | None = None,
) -> str:
    """The payload as ONE iframe element, self-contained and offline.

    A ``srcdoc`` iframe is used instead of a bare fragment because JupyterLab
    does not execute ``<script>`` tags in HTML output that it inserts into its
    own document, and because it makes every element id, every global and every
    event listener private to the instance — two widgets in two cells cannot
    collide.  The base64 payload contains no character that HTML-attribute
    escaping expands, so the escaping costs only the few hundred structural
    quotes of the JSON and the script.

    ``height`` is the height before the page has measured itself; the page then
    sizes the frame to its own content, never past ``max_height`` (which it
    reads from ``data-max-height``).  The cap is what keeps the cell BOUNDED:
    a tall strip stack scrolls inside the frame instead of pushing the rest of
    the notebook off the screen.
    """
    uid = uid or ("c" + uuid.uuid4().hex[:10])
    page = render_html(payload, uid=uid)
    doc = "<!doctype html><meta charset='utf-8'>" + page
    src = _html.escape(doc, quote=True)
    return (
        f'<iframe srcdoc="{src}" data-max-height="{int(max_height)}" '
        f'style="display:block;width:100%;height:{int(height)}px;'
        f"max-height:{int(max_height)}px;border:1px solid #ccc;"
        'border-radius:8px;resize:vertical;overflow:auto"></iframe>'
    )


def comb_explorer(
    frame: td.Frame,
    *,
    t0: float = 0.0,
    dur: float | None = None,
    height: int = 900,
    max_height: int = 2200,
    warn_mb: float = 30.0,
    verbose: bool = True,
    **kwargs,
):
    """Build and DISPLAY the comb explorer for one ``tdseries`` Frame.

    ``t0`` / ``dur`` are seconds measured from the start of the frame's audio
    (pass ``absolute=True`` to give absolute times instead); ``dur=None`` uses
    the whole audio.  Every other keyword is forwarded to
    :func:`build_widget_payload` — ``channels``, ``strip_channels``,
    ``rps_keys``, ``ks``, ``segs``, ``decim``, ``ylim``, ``strip_rows`` /
    ``strip_cols``, ``nfft``, ``fmax``, ``jobs``, ``gap_tol`` and so on.

    ``height`` is the starting height of the cell and ``max_height`` the most
    it can ever take: the page measures its own content and sizes the frame to
    it, between those two numbers, so the cell is stable and bounded.

    Returns ``None``: the page is displayed.  Use
    :func:`build_widget_payload` + :func:`widget_html` when the HTML itself is
    wanted.
    """
    from IPython.display import HTML, display

    payload, _found = build_widget_payload(frame, t0=t0, dur=dur, verbose=verbose, **kwargs)
    html = widget_html(payload, height=height, max_height=max_height)
    n = payload_bytes(payload)
    if verbose:
        print(
            f"[widget] payload {n / 1e6:.1f} MB, page {len(html) / 1e6:.1f} MB "
            f"({len(payload['spec'])} mic channel(s), strips for "
            f"{', '.join(payload['strips'])} x {len(payload['carriers'])} carrier(s) x "
            f"{payload['meta']['n_rotors']} rotor(s) x {len(payload['ks'])} harmonics)",
            flush=True,
        )
    if n > warn_mb * 1e6:
        print(
            f"[warn] this cell's output is {n / 1e6:.0f} MB and notebook outputs are saved "
            f"into the .ipynb. Clear it before committing, or cut the payload with "
            f"strip_channels=, channels=, rps_keys=, ks= or segs=",
            file=sys.stderr,
        )
    display(HTML(html))
    return None
