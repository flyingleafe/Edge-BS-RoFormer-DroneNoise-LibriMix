"""Evaluation-protocol window specs as DATA — loaders are injected by callers.

This module declares the frozen RPS-tracking evaluation protocols (which
recordings, which windows, which pools, on which grid) as plain declarative
structures. It deliberately contains NO audio/telemetry loading: the tracking
package must not import ``data_processing`` (the ``tracking stays pure``
import-linter contract), so scripts keep their loaders and consume the specs
from here (``scripts/beatvk_vk_arms.py``, ``scripts/vk_validation.py``,
``scripts/rps_eval.py``).

Protocols:

``beatvk``
    The frozen beat-VK campaign protocol on the published ``beatvk-valid-raw``
    dataset: 4 recordings (3 DREGON free-flight room1 + FLY124), contiguous
    non-overlapping 16 s windows tiling the frozen eval span, regime-tagged
    ground/warmup/cruise, scored on the fixed 0.032 s frame grid vs RAW
    telemetry (``scripts/beatvk_eval.py`` is the metric half). The window
    bounds live in the dataset manifest, so :func:`iter_windows` takes the
    manifest as input for this protocol — the spec here declares everything
    that is NOT per-window data: the grid, the regime thresholds, the pools
    (``dregon_cruise`` / ``fly124_cruise``), the frozen FLY124 alignment.

``vk37``
    The coupled-VK DREGON validation protocol (design
    ``docs/vk-order-tracking-design.md`` §5.1, ``scripts/vk_validation.py``):
    the 5 DREGON ``free-flight_*_room1`` recordings that carry
    ``motors_measured``, ONE 25 s mid-in-flight segment per recording
    (in-flight mask: median command AND measured > 30 rev/s), metrics on the
    0.032 s grid with 0.5 s edge trim, GT smoothing = 8-frame (0.25 s) boxcar.
    Segment bounds are derived by the loader (mid-flight placement), so the
    window specs carry ``start_s = end_s = None``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "BEATVK",
    "BEATVK_REPORT_POOLS",
    "FROZEN_FLY124_ALIGNMENT",
    "PROTOCOLS",
    "VK37",
    "PoolSpec",
    "ProtocolSpec",
    "WindowSpec",
    "get_protocol",
    "iter_windows",
    "pit_align",
    "pool_means",
    "regime_of",
    "slice_window",
    "to_frame",
    "window_name",
]

#: Regime thresholds on a window's mean rev/s (the publish-time tagging rule
#: of ``beatvk-valid-raw``): < 5 ground, < 45 warmup, else cruise.
REGIME_GROUND_BELOW = 5.0
REGIME_WARMUP_BELOW = 45.0

#: The hand-tuned FLY124 ``(time_offset, time_dilation)`` the FROZEN beat-VK
#: artifacts were built with (pre-2026-07-31-calibration; implicit rev/s
#: scale 1.0). The shipped ``sources/michaels.py`` constants have since
#: moved, so any bit-exact rebuild of the frozen protocol must pin these.
FROZEN_FLY124_ALIGNMENT: tuple[float, float] = (-20.84, 1.001)


def regime_of(mean_rps: float) -> str:
    """Regime tag of a window from its mean rev/s (publish-time rule)."""
    if mean_rps < REGIME_GROUND_BELOW:
        return "ground"
    if mean_rps < REGIME_WARMUP_BELOW:
        return "warmup"
    return "cruise"


def window_name(recording_id: str, index: int) -> str:
    return f"{recording_id}__w{index:02d}"


@dataclass(frozen=True)
class WindowSpec:
    """One evaluation window of one protocol.

    ``start_s`` / ``end_s`` are recording-absolute seconds; ``None`` means the
    bounds are derived by the protocol's loader (the vk37 mid-flight
    segment). ``regime`` is the manifest tag (``ground``/``warmup``/
    ``cruise``) where the protocol has one.
    """

    protocol: str
    recording_id: str
    index: int
    start_s: float | None = None
    end_s: float | None = None
    regime: str | None = None
    mean_rps: float | None = None

    @property
    def name(self) -> str:
        return window_name(self.recording_id, self.index)


@dataclass(frozen=True)
class PoolSpec:
    """A named pool = a (recordings, regimes, window indices) window filter.

    ``None`` means "any". Pool means are unweighted over member windows
    (the ``beatvk_eval`` convention).
    """

    name: str
    recordings: frozenset[str] | None = None
    regimes: frozenset[str] | None = None
    windows: frozenset[int] | None = None

    def matches(
        self, recording_id: str, regime: str | None = None, index: int | None = None
    ) -> bool:
        """Membership of one window given its identity fields."""
        if self.recordings is not None and recording_id not in self.recordings:
            return False
        if self.regimes is not None and (regime is None or regime not in self.regimes):
            return False
        return not (self.windows is not None and (index is None or index not in self.windows))

    def contains(self, spec: WindowSpec) -> bool:
        return self.matches(spec.recording_id, spec.regime, spec.index)


@dataclass(frozen=True)
class ProtocolSpec:
    """One protocol's frozen constants (everything that is not per-window)."""

    name: str
    dataset: str  #: dload dataset the recordings come from
    recordings: tuple[str, ...]
    sr: int  #: evaluation audio rate (Hz)
    hop_s: float  #: the fixed evaluation frame grid (s)
    window_s: float  #: nominal window/segment length (s)
    edge_trim_s: float  #: metric exclusion at window edges (s)
    n_rotors: int = 4
    pools: Mapping[str, PoolSpec] = field(default_factory=dict)
    #: vk37 only: in-flight mask threshold on the median telemetry (rev/s).
    min_motor_rps: float | None = None
    #: vk37 only: GT boxcar length on the frame grid (frames).
    smooth_frames: int | None = None

    @property
    def hop_samples(self) -> int:
        return int(round(self.hop_s * self.sr))

    def pools_of(self, spec: WindowSpec) -> tuple[str, ...]:
        return tuple(name for name, pool in self.pools.items() if pool.contains(spec))


#: The three DREGON recordings of the frozen beat-VK protocol.
BEATVK_DREGON_RECS: tuple[str, ...] = (
    "free-flight_nosource_room1",
    "free-flight_speech-low_room1",
    "free-flight_whitenoise-low_room1",
)
BEATVK_FLY124_REC = "FLY124"

BEATVK = ProtocolSpec(
    name="beatvk",
    dataset="beatvk-valid-raw",
    recordings=(*BEATVK_DREGON_RECS, BEATVK_FLY124_REC),
    sr=16000,
    hop_s=0.032,  # 512 samples @ 16 kHz — the fixed evaluation grid
    window_s=16.0,
    edge_trim_s=0.5,
    pools={
        "dregon_cruise": PoolSpec(
            "dregon_cruise", frozenset(BEATVK_DREGON_RECS), frozenset({"cruise"})
        ),
        "fly124_cruise": PoolSpec(
            "fly124_cruise", frozenset({BEATVK_FLY124_REC}), frozenset({"cruise"})
        ),
        "cruise": PoolSpec("cruise", None, frozenset({"cruise"})),
        "warmup": PoolSpec("warmup", None, frozenset({"warmup"})),
        "all": PoolSpec("all"),
    },
)

#: The 5 DREGON recordings that carry ``motors_measured`` (ground truth).
VK37_RECS: tuple[str, ...] = (
    "free-flight_nosource_room1",
    "free-flight_speech-low_room1",
    "free-flight_speech-high_room1",
    "free-flight_whitenoise-low_room1",
    "free-flight_whitenoise-high_room1",
)

VK37 = ProtocolSpec(
    name="vk37",
    dataset="DREGON",
    recordings=VK37_RECS,
    sr=16000,
    hop_s=0.032,
    window_s=25.0,  # ONE mid-recording in-flight segment per recording
    edge_trim_s=0.5,
    pools={"all": PoolSpec("all")},
    min_motor_rps=30.0,
    smooth_frames=8,  # 0.25 s boxcar on the frame grid
)

#: The pools the beat-VK ALTERNATION report tabulates
#: (``scripts/beatvk_flagship.py``): the protocol's own headline pools plus
#: the FLY124 warmup regime and the DREGON ramp/steady window split (window 0
#: of a DREGON recording is the takeoff ramp, windows 1-2 are steady flight).
BEATVK_REPORT_POOLS: dict[str, PoolSpec] = {
    "dregon_cruise": BEATVK.pools["dregon_cruise"],
    "fly124_cruise": BEATVK.pools["fly124_cruise"],
    "fly124_warmup": PoolSpec(
        "fly124_warmup", frozenset({BEATVK_FLY124_REC}), frozenset({"warmup"})
    ),
    "dregon_ramp": PoolSpec("dregon_ramp", frozenset(BEATVK_DREGON_RECS), windows=frozenset({0})),
    "dregon_steady": PoolSpec(
        "dregon_steady", frozenset(BEATVK_DREGON_RECS), windows=frozenset({1, 2})
    ),
    "all": BEATVK.pools["all"],
}

PROTOCOLS: dict[str, ProtocolSpec] = {p.name: p for p in (BEATVK, VK37)}


def get_protocol(name: str) -> ProtocolSpec:
    try:
        return PROTOCOLS[name]
    except KeyError:
        raise KeyError(f"unknown protocol {name!r}; known: {sorted(PROTOCOLS)}") from None


def _manifest_windows(entry: Any) -> list[Mapping[str, Any]]:
    """Accept both manifest shapes: ``[...]`` or ``{"windows": [...]}``."""
    if isinstance(entry, Mapping):
        return list(entry["windows"])
    return list(entry)


def iter_windows(
    protocol: str | ProtocolSpec,
    manifest: Mapping[str, Any] | None = None,
    *,
    recordings: set[str] | None = None,
) -> Iterator[WindowSpec]:
    """Yield the protocol's :class:`WindowSpec` rows in canonical order.

    ``beatvk``: ``manifest`` is REQUIRED — the per-recording window table of
    the frozen dataset (``{rid: {"windows": [...]}}`` — exactly the
    ``manifest.json["recordings"]`` shape of ``scripts/beatvk_vk_arms.py``,
    or ``{rid: [window dicts]}``). The window bounds are dataset facts, not
    code: this module never re-derives them.

    ``vk37``: fully static — one window per recording with loader-derived
    bounds (``start_s = end_s = None``).

    ``recordings`` restricts to a subset (unknown ids raise).
    """
    spec = get_protocol(protocol) if isinstance(protocol, str) else protocol
    if recordings is not None:
        unknown = recordings - set(spec.recordings)
        if unknown:
            raise KeyError(f"unknown recordings {sorted(unknown)}; known: {spec.recordings}")
    rids = [r for r in spec.recordings if recordings is None or r in recordings]

    if spec.name == "vk37":
        for rid in rids:
            yield WindowSpec(protocol=spec.name, recording_id=rid, index=0, regime="cruise")
        return

    if manifest is None:
        raise ValueError(
            f"protocol {spec.name!r} needs the dataset window manifest "
            "(e.g. beatvk_vk_arms.load_manifest(...)['recordings'])"
        )
    for rid in rids:
        if rid not in manifest:
            continue
        for w in _manifest_windows(manifest[rid]):
            yield WindowSpec(
                protocol=spec.name,
                recording_id=rid,
                index=int(w["index"]),
                start_s=float(w["start_s"]),
                end_s=float(w["end_s"]),
                regime=str(w["regime"]),
                mean_rps=float(w["mean_rps"]) if "mean_rps" in w else None,
            )


def to_frame(
    audio: Any,
    sr: float | int,
    spec: WindowSpec,
    *,
    rps: Any | None = None,
    frame_times: Any | None = None,
    rps_meas: Any | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Any:
    """Build the canonical tracking frame for one protocol window.

    A thin wrapper over :func:`tracking.stages.tracking_frame` that stamps the
    window's identity into the frame meta (``protocol`` / ``recording_id`` /
    ``window_index`` / ``regime`` / ``start_s`` / ``end_s``). ``audio`` is
    ``(T,)`` or ``(C, T)`` at ``sr``; ``rps`` / ``rps_meas`` are ``(R, N)``
    on the ``frame_times`` grid. Imported lazily so the spec tables stay
    importable without torch.
    """
    from tracking.stages import tracking_frame

    stamp: dict[str, Any] = {
        "protocol": spec.protocol,
        "recording_id": spec.recording_id,
        "window_index": spec.index,
        "regime": spec.regime,
        "start_s": spec.start_s,
        "end_s": spec.end_s,
    }
    stamp.update(dict(meta or {}))
    return tracking_frame(
        audio, sr, rps=rps, frame_times=frame_times, rps_meas=rps_meas, meta=stamp
    )


def slice_window(
    audio: Any,
    sr: float | int,
    spec: WindowSpec,
    ts: Any | None = None,
    vals: Any | None = None,
    *,
    seconds: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Cut one protocol window out of a recording: ``(seg, ft, r_meas, edge)``.

    THE window slicer of the protocols — every caller that turns a whole
    recording into a window's arrays goes through here, so the frame grid and
    the edge mask cannot drift between the dataset builder and its consumers.

    ``audio`` is ``(T,)`` or ``(C, T)`` at ``sr``, ALREADY at the protocol's
    rate (the resample is a ``data_processing`` concern and stays with the
    caller). ``ts`` / ``vals`` are the recording's telemetry stamps and
    ``(R, N)`` values; given, they are linearly interpolated onto the window
    frame grid as ``r_meas`` (else ``None``). The grid is ``spec``'s protocol
    hop from the window start, and ``edge`` masks the protocol's edge trim off
    both ends. ``seconds`` truncates the slice (smoke runs only).
    """
    p = get_protocol(spec.protocol)
    a = np.atleast_2d(np.asarray(audio))
    start = float(spec.start_s or 0.0)
    end = float(spec.end_s or 0.0)
    a0, a1 = int(round(start * sr)), int(round(end * sr))
    if seconds is not None:
        a1 = min(a1, a0 + int(round(seconds * sr)))
    if not (0 <= a0 < a1 <= a.shape[-1]):
        raise ValueError(
            f"{spec.name}: window [{start}, {end}] outside the {a.shape[-1]}-sample recording"
        )
    seg = a[:, a0:a1]
    ft = np.arange(0.0, (a1 - a0) / sr - p.hop_s / 2, p.hop_s)
    r_meas = None
    if ts is not None and vals is not None:
        vals = np.asarray(vals, dtype=np.float64)
        r_meas = np.stack(
            [
                np.interp(ft + start, np.asarray(ts, dtype=np.float64), vals[i])
                for i in range(p.n_rotors)
            ]
        )
    edge = (ft > p.edge_trim_s) & (ft < ft[-1] - p.edge_trim_s)
    return seg, ft, r_meas, edge


def pit_align(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """``(pred with its rotor rows permuted onto gt, the permutation)``.

    THE permutation-invariant rotor assignment of the tracking stack: one
    Hungarian match on the per-rotor-pair MSE, which is identical to the
    brute-force PIT search over all ``R!`` permutations.  ``pred`` and ``gt``
    are ``(R, F)`` with the rotor axis FIRST and the same frame count.
    ``losses.pit.align_rps_to_gt`` (the frame-level entry point, which also
    resamples and shape-guards) delegates here.
    """
    from scipy.optimize import linear_sum_assignment

    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    cost = np.mean((pred[:, None, :] - gt[None, :, :]) ** 2, axis=-1)  # (R, R)
    row, col = linear_sum_assignment(cost)
    perm = np.empty(gt.shape[0], dtype=int)
    perm[col] = row
    return pred[perm], [int(v) for v in perm]


def pool_means(
    rows: Iterable[Mapping[str, Any]],
    pools: Mapping[str, PoolSpec],
    *,
    key: str = "mae",
    ndigits: int | None = None,
) -> dict[str, float | None]:
    """Unweighted mean of ``row[key]`` over each pool's member windows.

    A row is one scored window: ``recording`` / ``regime`` / ``window`` plus
    the metric. A pool with no member window scores ``None``.
    """
    rows = list(rows)
    out: dict[str, float | None] = {}
    for name, pool in pools.items():
        sel = [
            float(r[key])
            for r in rows
            if pool.matches(str(r["recording"]), r.get("regime"), r.get("window"))
        ]
        mean = float(np.mean(sel)) if sel else None
        out[name] = mean if mean is None or ndigits is None else round(mean, ndigits)
    return out
