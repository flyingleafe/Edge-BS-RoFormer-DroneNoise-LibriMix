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

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "BEATVK",
    "FROZEN_FLY124_ALIGNMENT",
    "PROTOCOLS",
    "VK37",
    "PoolSpec",
    "ProtocolSpec",
    "WindowSpec",
    "get_protocol",
    "iter_windows",
    "regime_of",
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
    """A named pool = a (recordings, regimes) filter over a protocol's windows.

    ``None`` means "any". Pool means are unweighted over member windows
    (the ``beatvk_eval`` convention).
    """

    name: str
    recordings: frozenset[str] | None = None
    regimes: frozenset[str] | None = None

    def contains(self, spec: WindowSpec) -> bool:
        if self.recordings is not None and spec.recording_id not in self.recordings:
            return False
        return not (
            self.regimes is not None and (spec.regime is None or spec.regime not in self.regimes)
        )


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
