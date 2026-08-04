"""Canonicalize raw dataset frames before ``plots.dwym`` dispatch.

Raw recording frames name their entries after the source dataset, not after
the canonical plotting vocabulary. Examples of the real entry names:

* DREGON raw frames (``data_processing.sources.dregon``): ``audio``,
  ``motors_command`` (cleaned, canonical), ``motors_measured``,
  ``imu_accel`` / ``imu_gyro`` / ``source_position``.
* Michael's rich frames (``data_processing.sources.michaels``): ``audio``,
  ``rps`` (calibrated), plus one entry per DJI CSV sensor block — including
  ``motor_speed`` (RAW uncalibrated RPM; the calibrated track is ``rps``).

:func:`coerce_frame` renames known aliases to the canonical names, keeps
every other entry untouched (extra timeseries stay as additional tracks),
and prints one warning line per applied mapping so a wrong guess is
visible. Explicit ``overrides`` (``coerce_frame(frame, rps="motor_speed")``)
apply silently and beat the alias tables.
"""

from __future__ import annotations

import warnings

import tdseries as td

__all__ = ["CANONICAL_ENTRIES", "ENTRY_ALIASES", "coerce_frame"]

#: The canonical entry vocabulary ``plots.dwym`` dispatches on.
CANONICAL_ENTRIES = (
    "audio",
    "rps",
    "rps_pred",
    "mixture",
    "target",
    "enhanced",
    "salience",
    "generated",
)

#: Per-canonical-name alias tables, in priority order. The ``rps`` order
#: mirrors ``data_processing.frames.PUBLISHED_RPS_KEYS`` (``motors_command``
#: is DREGON's cleaned canonical track, ``motors_measured`` the raw one).
#: ``motor_speed`` is last on purpose: on Michael's frames it is the RAW
#: RPM block and only wins when no better track exists.
ENTRY_ALIASES: dict[str, tuple[str, ...]] = {
    "audio": ("waveform", "wav", "mix"),
    "rps": (
        "motors_command",
        "motors_measured",
        "motor_rps",
        "rotor_rps",
        "rotor_speed",
        "motor_speed",
    ),
    "rps_pred": ("pred_rps", "predicted_rps", "rps_prediction"),
    "salience": ("salience_map",),
    "generated": ("generated_audio", "gen_audio"),
}

#: Minimum ``GridIndex`` rate (Hz) for the pick-the-sole-waveform audio
#: fallback — telemetry blocks log well below this, audio well above.
_AUDIO_RATE_FLOOR = 4000.0


def _warn(alias: str, canonical: str) -> None:
    warnings.warn(
        f"plots.coerce: using entry {alias!r} as {canonical!r} — "
        f"pass {canonical}={alias!r} to make the mapping explicit",
        stacklevel=3,
    )


def _audio_candidates(entries: dict[str, object]) -> list[str]:
    """Entry names that look like a raw waveform (audio-rate GridIndex)."""
    names = []
    for name, value in entries.items():
        if not isinstance(value, td.Series):
            continue
        tindex = value.tindex if value.has_time else None
        if not isinstance(tindex, td.GridIndex):
            continue
        if float(tindex.rate) < _AUDIO_RATE_FLOOR:
            continue
        dims = [d for d in value.dims if d is not None and d != "time"]
        if dims and (len(dims) > 1 or dims[0] not in ("mic", "channel")):
            continue
        names.append(name)
    return names


def coerce_frame(frame: td.Frame, **overrides: str) -> td.Frame:
    """Return ``frame`` with alias entries renamed to canonical names.

    Parameters
    ----------
    frame
        Any ``td.Frame`` (raw recording frame, dataset sample, model output).
    **overrides
        ``canonical_name="actual_entry_name"`` mappings. Applied silently
        (no warning) and before the alias tables. An override whose source
        entry is missing raises ``ValueError``. If the canonical name is
        also present in ``frame``, the existing entry moves aside to
        ``"<canonical>_orig"`` so nothing is dropped.

    Behavior:

    * Alias tables (:data:`ENTRY_ALIASES`) map known source-specific names
      to the canonical vocabulary; the first present alias wins and one
      warning line names the applied mapping.
    * When no ``audio``/``mixture`` entry exists, a single entry that looks
      like a waveform (uniform, >= 4 kHz, dims within mic/channel + time)
      is renamed to ``audio`` (with a warning). Two or more candidates are
      ambiguous and left alone.
    * Every entry that is not renamed is KEPT unchanged — unknown extra
      timeseries stay as additional tracks.
    """
    bad = sorted(set(overrides) - set(CANONICAL_ENTRIES))
    if bad:
        raise ValueError(f"Unknown canonical entry name(s) {bad}; known: {list(CANONICAL_ENTRIES)}")

    entries: dict[str, object] = dict(frame.items())
    renames: dict[str, str] = {}  # source name -> canonical name

    # 1. Explicit overrides — silent, highest priority.
    for canonical, source in overrides.items():
        if source not in entries:
            raise ValueError(f"Override {canonical}={source!r}: no such entry in frame")
        if source != canonical:
            renames[source] = canonical
            if canonical in entries:
                # Move the pre-existing canonical entry aside, keep both.
                renames.setdefault(canonical, f"{canonical}_orig")

    # 2. Alias tables.
    taken = set(overrides)
    for canonical, aliases in ENTRY_ALIASES.items():
        if canonical in taken or canonical in entries:
            continue
        for alias in aliases:
            if alias in entries and alias not in renames:
                renames[alias] = canonical
                _warn(alias, canonical)
                break

    # 3. Sole-waveform fallback for the audio entry.
    if "audio" not in entries and "audio" not in renames.values() and "mixture" not in entries:
        candidates = [n for n in _audio_candidates(entries) if n not in renames]
        if len(candidates) == 1:
            renames[candidates[0]] = "audio"
            _warn(candidates[0], "audio")

    if not renames:
        return frame
    return td.Frame({renames.get(name, name): value for name, value in entries.items()})
