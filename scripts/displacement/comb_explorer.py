#!/usr/bin/env python3
"""Build the interactive comb explorer page for ANY recording and time slice.

This is the CLI front end.  The page itself — the payload builder and the one
HTML/JS template — lives in :mod:`plots.comb_page`, which is shared with the
notebook widget :mod:`plots.comb_widget`.  Everything below is source
resolution (DREGON and Michael's rigs from disk), refined-label loading, and
the file-per-microphone-channel layout that the size budget forces.

The page shows:

* the spectrogram of the slice, as a plain STFT or SYNCHROSQUEEZED (reassigned)
  in one toggle, with per-rotor harmonic combs from telemetry — solid at
  ``k * g_r(t)``, dashed at ``k * s * g_r(t)`` for a free scale factor ``s``,
  and dotted at the acoustically REFINED trajectory when ``--refined`` is given
* a linked spectrum cut at a draggable time marker, in the same transform
* per-harmonic DEMODULATED strips: the envelope of harmonic ``k`` after
  heterodyning by ``exp(-i k phi_r)``, frequency axis rescaled to a shaft-rate
  offset ``(f - k g) / k`` in rev/s, so the carrier is the centre line.  The
  carrier is telemetry or (with ``--refined``) the refined trajectory, chosen in
  the page.  Strips are always a plain STFT — a reassigned strip would move
  energy along the very axis the strip exists to measure

The scale-factor slider is the instrument the whole page exists for: if the
telemetry-vs-acoustic mismatch is a constant SCALE, one ``s`` puts every dashed
line on every tooth at every time; if it is a time-varying SHAPE error, no
single ``s`` does.

Microphone channels are first class: every page is built for ONE channel (a
single mic, or the incoherent average), named in the header, and the same
channel drives the spectrogram, the spectrum cut and the strips.  This matters
because the common phase-noise term measured in WP18 is predominantly PER-MIC
(cross-channel coherence 0.065 / 0.237 against 0.81-0.94 for a common-mode
control), so a mic-averaged picture is not the whole evidence.  A notebook has
no size budget, so the widget puts every channel in ONE page instead — see
``plots.comb_widget.comb_explorer``.

Usage::

    python scripts/displacement/comb_explorer.py --list
    python scripts/displacement/comb_explorer.py \\
        --recording free-flight_nosource_room1 --t0 22.56481 --dur 16 \\
        --out F0v2.html
    python scripts/displacement/comb_explorer.py \\
        --recording FLY124 --t0 40 --dur 16 --channels all --out-dir pages/

Sources: DREGON (``motors_measured`` when the recording has it, else
``motors_command`` — the page header says which) and Michael's DJI Matrice 100
rig (FLY124 / FLY125, the calibrated ``rps`` track).  Nothing assumes 4 rotors
or 8 microphones; both counts come from the data.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import hk_core as H
import numpy as np
import soundfile as sf

ROOT = H.ROOT
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from plots.comb_page import (  # noqa: E402  (needs the sys.path line above)
    DECIM,
    DEFAULT_CACHE,
    Carrier,
    PageOptions,
    Slice,
    build_payload,
    split_payload,
    write_index,
    write_page,
)

# ─── Source resolution ────────────────────────────────────────────────────────


@dataclass
class Recording:
    """Inventory row — everything ``--list`` prints, no audio loaded."""

    rid: str
    dataset: str
    duration: float
    sr: int
    n_mics: int
    n_rotors: int
    channels: list[str]
    note: str = ""
    paths: dict = field(default_factory=dict)


def _michaels_entries() -> list[tuple[str, Path, Path, float, float]]:
    from data_processing.sources import michaels as MI

    root = MI.resolve_raw_root(ROOT / "data")
    return [
        (Path(csv_rel).stem, root / wav_rel, root / csv_rel, off, dil)
        for wav_rel, csv_rel, off, dil in MI.MICHAELS_FILES
    ]


def _michaels_meta(csv: Path, offset: float, info) -> tuple[float, int]:
    """``(aligned duration, rotor count)`` without loading the audio.

    The duration is the length AFTER ``load_raw_aligned`` crops the wav to the
    telemetry span — the time base ``--t0`` is measured in, not the raw wav
    length.
    """
    import pandas as pd

    wav_dur = info.frames / info.samplerate
    csv_head = pd.read_csv(csv, nrows=1)
    n_rot = len([c for c in csv_head.columns if "Motor" in c][:4])
    col = "Clock:offsetTime"
    t = pd.read_csv(csv, usecols=(col,))[col].to_numpy(float)  # pyright: ignore[reportCallIssue,reportArgumentType]
    t = t[(t >= offset) & (t <= wav_dur + offset)]
    if len(t) < 2:
        return wav_dur, n_rot
    return float(min(t[-1], wav_dur + offset) - max(t[0], offset)), n_rot


def inventory() -> list[Recording]:
    """Every recording this tool can build a page for, both datasets."""
    import scipy.io

    recs: list[Recording] = []
    for d in sorted(H.DREGON.glob("DREGON_*")):
        rid = d.name[len("DREGON_") :]
        wav = d / f"DREGON_{rid}.wav"
        if not d.is_dir() or not wav.exists():
            continue
        chans = H.available_channels(rid)
        if not chans:
            continue
        info = sf.info(str(wav))
        mat = scipy.io.loadmat(str(H._motors_path(rid)))["motor"]
        n_rot = int(mat["measured" if H.MEASURED in chans else "command"][0, 0].shape[1])
        recs.append(
            Recording(
                rid=rid,
                dataset="DREGON",
                duration=info.frames / info.samplerate,
                sr=info.samplerate,
                n_mics=info.channels,
                n_rotors=n_rot,
                channels=chans,
                note="tachometer present" if H.MEASURED in chans else "COMMAND ONLY",
            )
        )
    try:
        for rid, wav, csv, off, dil in _michaels_entries():
            if not wav.exists():
                continue
            info = sf.info(str(wav))
            dur, n_rot = _michaels_meta(csv, off, info)
            recs.append(
                Recording(
                    rid=rid,
                    dataset="michaels",
                    duration=dur,
                    sr=info.samplerate,
                    n_mics=info.channels,
                    n_rotors=n_rot,
                    channels=["rps"],
                    note="calibrated rev/s (WP14); t0 is on the telemetry-cropped audio",
                    paths={"wav": wav, "csv": csv, "offset": off, "dilation": dil},
                )
            )
    except Exception as exc:  # pragma: no cover - inventory must not hard-fail
        print(f"[warn] michaels inventory unavailable: {exc}", file=sys.stderr)
    return recs


def _find(rid: str, recs: list[Recording] | None = None) -> Recording:
    recs = recs if recs is not None else inventory()
    want = rid[len("DREGON_") :] if rid.startswith("DREGON_") else rid
    for r in recs:
        if r.rid == want or r.rid.upper() == want.upper():
            return r
    raise SystemExit(f"unknown recording {rid!r}\navailable: {', '.join(r.rid for r in recs)}")


def load_slice(rid: str, t0: float, dur: float, channel: str = "auto") -> tuple[Slice, Carrier]:
    """``(slice, telemetry carrier)`` for ``dur`` seconds from ``t0``."""
    rec = _find(rid)
    if rec.dataset == "DREGON":
        chan = rec.channels[0] if channel == "auto" else channel
        if chan not in rec.channels:
            raise SystemExit(f"{rec.rid}: no {chan}; has {rec.channels}")
        if t0 + dur > rec.duration + 1e-6:
            raise SystemExit(
                f"{rec.rid}: [{t0}, {t0 + dur}] s exceeds the audio ({rec.duration:.1f} s)"
            )
        audio, sr, g, _rates = H.load_raw(rec.rid, t0, dur, channel=chan)
        note = (
            "measured tachometer"
            if chan == H.MEASURED
            else "COMMANDED value, not a measurement (leading logging freeze removed)"
        )
        sl = Slice(rec.rid, "DREGON", t0, dur, sr, audio, note)
        return sl, Carrier(id="t", label=chan, g=g, note=note)

    from data_processing.sources import michaels as MI

    p = rec.paths
    # `ts` comes back on the same clock as the CROPPED audio (t = 0 at the first
    # sample the loader keeps), which is the convention every other caller of
    # load_raw_aligned in this repo uses, so `--t0` means the same thing here.
    wav, ts, ms, sr = MI.load_raw_aligned(
        p["wav"], p["csv"], time_offset=p["offset"], time_dilation=p["dilation"], sr=None
    )
    a0, n = int(round(t0 * sr)), int(round(dur * sr))
    if a0 + n > wav.shape[1]:
        raise SystemExit(
            f"{rec.rid}: [{t0}, {t0 + dur}] s exceeds the aligned audio ({wav.shape[1] / sr:.1f} s)"
        )
    audio = np.ascontiguousarray(wav[:, a0 : a0 + n]).astype(np.float64)
    del wav
    t = (a0 + np.arange(n)) / sr
    g = np.stack([np.interp(t, ts, ms[r]) for r in range(ms.shape[0])])
    note = (
        f"DatCon Motor:Speed, aligned (offset {p['offset']:.4f} s, dilation "
        f"{p['dilation']:.7f}) and rev/s-calibrated x{MI.rps_scale_for(p['csv']):.5f}"
    )
    sl = Slice(rec.rid, "michaels", t0, dur, sr, audio, note)
    return sl, Carrier(id="t", label="rps", g=g, note=note)


# ─── Refined labels ───────────────────────────────────────────────────────────


@dataclass
class Refined:
    """An acoustically realigned rotor trajectory, on the slice's own clock.

    Produced by a pi_kalman refinement initialised from telemetry (see
    ``refine_labels.py``): arrays ``ft`` (s), ``r_init`` (R, T) and ``r_ref``
    (R, T).  ``ft`` is read as slice-relative unless the npz carries a ``t0``.
    """

    path: Path
    ft: np.ndarray
    r_ref: np.ndarray
    check_rms: list[float]  # |r_init - telemetry| per rotor, rev/s
    note: str


def load_refined(path: Path | None, sl: Slice, tel: Carrier) -> Refined | None:
    """Read a refined-label npz, or ``None`` when it is absent or unusable.

    Never fatal: the page must build before the refinement has finished, and
    the mismatch check is reported in the page instead of being hidden.
    """
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        print(f"[refined] {path} not present — page built without it", file=sys.stderr)
        return None
    try:
        with np.load(path) as f:
            ft = np.asarray(f["ft"], dtype=np.float64)
            r_ref = np.asarray(f["r_ref"], dtype=np.float64)
            r_init = np.asarray(f["r_init"], dtype=np.float64) if "r_init" in f else None
            npz_t0 = float(f["t0"]) if "t0" in f else None
    except Exception as exc:
        print(f"[refined] unreadable ({exc}) — page built without it", file=sys.stderr)
        return None
    if r_ref.ndim != 2 or r_ref.shape[0] != tel.n_rotors:
        print(
            f"[refined] {r_ref.shape} does not match {tel.n_rotors} rotors — ignored",
            file=sys.stderr,
        )
        return None
    if npz_t0 is not None:
        ft = ft + (npz_t0 - sl.t0)  # ft was written against the npz's own window
    if ft.max() < sl.dur * 0.25:
        print(f"[refined] covers only {ft.max():.1f} s of a {sl.dur:.1f} s window", file=sys.stderr)
    t_aud = sl.t_grid
    check = []
    if r_init is not None and r_init.shape == r_ref.shape:
        for r in range(tel.n_rotors):
            tel_r = np.interp(ft, t_aud, tel.g[r])
            check.append(round(float(np.sqrt(np.mean((r_init[r] - tel_r) ** 2))), 4))
    d = [
        round(float(np.mean(r_ref[r] - np.interp(ft, t_aud, tel.g[r]))), 4)
        for r in range(len(r_ref))
    ]
    note = f"{path.name}: refined-minus-telemetry mean {d} rev/s"
    if check:
        note += f"; init-vs-telemetry RMS {check} rev/s"
    # A refinement is only ever valid for the window it was run on.  Its own
    # initial condition IS the telemetry of that window, so a large disagreement
    # with this slice's telemetry means the npz belongs to a different window or
    # a different recording — draw nothing rather than a plausible-looking lie.
    ref_gap = max(check) if check else max(abs(x) for x in d)
    if ref_gap > 2.0:
        print(
            f"[refined] REJECTED — {note}: this npz does not belong to this window",
            file=sys.stderr,
        )
        return None
    print(f"[refined] {note}", flush=True)
    return Refined(path, ft, r_ref, check, note)


def refined_on(ref: Refined, t: np.ndarray) -> np.ndarray:
    """Refined trajectories resampled onto an arbitrary time grid (R, len(t))."""
    return np.stack([np.interp(t, ref.ft, ref.r_ref[r]) for r in range(ref.r_ref.shape[0])])


# ─── CLI ──────────────────────────────────────────────────────────────────────


def slug(rid: str, t0: float, dur: float) -> str:
    return f"{rid}__t{t0:g}__d{dur:g}".replace(".", "p").replace("-", "_")


def page_name(sl: Slice, ch: str) -> str:
    return f"comb_{slug(sl.rid, sl.t0, sl.dur)}__{ch}.html"


def parse_add(spec: str) -> tuple[str, float, float]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise SystemExit(f"--add wants RECORDING:T0:DUR, got {spec!r}")
    return parts[0], float(parts[1]), float(parts[2])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--list", action="store_true", help="print the recording inventory and exit")
    ap.add_argument("--recording", help="DREGON id (DREGON_ prefix optional) or FLY124 / FLY125")
    ap.add_argument("--t0", type=float, default=0.0, help="window start, seconds")
    ap.add_argument("--dur", type=float, default=16.0, help="window length, seconds")
    ap.add_argument(
        "--add",
        action="append",
        default=[],
        metavar="REC:T0:DUR",
        help="extra slice; one page per slice per channel, plus an index",
    )
    ap.add_argument("--out", type=Path, help="output .html (single page only)")
    ap.add_argument("--out-dir", type=Path, default=Path("."), help="output dir (several pages)")
    ap.add_argument(
        "--rps-channel",
        default="auto",
        choices=["auto", H.MEASURED, H.COMMAND],
        help="DREGON only; auto prefers the measured tachometer",
    )
    ap.add_argument(
        "--channels",
        default="avg",
        help="microphone channels: avg | all | a list like 0,3,avg (one page each)",
    )
    ap.add_argument("--refined", type=Path, help="npz with ft / r_init / r_ref refined labels")
    ap.add_argument("--ks", default="1-100", help="harmonics, e.g. 1-100 or 1-24,55-95")
    ap.add_argument("--k-max", type=int, default=100, help="hard ceiling on k")
    ap.add_argument("--segs", default="0.1,0.5,2.0", help="strip segment lengths, seconds")
    ap.add_argument("--strip-rows", type=int, default=80, help="offset-axis rows per strip")
    ap.add_argument("--strip-cols", type=int, default=110, help="max time columns per strip")
    ap.add_argument(
        "--strip-floor",
        type=float,
        default=60.0,
        help="strip black point, percentile of in-band dB",
    )
    ap.add_argument("--strip-top", type=float, default=99.5, help="strip white point, percentile")
    ap.add_argument("--ylim", type=float, default=6.0, help="strip half-range, rev/s")
    ap.add_argument("--decim", type=int, default=DECIM, help="envelope decimation (32 = 1378 Hz)")
    ap.add_argument("--nfft", type=int, default=2048)
    ap.add_argument("--spec-cols", type=int, default=600)
    ap.add_argument("--fmax", type=float, default=10000.0)
    ap.add_argument("--max-mb", type=float, default=9.0, help="hard page-size budget")
    ap.add_argument("--cache", action="store_true", help="write new envelope-cache entries")
    ap.add_argument("--cache-dir", default=None, help=f"envelope cache (default {DEFAULT_CACHE})")
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    args = ap.parse_args(argv)
    segs = tuple(float(s) for s in str(args.segs).split(",") if s.strip())

    if args.list:
        recs = inventory()
        w = max(len(r.rid) for r in recs)
        print(f"{'recording'.ljust(w)}  dataset   dur(s)   sr     mic rot  rps channel(s)")
        for r in recs:
            print(
                f"{r.rid.ljust(w)}  {r.dataset:9s} {r.duration:6.1f}  {r.sr:6d} {r.n_mics:3d} "
                f"{r.n_rotors:3d}  {', '.join(r.channels)}  [{r.note}]"
            )
        return 0

    if not args.recording:
        ap.error("--recording is required (or use --list)")

    opts = PageOptions(
        channels=args.channels,
        ks=args.ks,
        k_max=args.k_max,
        segs=segs,
        strip_rows=args.strip_rows,
        strip_cols=args.strip_cols,
        strip_floor=args.strip_floor,
        strip_top=args.strip_top,
        ylim=args.ylim,
        decim=args.decim,
        nfft=args.nfft,
        spec_cols=args.spec_cols,
        fmax=args.fmax,
        max_mb=args.max_mb,
        cache=args.cache,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        jobs=args.jobs,
    )

    specs = [(args.recording, args.t0, args.dur)] + [parse_add(s) for s in args.add]
    pages: list[tuple[Path, dict, int]] = []
    for rid, t0, dur in specs:
        print(f"\n=== {rid}  t0={t0}  dur={dur}", flush=True)
        sl, tel = load_slice(rid, t0, dur, args.rps_channel)
        print(
            f"[load] {sl.dataset} {sl.rid}: {sl.n_mics} mic, {tel.n_rotors} rotor, {sl.sr} Hz, "
            f"{tel.label}, rates {np.round(tel.rates, 2)}",
            flush=True,
        )
        carriers = [Carrier(id="t", label="telemetry", g=tel.g, note=tel.note)]
        refined = load_refined(args.refined, sl, tel)
        if refined is not None:
            carriers.append(
                Carrier(
                    id="r",
                    label="refined",
                    g=refined_on(refined, sl.t_grid),
                    note=refined.note,
                )
            )
        full = build_payload(
            sl,
            carriers,
            opts,
            meta_extra={
                "rps_channel": tel.label,
                "rps_note": tel.note,
                "refined": refined.note if refined is not None else "",
            },
        )
        per_channel = split_payload(full, partial(page_name, sl))
        for ch, payload in per_channel.items():
            single = len(specs) == 1 and len(payload["chans"]) == 1
            out = args.out if (single and args.out) else Path(args.out_dir) / page_name(sl, ch)
            size = write_page(payload, out)
            print(f"[page] {out}  {size / 1e6:.2f} MB", flush=True)
            if size > args.max_mb * 1e6:
                print(f"[warn] {out.name} exceeds the {args.max_mb} MB budget", file=sys.stderr)
            pages.append((out, payload, size))
        del sl

    if len(pages) > 1:
        idx = Path(args.out_dir) / "index.html"
        write_index(pages, idx)
        print(f"[index] {idx}  ({len(pages)} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
