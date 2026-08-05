#!/usr/bin/env python3
"""Build the interactive comb explorer page for ANY recording and time slice.

The page is one self-contained HTML fragment (no external hosts, no fetch, no
CDN) that shows:

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
control), so a mic-averaged picture is not the whole evidence.

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
import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import hk_core as H
import numpy as np
import soundfile as sf
from PIL import Image
from scipy.signal import resample_poly

ROOT = H.ROOT
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

CODE_VERSION = "comb_explorer.py v1 (2026-08-05)"

#: Strip envelope decimation.  44100 / 32 = 1378 Hz keeps +-689 Hz, i.e.
#: +-689/k rev/s of shaft-rate offset — +-7.25 rev/s even at k = 95, so the
#: whole +-6 rev/s bandwidth slider is real data at every harmonic in the set.
#: A decim-100 envelope (441 Hz, +-2.2 rev/s at k=100) draws a black band as
#: soon as the slider is widened, which is why the old ``hk_cache.py`` cache is
#: NOT read here: its entries are decim 100 and would silently truncate.
DECIM = 32

DEFAULT_CACHE = Path(os.environ.get("COMB_CACHE_DIR", ROOT / ".cache/comb_explorer"))


# ─── Source resolution ────────────────────────────────────────────────────────


@dataclass
class Slice:
    """One loaded (recording, t0, dur) window, dataset-agnostic."""

    rid: str
    dataset: str
    t0: float
    dur: float
    sr: int
    audio: np.ndarray  # (C, N)
    g: np.ndarray  # (R, N) rev/s on the audio grid
    rates: np.ndarray  # (R,) mean rev/s over the slice
    rps_channel: str
    rps_note: str = ""

    @property
    def n_mics(self) -> int:
        return int(self.audio.shape[0])

    @property
    def n_rotors(self) -> int:
        return int(self.g.shape[0])


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


def load_slice(rid: str, t0: float, dur: float, channel: str = "auto") -> Slice:
    """Load ``dur`` seconds from ``t0`` of any supported recording."""
    rec = _find(rid)
    if rec.dataset == "DREGON":
        chan = rec.channels[0] if channel == "auto" else channel
        if chan not in rec.channels:
            raise SystemExit(f"{rec.rid}: no {chan}; has {rec.channels}")
        if t0 + dur > rec.duration + 1e-6:
            raise SystemExit(
                f"{rec.rid}: [{t0}, {t0 + dur}] s exceeds the audio ({rec.duration:.1f} s)"
            )
        audio, sr, g, rates = H.load_raw(rec.rid, t0, dur, channel=chan)
        note = (
            "measured tachometer"
            if chan == H.MEASURED
            else "COMMANDED value, not a measurement (leading logging freeze removed)"
        )
        return Slice(rec.rid, "DREGON", t0, dur, sr, audio, g, rates, chan, note)

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
    return Slice(rec.rid, "michaels", t0, dur, sr, audio, g, g.mean(axis=1), "rps", note)


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


def load_refined(path: Path | None, sl: Slice) -> Refined | None:
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
    if r_ref.ndim != 2 or r_ref.shape[0] != sl.n_rotors:
        print(
            f"[refined] {r_ref.shape} does not match {sl.n_rotors} rotors — ignored",
            file=sys.stderr,
        )
        return None
    if npz_t0 is not None:
        ft = ft + (npz_t0 - sl.t0)  # ft was written against the npz's own window
    if ft.max() < sl.dur * 0.25:
        print(f"[refined] covers only {ft.max():.1f} s of a {sl.dur:.1f} s window", file=sys.stderr)
    t_aud = np.arange(sl.audio.shape[1]) / sl.sr
    check = []
    if r_init is not None and r_init.shape == r_ref.shape:
        for r in range(sl.n_rotors):
            tel = np.interp(ft, t_aud, sl.g[r])
            check.append(round(float(np.sqrt(np.mean((r_init[r] - tel) ** 2))), 4))
    d = [
        round(float(np.mean(r_ref[r] - np.interp(ft, t_aud, sl.g[r]))), 4)
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


# ─── Spectrogram ──────────────────────────────────────────────────────────────


def spectrograms(sl: Slice, mics: list[int], nfft: int, cols: int, fmax: float):
    """``(stft_u8, sst_u8, f, t)`` for one microphone channel selection.

    Both images are ``(NF, NT)`` uint8, dB quantised over the STFT's own
    1st..99.8th percentile so the two transforms share a colour scale.  ``sst``
    is the synchrosqueezed (frequency-reassigned) copy: a second FFT with the
    time-derivative of the window gives ``omega = f - Im(S_dw / S) / 2pi``, and
    each bin's power is moved to the row nearest ``omega``.  Reassignment is
    done per microphone and the reassigned powers are summed, never the other
    way round — summing first would smear the ridge it exists to sharpen.
    """
    n = sl.audio.shape[1]
    hop = max(int(round((n - nfft) / cols)), 1)
    starts = np.arange(0, n - nfft + 1, hop)
    win = np.hanning(nfft)
    dwin = np.gradient(win) * sl.sr  # d/dt of the window, per second
    df = sl.sr / nfft
    nkeep = min(int(fmax / df) + 1, nfft // 2 + 1)
    nt = len(starts)
    acc = np.zeros((nt, nkeep))
    sst = np.zeros(nkeep * nt)
    freqs = np.arange(nkeep) * df
    tcol = np.repeat(np.arange(nt)[:, None], nkeep, axis=1)
    for c in mics:
        fr = np.lib.stride_tricks.sliding_window_view(sl.audio[c], nfft)[::hop]
        S = np.fft.rfft(fr * win, axis=-1)[:, :nkeep]
        P = np.abs(S) ** 2
        acc += P
        Sd = np.fft.rfft(fr * dwin, axis=-1)[:, :nkeep]
        safe = np.abs(S) > 0
        om = freqs[None, :] - np.imag(np.divide(Sd, S, out=np.zeros_like(S), where=safe)) / (
            2 * np.pi
        )
        row = np.rint(om / df)
        ok = safe & np.isfinite(row) & (row >= 0) & (row <= nkeep - 1)
        idx = row[ok].astype(np.int64) * nt + tcol[ok]
        sst += np.bincount(idx, weights=P[ok], minlength=nkeep * nt)
    acc /= len(mics)
    sst = sst.reshape(nkeep, nt) / len(mics)
    db = 10.0 * np.log10(acc.T + 1e-20)
    lo, hi = float(np.percentile(db, 1.0)), float(np.percentile(db, 99.8))
    q = lambda x: np.clip((x - lo) / max(hi - lo, 1e-9) * 255.0, 0, 255).astype(np.uint8)  # noqa: E731
    return q(db), q(10.0 * np.log10(sst + 1e-20)), freqs, (starts + nfft / 2.0) / sl.sr


# ─── Demodulated envelopes (the expensive, cacheable product) ─────────────────


def _carrier_tag(g_rot: np.ndarray) -> str:
    """Short hash of the carrier trajectory — the cache key must depend on the
    trajectory itself, or a refined run would silently reuse telemetry."""
    return hashlib.sha1(np.round(g_rot, 6).tobytes()).hexdigest()[:8]


def _env_path(cache_dir: Path, sl: Slice, rotor: int, k_max: int, decim: int, tag: str) -> Path:
    return cache_dir / (
        f"{sl.rid}__t{round(sl.t0 * 1000):09d}ms__d{round(sl.dur * 1000):07d}ms"
        f"__k{k_max}__dec{decim}__r{rotor}__{tag}.npz"
    )


def _build_envelope(args):
    """``(rotor, z (K, C, n_env) complex64, fs_env)`` — audio demodulated by
    ``exp(-i k phi)`` and decimated, for every k and every microphone."""
    rotor, audio, sr, g_rot, k_max, decim = args
    phi = H.phase(g_rot, sr)
    n_env = len(resample_poly(audio[0], 1, decim))
    z = np.zeros((k_max, audio.shape[0], n_env), np.complex64)
    for a in range(k_max):
        carrier = np.exp(-1j * float(a + 1) * phi)
        for c in range(audio.shape[0]):
            z[a, c] = resample_poly(audio[c] * carrier, 1, decim).astype(np.complex64)
    return rotor, z, sr / decim


def envelope_batch(
    sl: Slice,
    g_all: np.ndarray,
    k_max: int,
    decim: int,
    cache_dir: Path | None,
    write_cache: bool,
    jobs: int,
) -> tuple[dict[int, np.ndarray], float, str]:
    """``({rotor: z}, fs_env, provenance)`` for one carrier (telemetry or refined).

    A cache entry is reused only when the recording, window, k ceiling, DECIM
    *and* the carrier trajectory hash all match; anything else is recomputed.
    That is deliberate — the historical decim-100 cache would fit every other
    field and quietly halve the usable bandwidth.
    """
    fs_env = sl.sr / decim
    z_by: dict[int, np.ndarray] = {}
    reused: list[int] = []
    tags = [_carrier_tag(g_all[r]) for r in range(sl.n_rotors)]
    if cache_dir:
        for r in range(sl.n_rotors):
            p = _env_path(Path(cache_dir), sl, r, k_max, decim, tags[r])
            if not p.exists():
                continue
            try:
                with np.load(p) as f:
                    if int(f["decim"]) != decim or int(f["k_max"]) < k_max:
                        continue
                    z_by[r] = f["z"][:k_max].copy()
                    fs_env = float(f["fs_env"])
            except Exception:
                continue
            reused.append(r)

    todo = [r for r in range(sl.n_rotors) if r not in z_by]
    if todo:
        t_start = time.time()
        args = [(r, sl.audio, sl.sr, g_all[r], k_max, decim) for r in todo]
        if jobs > 1 and len(todo) > 1:
            with ProcessPoolExecutor(max_workers=min(jobs, len(todo))) as pool:
                results = list(pool.map(_build_envelope, args))
        else:
            results = [_build_envelope(a) for a in args]
        for r, z, fe in results:
            z_by[r], fs_env = z, fe
        del results
        print(f"[env] demodulated {todo} k=1..{k_max} in {time.time() - t_start:.0f} s", flush=True)
        if write_cache and cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            for r in todo:
                out = _env_path(Path(cache_dir), sl, r, k_max, decim, tags[r])
                np.savez(
                    out,
                    z=z_by[r],
                    g=g_all[r][:: max(int(sl.sr / fs_env), 1)],
                    fs_env=np.float64(fs_env),
                    decim=np.int32(decim),
                    k_max=np.int32(k_max),
                    sr_audio=np.int32(sl.sr),
                    t_start=np.float64(sl.t0),
                    dur=np.float64(sl.dur),
                    rotor=np.int32(r),
                    rps_channel=sl.rps_channel,
                )
                print(f"[env] cached {out.name} ({out.stat().st_size / 1e6:.0f} MB)", flush=True)

    if reused and todo:
        src = f"cache hit rotors {sorted(reused)}, rotors {todo} demodulated"
    elif reused:
        src = f"cache hit rotors {sorted(reused)}"
    else:
        src = "demodulated on the fly"
    return z_by, fs_env, src


# ─── Strips ───────────────────────────────────────────────────────────────────


def strip_stack(
    z: np.ndarray,
    mics: list[int],
    ks: np.ndarray,
    fs_env: float,
    seg_s: float,
    ylim: float,
    rows: int,
    cols: int,
    floor_pct: float = 60.0,
    top_pct: float = 99.5,
):
    """``(u8 (len(ks), rows, ncol), t_first, t_last)`` for one rotor and segment.

    The row axis is a FIXED shaft-rate offset grid ``[-ylim, +ylim]`` rev/s, so
    every harmonic shares an axis; rows outside the usable band
    ``+-fs_env/2/k`` stay 0 (black).  Each harmonic is contrast-normalised on
    its own values, because the absolute level falls by tens of dB across k:
    ``floor_pct`` / ``top_pct`` are the black and white points, as percentiles
    of that harmonic's in-band dB, so the noise floor drops out and the tooth
    is what is left.

    Frames are subsampled to at most ``cols`` BEFORE the FFT — the time axis of
    a strip is only ever a few hundred pixels wide, so computing more is waste.
    """
    n_env = z.shape[-1]
    n_seg = max(int(round(seg_s * fs_env)) & ~1, 8)
    if n_seg > n_env:
        return None, 0.0, 0.0
    hop = max(n_seg // 4, 1)
    starts = np.arange(0, n_env - n_seg + 1, hop)
    if len(starts) < 2:
        return None, 0.0, 0.0
    if len(starts) > cols:
        starts = starts[np.linspace(0, len(starts) - 1, cols).round().astype(int)]
    n_t = len(starts)
    win = np.hanning(n_seg)
    freqs = np.fft.fftshift(np.fft.fftfreq(n_seg, d=1.0 / fs_env))
    band_hz = fs_env / 2.0
    grid = np.linspace(-ylim, ylim, rows)
    out = np.zeros((len(ks), rows, n_t), np.uint8)

    for a, k in enumerate(ks):
        idx = int(k) - 1
        inb = np.abs(grid) <= band_hz / k
        if not inb.any():
            continue
        rev = freqs / k
        sel = np.abs(rev) <= ylim
        if sel.sum() < 3:
            continue
        prof = np.zeros((rows, n_t))
        for c in mics:
            fr = np.stack([z[idx, c, s : s + n_seg] for s in starts]) * win
            P = np.abs(np.fft.fftshift(np.fft.fft(fr, axis=-1), axes=-1)) ** 2
            for j in range(n_t):
                prof[:, j] += np.interp(grid, rev[sel], P[j][sel])
        db = 10.0 * np.log10(prof / len(mics) + 1e-30)
        v = db[inb]
        lo = float(np.percentile(v, floor_pct))  # black point = the noise floor
        hi = float(np.percentile(v, top_pct))  # white point = the tooth
        q = np.clip((db - lo) / max(hi - lo, 1e-9) * 255.0, 0, 255).astype(np.uint8)
        q[~inb] = 0
        out[a] = q
    t_first = float((starts[0] + n_seg / 2) / fs_env)
    t_last = float((starts[-1] + n_seg / 2) / fs_env)
    return out, t_first, t_last


def parse_ks(spec: str, k_max: int) -> np.ndarray:
    """``"1-24,55-95"`` / ``"2,4,71"`` / ``"1-100"`` -> a sorted k array."""
    out: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return np.array(sorted(k for k in out if 1 <= k <= k_max), dtype=int)


def parse_channels(spec: str, n_mics: int) -> list[str]:
    """``"avg"`` / ``"all"`` / ``"0,3,avg"`` -> channel ids, ``avg`` or ``micNN``."""
    spec = str(spec).strip().lower()
    if spec == "all":
        return ["avg"] + [f"mic{c:02d}" for c in range(n_mics)]
    out: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if part in ("avg", "mean", "average"):
            out.append("avg")
            continue
        c = int(part)
        if not 0 <= c < n_mics:
            raise SystemExit(f"--channels: mic {c} out of range (0..{n_mics - 1})")
        out.append(f"mic{c:02d}")
    return out or ["avg"]


def channel_mics(ch: str, n_mics: int) -> list[int]:
    return list(range(n_mics)) if ch == "avg" else [int(ch[3:])]


def channel_label(ch: str, n_mics: int) -> str:
    return f"incoherent average of {n_mics} mics" if ch == "avg" else f"mic {int(ch[3:])} alone"


# ─── Encoding ─────────────────────────────────────────────────────────────────


def png_b64(a: np.ndarray) -> str:
    """Grayscale PNG of a 2-D uint8 array, base64.  About 2x smaller than the
    raw base64 the hand-built page used, which is what buys the second carrier
    and the second transform."""
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(a)).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        return "unknown"


# ─── Payload ──────────────────────────────────────────────────────────────────


def build_payloads(sl: Slice, args, refined: Refined | None) -> dict[str, dict]:
    """``{channel_id: payload}`` — one page's data per microphone channel.

    The envelope of a rotor is built once and every channel and segment length
    is cut from it, so N channels cost N times the (cheap) strip FFTs, not N
    times the (expensive) demodulation.
    """
    chans = parse_channels(args.channels, sl.n_mics)
    k_ceil = min(args.k_max, int(np.floor((sl.sr / 2) / max(sl.rates.max(), 1e-6))))
    ks = parse_ks(args.ks, k_ceil)
    if not len(ks):
        raise SystemExit(f"--ks {args.ks!r} selects nothing at k <= {k_ceil}")
    k_max = int(ks.max())
    contiguous = bool(len(ks) == ks[-1] - ks[0] + 1)

    specs: dict[str, dict] = {}
    for ch in chans:
        u8, sst, f, t = spectrograms(
            sl, channel_mics(ch, sl.n_mics), args.nfft, args.spec_cols, args.fmax
        )
        specs[ch] = {"stft": u8, "sst": sst}
        print(f"[spec] {ch}: {u8.shape[0]} x {u8.shape[1]} bins, {f[-1]:.0f} Hz", flush=True)
    nf, nt = specs[chans[0]]["stft"].shape

    carriers: list[tuple[str, str, np.ndarray]] = [("t", "telemetry", sl.g)]
    if refined is not None:
        carriers.append(("r", "refined", refined_on(refined, np.arange(sl.audio.shape[1]) / sl.sr)))

    stacks: dict[str, dict[str, np.ndarray]] = {ch: {} for ch in chans}
    spans: dict[str, tuple[float, float]] = {}
    cache_src = ""
    for tag, cname, g_all in carriers:
        z_by, fs_env, cache_src_c = envelope_batch(
            sl,
            g_all,
            k_max,
            args.decim,
            Path(args.cache_dir) if args.cache_dir else DEFAULT_CACHE,
            args.cache,
            args.jobs,
        )
        cache_src = cache_src_c if not cache_src else f"{cache_src}; {cname}: {cache_src_c}"
        for r in range(sl.n_rotors):
            for si, seg in enumerate(args.segs):
                for ch in chans:
                    t_s = time.time()
                    st, ta, tb = strip_stack(
                        z_by[r],
                        channel_mics(ch, sl.n_mics),
                        ks,
                        fs_env,
                        seg,
                        args.ylim,
                        args.strip_rows,
                        args.strip_cols,
                        args.strip_floor,
                        args.strip_top,
                    )
                    if st is None:
                        continue
                    key = f"{r}|{tag}|{si}"
                    stacks[ch][key] = st
                    spans[key] = (ta, tb)
                    print(
                        f"[strip] {ch} r{r} {cname} {seg * 1000:.0f} ms {st.shape} "
                        f"in {time.time() - t_s:.1f} s",
                        flush=True,
                    )
        del z_by
    if not spans:
        raise SystemExit("no strips could be built (window shorter than every segment length)")

    live_segs = sorted({int(k.split("|")[2]) for k in spans})
    remap = {old: new for new, old in enumerate(live_segs)}
    for ch in chans:
        stacks[ch] = {
            f"{k.split('|')[0]}|{k.split('|')[1]}|{remap[int(k.split('|')[2])]}": v
            for k, v in stacks[ch].items()
        }
    spans = {
        f"{k.split('|')[0]}|{k.split('|')[1]}|{remap[int(k.split('|')[2])]}": v
        for k, v in spans.items()
    }
    segs = [args.segs[i] for i in live_segs]

    # Encode, then shrink the TIME axis until each page fits the budget.  Rotors,
    # harmonics and carriers are never cut: columns are.
    budget = args.max_mb * 1e6
    cap = args.strip_cols
    blobs: dict[str, dict[str, dict]] = {}
    for _ in range(8):
        blobs = {}
        worst = 0
        for ch in chans:
            blobs[ch] = {}
            for key, v in stacks[ch].items():
                th = (
                    v
                    if v.shape[-1] <= cap
                    else v[..., np.linspace(0, v.shape[-1] - 1, cap).round().astype(int)]
                )
                blobs[ch][key] = {
                    "data": png_b64(th.reshape(th.shape[0] * th.shape[1], th.shape[2])),
                    "nk": int(th.shape[0]),
                    "nr": int(th.shape[1]),
                    "nc": int(th.shape[2]),
                    "t0": round(spans[key][0], 4),
                    "t1": round(spans[key][1], 4),
                    "mean": round(float(th.mean()), 4),
                }
            worst = max(worst, sum(len(b["data"]) for b in blobs[ch].values()))
        print(f"[budget] worst-channel strips {worst / 1e6:.2f} MB at <= {cap} columns", flush=True)
        if worst < budget * 0.90 or cap <= 8:
            break
        cap = max(int(cap * 0.75), 8)

    t_full = np.arange(sl.audio.shape[1]) / sl.sr
    f = np.arange(nf) * (sl.sr / args.nfft)
    hop = max(int(round((sl.audio.shape[1] - args.nfft) / args.spec_cols)), 1)
    t = (np.arange(nt) * hop + args.nfft / 2.0) / sl.sr
    g_cols = np.stack([np.interp(t, t_full, sl.g[r]) for r in range(sl.n_rotors)])
    gref_cols = refined_on(refined, t) if refined is not None else None

    pages: dict[str, dict] = {}
    nav = [{"id": c, "label": channel_label(c, sl.n_mics), "file": page_name(sl, c)} for c in chans]
    cols_note = ", ".join(
        f"{int(s * 1000)} ms {blobs[chans[0]][f'0|t|{i}']['nc']} col" for i, s in enumerate(segs)
    )
    for ch in chans:
        pages[ch] = {
            "meta": {
                "recording": sl.rid,
                "dataset": sl.dataset,
                "t0": sl.t0,
                "dur": sl.dur,
                "sr": sl.sr,
                "n_mics": sl.n_mics,
                "n_rotors": sl.n_rotors,
                "rps_channel": sl.rps_channel,
                "rps_note": sl.rps_note,
                "rates": [round(float(x), 3) for x in sl.rates],
                "channel": ch,
                "channel_label": channel_label(ch, sl.n_mics),
                "channels": nav,
                "code_version": CODE_VERSION,
                "git": _git_sha(),
                "built": time.strftime("%Y-%m-%d %H:%M"),
                "nfft": args.nfft,
                "env_rate_hz": round(sl.sr / args.decim, 2),
                "cache": cache_src,
                "k_note": (
                    f"k = {int(ks[0])}..{int(ks[-1])} contiguous"
                    if contiguous
                    else f"k = {', '.join(str(int(x)) for x in ks[:12])}... NOT contiguous"
                ),
                "refined": refined.note if refined is not None else "",
                "budget": (
                    f"{sl.n_rotors} rotors x {len(ks)} harmonics x {len(segs)} segment lengths "
                    f"({cols_note}) x {len(carriers)} carrier(s), {args.strip_rows} offset rows, "
                    f"black/white points p{args.strip_floor:g}/p{args.strip_top:g}, "
                    f"cap {args.max_mb:.0f} MB"
                ),
                "channel_trade": (
                    f"one page per microphone channel: {sl.n_mics} mics x {len(ks)} harmonics x "
                    f"{sl.n_rotors} rotors x {len(segs)} segment lengths does not fit "
                    f"{args.max_mb:.0f} MB, and dropping harmonics would leave holes in k, so the "
                    "channel is chosen by the selector above (each choice is its own page)"
                ),
            },
            "spec": {
                "stft": png_b64(specs[ch]["stft"]),
                "sst": png_b64(specs[ch]["sst"]),
                "nf": int(nf),
                "nt": int(nt),
                "f0": float(f[0]),
                "f1": float(f[-1]),
                "t0": float(t[0]),
                "t1": float(t[-1]),
                "mean": round(float(specs[ch]["stft"].mean()), 4),
                "mean_sst": round(float(specs[ch]["sst"].mean()), 4),
            },
            "traj": {
                "g": [[round(float(x), 4) for x in g_cols[r]] for r in range(sl.n_rotors)],
                "gref": (
                    [[round(float(x), 4) for x in gref_cols[r]] for r in range(sl.n_rotors)]
                    if gref_cols is not None
                    else None
                ),
                "t": [round(float(x), 4) for x in t],
                "rates": [round(float(x), 3) for x in sl.rates],
            },
            "strips": blobs[ch],
            "ks": [int(k) for k in ks],
            "segs": segs,
            "carriers": [{"id": tag, "label": name} for tag, name, _ in carriers],
            "ylim": args.ylim,
            "fmax": float(f[-1]),
        }
    return pages


# ─── Page ─────────────────────────────────────────────────────────────────────

HTML = r"""<title>Comb explorer — __TITLE__</title>
<style>
:root{--bg:#fbfaf7;--fg:#1f2430;--mut:#5c6472;--line:#ded8cd;--card:#fff;--code:#f1ede4;--acc:#b3452e}
@media (prefers-color-scheme:dark){:root{--bg:#14161c;--fg:#e8e6e0;--mut:#9aa1ad;--line:#2c3038;--card:#1c1f27;--code:#22252d;--acc:#e0704f}}
:root[data-theme="dark"]{--bg:#14161c;--fg:#e8e6e0;--mut:#9aa1ad;--line:#2c3038;--card:#1c1f27;--code:#22252d;--acc:#e0704f}
:root[data-theme="light"]{--bg:#fbfaf7;--fg:#1f2430;--mut:#5c6472;--line:#ded8cd;--card:#fff;--code:#f1ede4;--acc:#b3452e}
body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;margin:0;padding:14px}
h1{font-size:1.15rem;margin:0 0 .2rem}
p.sub{color:var(--mut);margin:0 0 .8rem;font-size:.86rem;max-width:80ch}
.panel{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:.6rem .8rem;margin-bottom:.7rem}
.prov{font-family:ui-monospace,monospace;font-size:.76rem;color:var(--mut);display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:.15rem .9rem}
.prov b{color:var(--fg);font-weight:600}
.row{display:flex;flex-wrap:wrap;gap:1.1rem;align-items:center}
label{font-size:.83rem;color:var(--mut)}
input[type=range]{width:190px;vertical-align:middle}
input[type=text]{background:var(--code);color:var(--fg);border:1px solid var(--line);border-radius:4px;padding:3px 6px;width:230px;font-family:ui-monospace,monospace}
select{background:var(--code);color:var(--fg);border:1px solid var(--line);border-radius:4px;padding:3px 6px;font-size:.82rem}
button{background:var(--code);color:var(--fg);border:1px solid var(--line);border-radius:4px;padding:3px 8px;cursor:pointer;font-size:.82rem}
button:hover{border-color:var(--acc)}
canvas{display:block;width:100%;border-radius:4px}
.mono{font-family:ui-monospace,monospace;font-size:.8rem}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:3px;vertical-align:-1px}
#strips{display:grid;gap:5px}
.stripwrap{position:relative}
.striplab{position:absolute;left:6px;top:3px;font:600 11px ui-monospace,monospace;color:#fff;text-shadow:0 0 4px #000;pointer-events:none}
.miss{padding:6px 8px;border:1px dashed var(--line);border-radius:4px;color:var(--mut);font:12px ui-monospace,monospace}
.warn{color:var(--acc);font-weight:600}
</style>
<h1>Comb explorer — <span class="mono">__TITLE__</span></h1>
<p class="sub">Spectrogram with per-rotor harmonic overlays. Solid = telemetry <span class="mono">k&middot;g(t)</span>; dashed = telemetry &times; the scale factor below; dotted = the refined trajectory, when the page carries one. Drag on the spectrogram to move the time marker. Strips at the bottom are the demodulated envelope of each selected harmonic, frequency axis rescaled to shaft-rate offset <span class="mono">(f&minus;kg)/k</span> in rev/s, so the carrier is the centre line. A pure SCALE error is one slider setting that lands every dashed line on every tooth at every time; a SHAPE error is not.</p>

<div class="panel"><div class="prov" id="prov"></div></div>

<div class="panel"><div class="row">
<span><label>microphone channel</label> <select id="chan"></select></span>
<span><label>transform (spectrogram + cut)</label> <select id="tf"><option value="stft">STFT</option><option value="sst">synchrosqueezed</option></select></span>
<span><label>scale factor <b id="sfv" class="mono">1.00000</b></label><br><input type="range" id="sf" min="0.985" max="1.015" step="0.00002" value="1"><button id="sfr">reset</button><button id="sfc">0.99458</button></span>
</div></div>

<div class="panel"><div class="row">
<span><label>rotors</label>
<span id="rotchk"></span></span>
</div></div>

<div class="panel"><div class="row">
<span><label>freq low <b id="flv" class="mono"></b></label><br><input type="range" id="fl" min="0" max="10000" step="10" value="0"></span>
<span><label>freq high <b id="fhv" class="mono"></b></label><br><input type="range" id="fh" min="0" max="10000" step="10" value="10000"></span>
<span id="fpre"></span>
<span><label>time <b id="tv" class="mono"></b></label></span>
</div></div>

<canvas id="spec" height="420"></canvas>
<canvas id="slice" height="150" style="margin-top:6px"></canvas>

<div class="panel"><div class="row">
<span><label>strips for rotor</label> <select id="srot"></select></span>
<span><label>carrier</label> <select id="car"></select></span>
<span><label>strip segment</label> <select id="seg"></select></span>
<span><label>strip bandwidth &plusmn;<b id="bwv" class="mono">1.50</b> rev/s</label><br><input type="range" id="bw" min="0.15" max="6" step="0.05" value="1.5"></span>
<span class="mono" id="stripnote" style="color:var(--mut)"></span>
</div></div>
<div id="strips"><p class="sub">decoding strips&hellip;</p></div>
<script>
const D = __PAYLOAD__;
const M = D.meta, NROT = M.n_rotors, KS = D.ks, KMIN = KS[0], KMAX = KS[KS.length-1];
const NF = D.spec.nf, NT = D.spec.nt;
const COL = ["#4da3ff","#ff6b6b","#57d68b","#c48cff","#ffb347","#4dd0e1","#f06292","#aed581"];
const SPECS = {stft:null, sst:null}, STR = {};
let ready=false, rotOn=[], ksPer=[], stripRot=0, tf="stft", carrier=D.carriers[0].id,
    sf=1, fl=0, fh=D.fmax, tIdx=Math.floor(NT/2), segIdx=0, bw=1.5;
const KI={}; KS.forEach((k,i)=>{KI[k]=i;});
const defK=[...new Set([2,4,Math.round(KMAX*0.70),Math.round(KMAX*0.75)])].filter(k=>KI[k]!==undefined);
for(let r=0;r<NROT;r++){ rotOn.push(r==0); ksPer.push(defK.length?defK.slice():[KS[0]]); }

/* ---- provenance: this figure is evidence, it must describe itself ---- */
const prov=document.getElementById("prov");
const chanWarn = M.rps_channel=="motors_command" ? ' <span class="warn">(commanded, NOT measured)</span>' : "";
const rows=[["recording", M.dataset+" / "+M.recording],
 ["window", M.t0.toFixed(5)+" s + "+M.dur.toFixed(2)+" s"],
 ["RPS channel", M.rps_channel+chanWarn],
 ["RPS note", M.rps_note],
 ["audio", M.sr+" Hz, "+M.n_mics+" mic, "+M.n_rotors+" rotor"],
 ["mic channel", M.channel_label],
 ["rotor mean rates", M.rates.map(x=>x.toFixed(2)).join(" / ")+" rev/s"],
 ["spectrogram", "nfft "+M.nfft+", "+NF+" x "+NT+" bins, STFT + reassigned"],
 ["envelope", M.env_rate_hz+" Hz, "+M.cache],
 ["harmonics", M.k_note],
 ["strips", M.budget],
 ["channel trade", M.channel_trade],
 ["code", M.code_version+" @ "+M.git+", built "+M.built]];
if(M.refined) rows.splice(10,0,["refined labels", M.refined]);
rows.forEach(([k,v])=>{const d=document.createElement("div");d.innerHTML="<b>"+k+"</b> "+v;prov.appendChild(d);});

/* ---- controls ---- */
const chanSel=document.getElementById("chan");
M.channels.forEach(c=>{const o=document.createElement("option");o.value=c.file;o.textContent=c.label;
  if(c.id==M.channel)o.selected=true;chanSel.appendChild(o);});
chanSel.value=(M.channels.find(c=>c.id==M.channel)||{}).file||"";
chanSel.onchange=e=>{ if(typeof location!=="undefined"&&e.target.value) location.href=e.target.value; };
if(M.channels.length<2) chanSel.disabled=true;

const rc=document.getElementById("rotchk");
D.traj.rates.forEach((r,i)=>{const d=document.createElement("div");d.style.cssText="display:flex;align-items:center;gap:6px;margin:2px 0";
 d.innerHTML=`<label style="min-width:104px"><input type="checkbox" ${i==0?"checked":""} data-i="${i}"><span class="sw" style="background:${COL[i%COL.length]}"></span>r${i} <span class="mono">${r}</span></label>`+
 `<input type="text" class="kin" data-i="${i}" value="${ksPer[i].join(",")}" style="width:200px"><button class="kview" data-i="${i}">in view</button>`;
 rc.appendChild(d);});
rc.addEventListener("input",e=>{ if(!e.target.classList.contains("kin"))return;
 const i=+e.target.dataset.i;
 ksPer[i]=e.target.value.split(/[,\s]+/).map(Number).filter(n=>n>=1&&n<=KMAX);
 draw(); drawStrips(); });
rc.addEventListener("click",e=>{ if(!e.target.classList.contains("kview"))return;
 const i=+e.target.dataset.i, g=D.traj.g[i][tIdx];
 const a=Math.max(1,Math.ceil(fl/Math.max(g,1e-6))), b=Math.min(Math.floor(fh/Math.max(g,1e-6)),KMAX); const out=[];
 for(let k=a;k<=b&&out.length<6;k++) out.push(k);
 ksPer[i]=out.length?out:[KS[0]];
 const el=rc.querySelector(`.kin[data-i="${i}"]`); if(el) el.value=ksPer[i].join(",");
 draw(); drawStrips(); });
rc.addEventListener("change",e=>{rotOn[+e.target.dataset.i]=e.target.checked;draw();});

const segSel=document.getElementById("seg");
D.segs.forEach((s,i)=>{const o=document.createElement("option");o.value=i;o.textContent=(s*1000)+" ms";segSel.appendChild(o);});
segSel.onchange=e=>{segIdx=+e.target.value;drawStrips();};
const srotSel=document.getElementById("srot");
D.traj.rates.forEach((r,i)=>{const o=document.createElement("option");o.value=i;o.textContent="r"+i+"  "+r+" rev/s";srotSel.appendChild(o);});
srotSel.onchange=e=>{stripRot=+e.target.value;drawStrips();};
const carSel=document.getElementById("car");
D.carriers.forEach(c=>{const o=document.createElement("option");o.value=c.id;o.textContent=c.label;carSel.appendChild(o);});
carSel.onchange=e=>{carrier=e.target.value;drawStrips();};
if(D.carriers.length<2) carSel.disabled=true;
const tfSel=document.getElementById("tf");
tfSel.onchange=e=>{tf=e.target.value;draw();};
document.getElementById("stripnote").textContent=
  "strips: all "+NROT+" rotors, "+M.k_note+", plain STFT at "+M.env_rate_hz+" Hz";
const flEl=document.getElementById("fl"), fhEl=document.getElementById("fh");
const fpre=document.getElementById("fpre");
const hiK=Math.round(Math.min(D.fmax, 70*D.traj.rates[0]));
[[0,D.fmax],[Math.max(0,hiK-500),Math.min(D.fmax,hiK+500)],[0,Math.min(1200,D.fmax)]].forEach(([a,z])=>{
  const b=document.createElement("button");b.textContent=(a/1000).toFixed(1)+"-"+(z/1000).toFixed(1)+"k";
  b.onclick=()=>{flEl.value=a;fhEl.value=z;fr();};fpre.appendChild(b);});

const cv=document.getElementById("spec"), cx=cv.getContext("2d");
const sc=document.getElementById("slice"), sx=sc.getContext("2d");
function fit(c){c.width=c.clientWidth*devicePixelRatio;c.height=c.getAttribute("height")*devicePixelRatio;c.getContext("2d").setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);}
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fBin=hz=>(hz-D.spec.f0)/(D.spec.f1-D.spec.f0)*(NF-1);
const yOf=(hz,H)=>H-(hz-fl)/(fh-fl)*H;

/* PNG -> Uint8Array.  Blobs ship as grayscale PNG (about half the bytes of raw
   base64, which is what buys the second carrier); one decode at load keeps
   every redraw on plain typed arrays. */
function decodeGray(b64){return new Promise((res,rej)=>{const im=new Image();
  im.onload=()=>{const c=document.createElement("canvas");c.width=im.width;c.height=im.height;
    const g=c.getContext("2d",{willReadFrequently:true});g.drawImage(im,0,0);
    const d=g.getImageData(0,0,im.width,im.height).data;const out=new Uint8Array(im.width*im.height);
    for(let i=0,j=0;i<d.length;i+=4,j++)out[j]=d[i];res(out);};
  im.onerror=rej;im.src="data:image/png;base64,"+b64;});}

/* Native-resolution offscreen canvas, then ONE smoothed scale up: drawing the
   quantised image straight into a wider canvas is what made the old page look
   pixelated. */
function blit(ctx,src,w,h,W,H,gamma){
  const oc=document.createElement("canvas"); oc.width=w; oc.height=h;
  const octx=oc.getContext("2d"); const img=octx.createImageData(w,h);
  for(let i=0,o=0;i<src.length;i++,o+=4){ const c=Math.pow(src[i]/255,gamma);
    img.data[o]=Math.min(255,c*300); img.data[o+1]=Math.min(255,Math.pow(c,1.6)*260);
    img.data[o+2]=Math.min(255,Math.pow(c,0.6)*160+c*40); img.data[o+3]=255; }
  octx.putImageData(img,0,0);
  ctx.imageSmoothingEnabled=true; ctx.imageSmoothingQuality="high";
  ctx.drawImage(oc,0,0,w,h,0,0,W,H);
}

function draw(){
  if(!SPECS[tf])return;
  const SPEC=SPECS[tf];
  fit(cv); const W=cv.clientWidth,H=+cv.getAttribute("height");
  const i0=Math.max(0,Math.floor(fBin(fl))), i1=Math.min(NF-1,Math.ceil(fBin(fh)));
  const nfv=i1-i0+1;
  const buf=new Uint8Array(NT*nfv);
  for(let py=0;py<nfv;py++){ const fi=i1-py;
    for(let px=0;px<NT;px++) buf[py*NT+px]=SPEC[fi*NT+px]; }
  cx.clearRect(0,0,W,H);
  blit(cx,buf,NT,nfv,W,H,0.85);
  const T=D.traj.t;
  for(let r=0;r<NROT;r++){ if(!rotOn[r])continue;
    const lines=[[D.traj.g[r],1,"solid"],[D.traj.g[r],sf,"dash"]];
    if(D.traj.gref) lines.push([D.traj.gref[r],1,"dot"]);
    for(const [g,mult,style] of lines){
      if(style=="dash"&&Math.abs(sf-1)<1e-9)continue;
      for(const k of ksPer[r]){
        cx.beginPath(); cx.strokeStyle=COL[r%COL.length];
        cx.lineWidth=style=="solid"?1.1:1.6;
        cx.setLineDash(style=="dash"?[5,4]:(style=="dot"?[1.5,3]:[]));
        let started=false;
        for(let i=0;i<T.length;i++){ const hz=k*g[i]*mult; if(hz<fl||hz>fh){started=false;continue;}
          const x=i/(T.length-1)*W, y=yOf(hz,H); if(!started){cx.moveTo(x,y);started=true;}else cx.lineTo(x,y); }
        cx.stroke(); } } }
  cx.setLineDash([]);
  cx.strokeStyle=css("--acc"); cx.lineWidth=1.5; const mx=tIdx/(NT-1)*W;
  cx.beginPath(); cx.moveTo(mx,0); cx.lineTo(mx,H); cx.stroke();
  cx.fillStyle=css("--mut"); cx.font="11px ui-monospace";
  for(let i=0;i<=5;i++){const hz=fl+(fh-fl)*i/5;
    cx.fillText((hz/1000).toFixed(2)+"k",3,Math.min(H-2,Math.max(10,yOf(hz,H)-2)));}
  document.getElementById("tv").textContent=(M.t0+D.spec.t0+(D.spec.t1-D.spec.t0)*tIdx/(NT-1)).toFixed(2)+" s";
  drawSlice();
}
function drawSlice(){
  if(!SPECS[tf])return;
  const SPEC=SPECS[tf];
  fit(sc); const W=sc.clientWidth,H=+sc.getAttribute("height");
  sx.clearRect(0,0,W,H);
  const i0=Math.max(0,Math.floor(fBin(fl))), i1=Math.min(NF-1,Math.ceil(fBin(fh)));
  sx.beginPath(); sx.strokeStyle=css("--fg"); sx.lineWidth=1;
  for(let fi=i0;fi<=i1;fi++){ const x=(fi-i0)/Math.max(i1-i0,1)*W, v=SPEC[fi*NT+tIdx]/255, y=H-v*(H-8)-4;
    fi==i0?sx.moveTo(x,y):sx.lineTo(x,y); }
  sx.stroke();
  for(let r=0;r<NROT;r++){ if(!rotOn[r])continue;
    const lines=[[D.traj.g[r][tIdx],1,"solid"],[D.traj.g[r][tIdx],sf,"dash"]];
    if(D.traj.gref) lines.push([D.traj.gref[r][tIdx],1,"dot"]);
    for(const [g,mult,style] of lines){
      if(style=="dash"&&Math.abs(sf-1)<1e-9)continue;
      const kk=Math.min(Math.ceil(fh/Math.max(g*mult,1e-6)), 4000);
      for(let k=1;k<=kk;k++){
        const hz=k*g*mult; if(hz<fl||hz>fh)continue;
        const x=(fBin(hz)-i0)/Math.max(i1-i0,1)*W;
        sx.beginPath(); sx.strokeStyle=COL[r%COL.length];
        sx.globalAlpha=style=="solid"?0.4:0.95; sx.lineWidth=style=="solid"?1:1.2;
        sx.setLineDash(style=="dash"?[3,3]:(style=="dot"?[1.5,3]:[]));
        sx.moveTo(x,0); sx.lineTo(x,H); sx.stroke(); } } }
  sx.globalAlpha=1; sx.setLineDash([]);
}
function missing(host,msg){const p=document.createElement("div");p.className="miss";p.textContent=msg;host.appendChild(p);}
function drawStrips(){
  const host=document.getElementById("strips"); host.innerHTML="";
  if(!ready){host.innerHTML='<p class="sub">decoding strips&hellip;</p>';return;}
  const key=stripRot+"|"+carrier+"|"+segIdx, meta=D.strips[key], arr=STR[key];
  if(!meta||!arr){ missing(host,`no strips for rotor ${stripRot}, carrier ${carrier}, `+
    `${D.segs[segIdx]*1000} ms — this page carries ${Object.keys(D.strips).length} strip stacks`); return; }
  const cname=(D.carriers.find(c=>c.id==carrier)||{}).label||carrier;
  for(const k of ksPer[stripRot]){ const kidx=KI[k];
    if(kidx===undefined){ missing(host,`k=${k} not available — this page carries k=${KMIN}..${KMAX}`+
      (KS.length==KMAX-KMIN+1?" (contiguous)":" (with gaps)")); continue; }
    const wrap=document.createElement("div"); wrap.className="stripwrap";
    const c=document.createElement("canvas"); c.height=110; wrap.appendChild(c);
    const lab=document.createElement("div"); lab.className="striplab";
    const g0=D.traj.g[stripRot][tIdx];
    lab.textContent=`r${stripRot} k=${k}   ${(k*g0/1000).toFixed(2)} kHz   ${cname} carrier   `+
      `${D.segs[segIdx]*1000} ms   res ${(1/D.segs[segIdx]/k).toFixed(3)} rev/s   `+
      `usable +-${(M.env_rate_hz/2/k).toFixed(2)} rev/s`;
    wrap.appendChild(lab); host.appendChild(wrap);
    const W=c.clientWidth||800; c.width=W*devicePixelRatio; c.height=110*devicePixelRatio;
    const g=c.getContext("2d"); g.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
    const H=110, NR=meta.nr, NC=meta.nc, off=kidx*NR*NC;
    const r0=Math.max(0,Math.round((-bw+D.ylim)/(2*D.ylim)*(NR-1)));
    const r1=Math.min(NR-1,Math.round((bw+D.ylim)/(2*D.ylim)*(NR-1)));
    const nrv=Math.max(r1-r0+1,1);
    const buf=new Uint8Array(NC*nrv);
    for(let py=0;py<nrv;py++){ const ri=r1-py;
      for(let px=0;px<NC;px++) buf[py*NC+px]=arr[off+ri*NC+px]; }
    blit(g,buf,NC,nrv,W,H,0.9);
    const yr=rev=>H-(rev+bw)/(2*bw)*H;
    g.strokeStyle="#22d3ee"; g.setLineDash([6,4]); g.lineWidth=1.2;
    g.beginPath(); g.moveTo(0,yr(0)); g.lineTo(W,yr(0)); g.stroke();
    if(Math.abs(sf-1)>1e-9){ const d=(sf-1)*D.traj.rates[stripRot];
      g.strokeStyle="#ff4d4d"; g.setLineDash([]); g.beginPath(); g.moveTo(0,yr(d)); g.lineTo(W,yr(d)); g.stroke(); }
    /* the OTHER trajectory, as an offset curve: with the telemetry carrier the
       refined track should follow the ridge, with the refined carrier the ridge
       should be flat at 0 and telemetry should be the one that wanders */
    if(D.traj.gref){ const a=carrier=="t"?D.traj.gref[stripRot]:D.traj.g[stripRot],
        b=carrier=="t"?D.traj.g[stripRot]:D.traj.gref[stripRot];
      g.strokeStyle="#ffd166"; g.globalAlpha=0.7; g.setLineDash([]); g.lineWidth=1; g.beginPath();
      const span=Math.max(D.spec.t1-D.spec.t0,1e-6);
      for(let px=0;px<=W;px++){ const tt=meta.t0+(meta.t1-meta.t0)*(px/W);
        const ti=Math.max(0,Math.min(NT-1,Math.round((tt-D.spec.t0)/span*(NT-1))));
        const y=yr(a[ti]-b[ti]); px?g.lineTo(px,y):g.moveTo(px,y); }
      g.stroke(); g.globalAlpha=1; }
    g.setLineDash([]);
  }
}
const sfEl=document.getElementById("sf");
sfEl.oninput=e=>{sf=+e.target.value;document.getElementById("sfv").textContent=sf.toFixed(5);draw();drawStrips();};
document.getElementById("sfr").onclick=()=>{sfEl.value=1;sfEl.oninput({target:sfEl});};
document.getElementById("sfc").onclick=()=>{sfEl.value=0.99458;sfEl.oninput({target:sfEl});};
flEl.max=fhEl.max=Math.round(D.fmax); fhEl.value=Math.round(D.fmax);
function fr(){ fl=Math.min(+flEl.value,+fhEl.value-100); fh=Math.max(+fhEl.value,fl+100);
  document.getElementById("flv").textContent=(fl/1000).toFixed(2)+" kHz";
  document.getElementById("fhv").textContent=(fh/1000).toFixed(2)+" kHz"; draw(); }
flEl.oninput=fr; fhEl.oninput=fr;
const bwEl=document.getElementById("bw");
bwEl.oninput=e=>{bw=+e.target.value;document.getElementById("bwv").textContent=bw.toFixed(2);drawStrips();};
let drag=false;
const move=e=>{const r=cv.getBoundingClientRect();tIdx=Math.max(0,Math.min(NT-1,Math.round((e.clientX-r.left)/r.width*(NT-1))));draw();drawStrips();};
cv.addEventListener("pointerdown",e=>{drag=true;move(e);});
addEventListener("pointermove",e=>{if(drag)move(e);}); addEventListener("pointerup",()=>drag=false);
addEventListener("keydown",e=>{if(e.key=="ArrowLeft"){tIdx=Math.max(0,tIdx-1);draw();drawStrips();}if(e.key=="ArrowRight"){tIdx=Math.min(NT-1,tIdx+1);draw();drawStrips();}});
addEventListener("resize",()=>{draw();drawStrips();});

/* Test seam: the verification harness drives these instead of guessing at
   scope, so a render path that only a human could reach is still exercised. */
const api={draw,drawStrips,drawSlice,
  set:o=>{if("tf"in o)tf=o.tf; if("carrier"in o)carrier=o.carrier; if("stripRot"in o)stripRot=o.stripRot;
    if("segIdx"in o)segIdx=o.segIdx; if("bw"in o)bw=o.bw; if("sf"in o)sf=o.sf; if("tIdx"in o)tIdx=o.tIdx;
    if("ks"in o)ksPer[stripRot]=o.ks; if("fl"in o)fl=o.fl; if("fh"in o)fh=o.fh;
    if("rotOn"in o)rotOn=o.rotOn;},
  state:()=>({tf,carrier,stripRot,segIdx,bw,sf,tIdx,fl,fh,ready,strips:Object.keys(STR).length}),
  D};
if(typeof window!=="undefined") window.__comb=api;

(async()=>{
  SPECS.stft=await decodeGray(D.spec.stft);
  SPECS.sst=await decodeGray(D.spec.sst);
  let m=0; for(let i=0;i<SPECS.stft.length;i++)m+=SPECS.stft[i]; m/=SPECS.stft.length;
  if(Math.abs(m-D.spec.mean)>0.5) console.warn("COMB_DECODE_MISMATCH spec",m,D.spec.mean);
  fr(); draw();
  for(const key in D.strips){ const b=D.strips[key];
    STR[key]=await decodeGray(b.data);
    if(STR[key].length!=b.nk*b.nr*b.nc) console.warn("COMB_SHAPE_MISMATCH",key,STR[key].length,b.nk*b.nr*b.nc);
    let s=0; for(let i=0;i<STR[key].length;i++)s+=STR[key][i]; s/=STR[key].length;
    if(Math.abs(s-b.mean)>0.5) console.warn("COMB_DECODE_MISMATCH",key,s,b.mean); }
  ready=true; console.log("COMB_READY strips="+Object.keys(STR).length);
  drawStrips();
})();
</script>
"""

INDEX = r"""<title>Comb explorer — __N__ pages</title>
<style>
:root{--bg:#fbfaf7;--fg:#1f2430;--mut:#5c6472;--line:#ded8cd;--card:#fff;--code:#f1ede4;--acc:#b3452e}
@media (prefers-color-scheme:dark){:root{--bg:#14161c;--fg:#e8e6e0;--mut:#9aa1ad;--line:#2c3038;--card:#1c1f27;--code:#22252d;--acc:#e0704f}}
:root[data-theme="dark"]{--bg:#14161c;--fg:#e8e6e0;--mut:#9aa1ad;--line:#2c3038;--card:#1c1f27;--code:#22252d;--acc:#e0704f}
:root[data-theme="light"]{--bg:#fbfaf7;--fg:#1f2430;--mut:#5c6472;--line:#ded8cd;--card:#fff;--code:#f1ede4;--acc:#b3452e}
body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;margin:0;padding:14px}
h1{font-size:1.15rem;margin:0 0 .6rem}
a{color:var(--acc)}
table{border-collapse:collapse;font-size:.85rem}
td,th{border-bottom:1px solid var(--line);padding:.3rem .8rem;text-align:left}
.mono{font-family:ui-monospace,monospace;font-size:.8rem;color:var(--mut)}
</style>
<h1>Comb explorer — __N__ pages</h1>
<p class="mono">One page per (slice, microphone channel): the strip payload — every rotor x every harmonic x every segment length x every carrier — leaves no room for several channels in one document, and cutting harmonics instead would leave holes in k.</p>
<table><tr><th>recording</th><th>window</th><th>mic channel</th><th>RPS channel</th><th>rates (rev/s)</th><th>size</th><th></th></tr>
__ROWS__
</table>
"""


def page_name(sl: Slice, ch: str) -> str:
    return f"comb_{slug(sl.rid, sl.t0, sl.dur)}__{ch}.html"


def write_page(payload: dict, out: Path) -> int:
    m = payload["meta"]
    title = f"{m['recording']}  t={m['t0']:.3f} s  +{m['dur']:.1f} s  —  {m['channel_label']}"
    html = HTML.replace("__TITLE__", title).replace("__PAYLOAD__", json.dumps(payload))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out.stat().st_size


def write_index(pages: list[tuple[Path, dict, int]], out: Path) -> None:
    rows = []
    for p, payload, size in pages:
        m = payload["meta"]
        rows.append(
            f"<tr><td>{m['dataset']} / {m['recording']}</td>"
            f"<td class='mono'>{m['t0']:.3f} + {m['dur']:.1f} s</td>"
            f"<td class='mono'>{m['channel_label']}</td>"
            f"<td class='mono'>{m['rps_channel']}</td>"
            f"<td class='mono'>{' / '.join(f'{x:.1f}' for x in m['rates'])}</td>"
            f"<td class='mono'>{size / 1e6:.1f} MB</td>"
            f"<td><a href='{p.name}'>open</a></td></tr>"
        )
    out.write_text(INDEX.replace("__N__", str(len(pages))).replace("__ROWS__", "\n".join(rows)))


# ─── CLI ──────────────────────────────────────────────────────────────────────


def slug(rid: str, t0: float, dur: float) -> str:
    return f"{rid}__t{t0:g}__d{dur:g}".replace(".", "p").replace("-", "_")


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
    args.segs = [float(s) for s in str(args.segs).split(",") if s.strip()]

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

    specs = [(args.recording, args.t0, args.dur)] + [parse_add(s) for s in args.add]
    pages: list[tuple[Path, dict, int]] = []
    for rid, t0, dur in specs:
        print(f"\n=== {rid}  t0={t0}  dur={dur}", flush=True)
        sl = load_slice(rid, t0, dur, args.rps_channel)
        print(
            f"[load] {sl.dataset} {sl.rid}: {sl.n_mics} mic, {sl.n_rotors} rotor, {sl.sr} Hz, "
            f"{sl.rps_channel}, rates {np.round(sl.rates, 2)}",
            flush=True,
        )
        refined = load_refined(args.refined, sl)
        for ch, payload in build_payloads(sl, args, refined).items():
            single = len(specs) == 1 and len(payload["meta"]["channels"]) == 1
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
