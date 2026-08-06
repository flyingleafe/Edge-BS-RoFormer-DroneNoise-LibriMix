"""The comb-explorer page: payload builder + the one HTML/JS template.

This module is the dataset-agnostic core shared by the two front ends:

* ``scripts/displacement/comb_explorer.py`` — the CLI, which resolves DREGON
  and Michael's recordings from disk and writes one ``.html`` file per
  microphone channel, plus an index
* ``plots.comb_widget`` — the notebook widget, which takes a ``tdseries``
  Frame the user already has and displays ONE page with every channel and
  every rotor-speed track in it

The page is one self-contained HTML fragment (no external hosts, no fetch, no
CDN) that shows:

* the spectrogram of the slice, as a plain STFT or SYNCHROSQUEEZED (reassigned)
  in one toggle, with per-rotor harmonic combs — solid at ``k * g_r(t)`` for
  the SELECTED carrier, dashed at ``k * s * g_r(t)`` for a free scale factor
  ``s``, and dotted for every other carrier the page carries
* a linked spectrum cut at a draggable time marker, in the same transform
* per-harmonic DEMODULATED strips: the envelope of harmonic ``k`` after
  heterodyning by ``exp(-i k phi_r)``, frequency axis rescaled to a shaft-rate
  offset ``(f - k g) / k`` in rev/s, so the carrier is the centre line.  Strips
  are always a plain STFT — a reassigned strip would move energy along the very
  axis the strip exists to measure

The scale-factor slider is the instrument the whole page exists for: if the
telemetry-vs-acoustic mismatch is a constant SCALE, one ``s`` puts every dashed
line on every tooth at every time; if it is a time-varying SHAPE error, no
single ``s`` does.

Two things are first class and generalised over:

* **Carriers.**  Any number of named rotor-speed trajectories ship in one page
  and the carrier that drives the combs and the strip demodulation is picked
  in the page.  That is what makes ``motors_measured`` vs ``motors_command``
  vs an acoustically refined track a by-eye comparison instead of a rebuild.
* **Microphone channels.**  Every page names the channel it draws, because the
  common phase-noise term measured in WP18 is predominantly PER-MIC
  (cross-channel coherence 0.065 / 0.237 against 0.81-0.94 for a common-mode
  control), so a mic-averaged picture is not the whole evidence.  A channel is
  in-page state when the payload carries several (the notebook widget), and a
  link to a sibling file when it does not (the CLI, which has a size budget).

Nothing assumes 4 rotors or 8 microphones; both counts come from the data.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import subprocess
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]

CODE_VERSION = "comb_page.py v3 (2026-08-06)"

#: Strip envelope decimation.  44100 / 32 = 1378 Hz keeps +-689 Hz, i.e.
#: +-689/k rev/s of shaft-rate offset — +-7.25 rev/s even at k = 95, so the
#: whole +-6 rev/s bandwidth slider is real data at every harmonic in the set.
#: A decim-100 envelope (441 Hz, +-2.2 rev/s at k=100) draws a black band as
#: soon as the slider is widened, which is why the old ``hk_cache.py`` cache is
#: NOT read here: its entries are decim 100 and would silently truncate.
DECIM = 32

DEFAULT_CACHE = Path(os.environ.get("COMB_CACHE_DIR", ROOT / ".cache/comb_explorer"))


# ─── The three inputs: a slice, its carriers, and the page options ────────────


@dataclass
class Slice:
    """One loaded (recording, t0, dur) window of audio, dataset-agnostic."""

    rid: str
    dataset: str
    t0: float
    dur: float
    sr: int
    audio: np.ndarray  # (C, N)
    note: str = ""

    @property
    def n_mics(self) -> int:
        return int(self.audio.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.audio.shape[1])

    @property
    def t_grid(self) -> np.ndarray:
        """Slice-relative time of every audio sample, seconds."""
        return np.arange(self.n_samples) / self.sr


@dataclass
class Carrier:
    """One named rotor-speed trajectory on the slice's audio sample grid.

    ``id`` is the short token the page uses in its strip keys and its carrier
    selector; ``label`` is what a reader sees.  ``g`` is ``(R, N)`` rev/s.
    """

    id: str
    label: str
    g: np.ndarray
    note: str = ""

    @property
    def n_rotors(self) -> int:
        return int(self.g.shape[0])

    @property
    def rates(self) -> np.ndarray:
        return self.g.mean(axis=1)


@dataclass
class PageOptions:
    """Everything the builder needs that is not the data itself.

    The CLI fills this from argparse; the widget fills it from keyword
    arguments.  ``max_mb`` is a hard per-channel-page budget that shortens the
    strips' TIME axis until it fits — set it to ``None`` (the widget default)
    when there is no budget, because a notebook holds the payload in the
    ``.ipynb`` rather than in a file that has to be moved around.

    ``channels`` are the microphone channels the page can DISPLAY: each one
    costs two small spectrogram images, so every microphone of an array fits.
    ``strip_channels`` (``None`` = all of them) are the ones that additionally
    carry demodulated strips, which cost megabytes each — that is the knob
    that keeps "every mic is selectable" from meaning "every mic is paid for
    in full".
    """

    channels: str | int | list | tuple = "avg"
    strip_channels: str | int | list | tuple | None = None
    ks: str = "1-100"
    k_max: int = 100
    segs: tuple[float, ...] = (0.1, 0.5, 2.0)
    strip_rows: int = 80
    strip_cols: int = 110
    strip_floor: float = 60.0
    strip_top: float = 99.5
    ylim: float = 6.0
    decim: int = DECIM
    nfft: int = 2048
    spec_cols: int = 600
    fmax: float = 10000.0
    max_mb: float | None = 9.0
    cache: bool = False
    cache_dir: Path | None = None
    jobs: int = field(default_factory=lambda: min(4, os.cpu_count() or 1))


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

    def q(x: np.ndarray) -> np.ndarray:
        return np.clip((x - lo) / max(hi - lo, 1e-9) * 255.0, 0, 255).astype(np.uint8)

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


def phase(g_r: np.ndarray, sr: int) -> np.ndarray:
    """Cumulative carrier phase in radians (2 pi * revolutions)."""
    return 2.0 * np.pi * np.cumsum(g_r) / sr


def _build_envelope(args):
    """``(rotor, z (K, C, n_env) complex64, fs_env)`` — audio demodulated by
    ``exp(-i k phi)`` and decimated, for every k and every microphone.

    One call of ``tracking.dsp.demod``, THE demodulation of the stack: the
    same brickwall zoom-IFFT the tracker's envelopes come from, so what the
    page draws is what the tracker sees. The band is the full decimated
    Nyquist (``band_env = 0.5``) because the page's whole point is to show
    the neighbourhood of a tooth, not a tracker's capture band.
    """
    from tracking.dsp import demod

    rotor, audio, sr, g_rot, k_max, decim = args
    phi = phase(g_rot, sr)
    n_env = len(range(0, audio.shape[-1], decim))
    c1 = np.exp(-1j * phi).astype(np.complex64)[None, :]
    z_on, _ = demod(
        audio,
        c1=c1,
        rotor=np.zeros(k_max, dtype=np.int64),
        k=np.arange(1, k_max + 1, dtype=np.int64),
        stride=decim,
        n_env=n_env,
        band_env=0.5,
    )
    return rotor, np.ascontiguousarray(z_on.transpose(1, 0, 2)), sr / decim


def envelope_batch(
    sl: Slice,
    car: Carrier,
    k_max: int,
    decim: int,
    cache_dir: Path | None,
    write_cache: bool,
    jobs: int,
) -> tuple[dict[int, np.ndarray], float, str]:
    """``({rotor: z}, fs_env, provenance)`` for ONE carrier.

    A cache entry is reused only when the recording, window, k ceiling, DECIM
    *and* the carrier trajectory hash all match; anything else is recomputed.
    That is deliberate — the historical decim-100 cache would fit every other
    field and quietly halve the usable bandwidth.
    """
    fs_env = sl.sr / decim
    z_by: dict[int, np.ndarray] = {}
    reused: list[int] = []
    n_rotors = car.n_rotors
    tags = [_carrier_tag(car.g[r]) for r in range(n_rotors)]
    if cache_dir:
        for r in range(n_rotors):
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

    todo = [r for r in range(n_rotors) if r not in z_by]
    if todo:
        t_start = time.time()
        args = [(r, sl.audio, sl.sr, car.g[r], k_max, decim) for r in todo]
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
                    g=car.g[r][:: max(int(sl.sr / fs_env), 1)],
                    fs_env=np.float64(fs_env),
                    decim=np.int32(decim),
                    k_max=np.int32(k_max),
                    sr_audio=np.int32(sl.sr),
                    t_start=np.float64(sl.t0),
                    dur=np.float64(sl.dur),
                    rotor=np.int32(r),
                    rps_channel=car.id,
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


# ─── Selection helpers ────────────────────────────────────────────────────────


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


def parse_channels(spec: str | int | list | tuple, n_mics: int) -> list[str]:
    """``"avg"`` / ``"all"`` / ``"0,3,avg"`` / ``[0, "avg"]`` -> channel ids.

    A channel id is ``avg`` (the incoherent average of every microphone) or
    ``micNN``.  A list or tuple is accepted so a notebook caller does not have
    to build a string.
    """
    if isinstance(spec, (list, tuple)):
        parts = [str(p) for p in spec]
    else:
        parts = str(spec).strip().lower().split(",")
    if len(parts) == 1 and parts[0].strip().lower() == "all":
        return ["avg"] + [f"mic{c:02d}" for c in range(n_mics)]
    out: list[str] = []
    for raw in parts:
        part = raw.strip().lower()
        if not part:
            continue
        if part in ("avg", "mean", "average"):
            if "avg" not in out:
                out.append("avg")
            continue
        # already-normalised ids come back through here when a caller passes
        # the output of a previous parse (the widget's "auto" channel choice)
        c = int(part[3:]) if part.startswith("mic") else int(part)
        if not 0 <= c < n_mics:
            raise ValueError(f"channels: mic {c} out of range (0..{n_mics - 1})")
        cid = f"mic{c:02d}"
        if cid not in out:
            out.append(cid)
    return out or ["avg"]


def channel_mics(ch: str, n_mics: int) -> list[int]:
    return list(range(n_mics)) if ch == "avg" else [int(ch[3:])]


def channel_label(ch: str, n_mics: int) -> str:
    if ch == "avg":
        return f"incoherent average of {n_mics} mic" + ("s" if n_mics != 1 else "")
    return f"mic {int(ch[3:])} alone"


# ─── Encoding ─────────────────────────────────────────────────────────────────


def png_b64(a: np.ndarray) -> str:
    """Grayscale PNG of a 2-D uint8 array, base64.  About 2x smaller than the
    raw base64 the hand-built page used, which is what buys the second carrier
    and the second transform."""
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(a)).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def git_sha() -> str:
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


def build_payload(
    sl: Slice,
    carriers: list[Carrier],
    opts: PageOptions,
    *,
    meta_extra: dict | None = None,
    verbose: bool = True,
) -> dict:
    """One payload holding EVERY requested channel and EVERY carrier.

    The envelope of a rotor is built once per carrier and every channel and
    segment length is cut from it, so N channels cost N times the (cheap) strip
    FFTs, not N times the (expensive) demodulation.

    Use :func:`split_payload` afterwards to cut this into one payload per
    channel when the pages must live in separate files (the CLI).
    """
    if not carriers:
        raise ValueError("build_payload needs at least one carrier")
    n_rotors = carriers[0].n_rotors
    for c in carriers:
        if c.n_rotors != n_rotors:
            raise ValueError(
                f"carrier {c.id!r} has {c.n_rotors} rotors, {carriers[0].id!r} has {n_rotors}"
            )
        if c.g.shape[1] != sl.n_samples:
            raise ValueError(
                f"carrier {c.id!r} is {c.g.shape[1]} samples long, the audio is {sl.n_samples}"
            )
    if len({c.id for c in carriers}) != len(carriers):
        raise ValueError("carrier ids must be unique")

    def say(*a):
        if verbose:
            print(*a, flush=True)

    rates = carriers[0].rates
    chans = parse_channels(opts.channels, sl.n_mics)
    if opts.strip_channels is None:
        strip_chans = list(chans)
    else:
        strip_chans = parse_channels(opts.strip_channels, sl.n_mics)
        chans = chans + [c for c in strip_chans if c not in chans]
    k_ceil = min(opts.k_max, int(np.floor((sl.sr / 2) / max(float(rates.max()), 1e-6))))
    ks = parse_ks(opts.ks, k_ceil)
    if not len(ks):
        raise ValueError(f"ks={opts.ks!r} selects nothing at k <= {k_ceil}")
    k_max = int(ks.max())
    contiguous = bool(len(ks) == ks[-1] - ks[0] + 1)

    specs: dict[str, dict] = {}
    f = np.zeros(1)
    t = np.zeros(1)
    for ch in chans:
        u8, sst, f, t = spectrograms(
            sl, channel_mics(ch, sl.n_mics), opts.nfft, opts.spec_cols, opts.fmax
        )
        specs[ch] = {"stft": u8, "sst": sst}
        say(f"[spec] {ch}: {u8.shape[0]} x {u8.shape[1]} bins, {f[-1]:.0f} Hz")
    nf, nt = specs[chans[0]]["stft"].shape

    cache_dir = Path(opts.cache_dir) if opts.cache_dir else DEFAULT_CACHE
    stacks: dict[str, dict[str, np.ndarray]] = {ch: {} for ch in strip_chans}
    spans: dict[str, tuple[float, float]] = {}
    cache_src = ""
    for car in carriers:
        z_by, fs_env, cache_src_c = envelope_batch(
            sl, car, k_max, opts.decim, cache_dir, opts.cache, opts.jobs
        )
        cache_src = cache_src_c if not cache_src else f"{cache_src}; {car.label}: {cache_src_c}"
        for r in range(n_rotors):
            for si, seg in enumerate(opts.segs):
                for ch in strip_chans:
                    t_s = time.time()
                    st, ta, tb = strip_stack(
                        z_by[r],
                        channel_mics(ch, sl.n_mics),
                        ks,
                        fs_env,
                        seg,
                        opts.ylim,
                        opts.strip_rows,
                        opts.strip_cols,
                        opts.strip_floor,
                        opts.strip_top,
                    )
                    if st is None:
                        continue
                    key = f"{r}|{car.id}|{si}"
                    stacks[ch][key] = st
                    spans[key] = (ta, tb)
                    say(
                        f"[strip] {ch} r{r} {car.label} {seg * 1000:.0f} ms {st.shape} "
                        f"in {time.time() - t_s:.1f} s"
                    )
        del z_by
    if not spans:
        raise ValueError("no strips could be built (window shorter than every segment length)")

    live_segs = sorted({int(k.split("|")[2]) for k in spans})
    remap = {old: new for new, old in enumerate(live_segs)}

    def rekey(key: str) -> str:
        a, b, c = key.split("|")
        return f"{a}|{b}|{remap[int(c)]}"

    for ch in strip_chans:
        stacks[ch] = {rekey(k): v for k, v in stacks[ch].items()}
    spans = {rekey(k): v for k, v in spans.items()}
    segs = [opts.segs[i] for i in live_segs]

    # Encode, then shrink the TIME axis until each channel's page fits the
    # budget.  Rotors, harmonics, channels and carriers are never cut: columns
    # are.  ``max_mb=None`` (the notebook) skips the shrinking entirely.
    cap = opts.strip_cols
    blobs: dict[str, dict[str, dict]] = {}
    for _ in range(8):
        blobs = {}
        worst = 0
        for ch in strip_chans:
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
        say(f"[budget] worst-channel strips {worst / 1e6:.2f} MB at <= {cap} columns")
        if opts.max_mb is None or worst < opts.max_mb * 1e6 * 0.90 or cap <= 8:
            break
        cap = max(int(cap * 0.75), 8)

    t_full = sl.t_grid
    f = np.arange(nf) * (sl.sr / opts.nfft)
    hop = max(int(round((sl.n_samples - opts.nfft) / opts.spec_cols)), 1)
    t = (np.arange(nt) * hop + opts.nfft / 2.0) / sl.sr

    def on_cols(g: np.ndarray) -> list[list[float]]:
        return [[round(float(x), 4) for x in np.interp(t, t_full, g[r])] for r in range(n_rotors)]

    strip_ref = blobs[strip_chans[0]]
    cols_note = ", ".join(
        f"{int(s * 1000)} ms {strip_ref[f'{0}|{carriers[0].id}|{i}']['nc']} col"
        for i, s in enumerate(segs)
    )
    nav = [{"id": c, "label": channel_label(c, sl.n_mics), "file": None} for c in chans]
    primary = carriers[0]
    meta = {
        "recording": sl.rid,
        "dataset": sl.dataset,
        "t0": sl.t0,
        "dur": sl.dur,
        "sr": sl.sr,
        "n_mics": sl.n_mics,
        "n_rotors": n_rotors,
        "rps_channel": primary.id,
        "rps_note": primary.note or sl.note,
        "rates": [round(float(x), 3) for x in rates],
        "channel": chans[0],
        "code_version": CODE_VERSION,
        "git": git_sha(),
        "built": time.strftime("%Y-%m-%d %H:%M"),
        "nfft": opts.nfft,
        "env_rate_hz": round(sl.sr / opts.decim, 2),
        "cache": cache_src,
        "k_note": (
            f"k = {int(ks[0])}..{int(ks[-1])} contiguous"
            if contiguous
            else f"k = {', '.join(str(int(x)) for x in ks[:12])}... NOT contiguous"
        ),
        "budget": (
            f"{n_rotors} rotors x {len(ks)} harmonics x {len(segs)} segment lengths "
            f"({cols_note}) x {len(carriers)} carrier(s) x {len(strip_chans)} strip channel(s) "
            f"({', '.join(strip_chans)}), "
            f"{opts.strip_rows} offset rows, black/white points "
            f"p{opts.strip_floor:g}/p{opts.strip_top:g}, "
            + ("no size cap" if opts.max_mb is None else f"cap {opts.max_mb:.0f} MB")
        ),
        "channel_trade": (
            f"{len(chans)} microphone channel(s) in ONE page, switched above without a rebuild"
            + (
                f"; demodulated strips for {', '.join(strip_chans)} only "
                "(a strip stack costs megabytes, a spectrogram does not)"
                if len(strip_chans) < len(chans)
                else ""
            )
            if len(chans) > 1
            else "one microphone channel in this page"
        ),
    }
    if meta_extra:
        meta.update(meta_extra)

    return {
        "meta": meta,
        "chans": nav,
        "spec": {
            ch: {
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
            }
            for ch in chans
        },
        "strips": blobs,
        "traj": {
            "G": {c.id: on_cols(c.g) for c in carriers},
            "t": [round(float(x), 4) for x in t],
            "rates": [round(float(x), 3) for x in rates],
        },
        "ks": [int(k) for k in ks],
        "segs": segs,
        "carriers": [{"id": c.id, "label": c.label, "note": c.note} for c in carriers],
        "ylim": opts.ylim,
        "fmax": float(f[-1]),
    }


def split_payload(payload: dict, page_file: Callable[[str], str]) -> dict[str, dict]:
    """``{channel_id: single-channel payload}`` for the file-per-channel CLI.

    Each copy keeps the FULL channel list so the selector can navigate, but
    only its own channel's spectrogram and strips, so the size budget applies
    per file.  ``page_file(channel_id)`` names the sibling file.
    """
    out: dict[str, dict] = {}
    nav = [dict(c, file=page_file(c["id"])) for c in payload["chans"]]
    n_mics = payload["meta"]["n_mics"]
    for ch in payload["chans"]:
        cid = ch["id"]
        meta = dict(payload["meta"])
        meta["channel"] = cid
        meta["channel_label"] = channel_label(cid, n_mics)
        meta["channel_trade"] = (
            f"one page per microphone channel: {n_mics} mics x "
            f"{len(payload['ks'])} harmonics x {payload['meta']['n_rotors']} rotors x "
            f"{len(payload['segs'])} segment lengths x {len(payload['carriers'])} carrier(s) "
            "does not fit the size cap, and dropping harmonics would leave holes in k, so "
            "the channel is chosen by the selector above (each choice is its own page)"
        )
        out[cid] = dict(
            payload,
            meta=meta,
            chans=nav,
            spec={cid: payload["spec"][cid]},
            strips=(
                {cid: payload["strips"][cid]} if cid in payload["strips"] else {}
            ),  # a channel can carry a spectrogram and no strips
        )
    return out


def payload_bytes(payload: dict) -> int:
    """Serialised size of one payload, which is what a page or a notebook
    output cell actually costs."""
    return len(json.dumps(payload))


# ─── Page ─────────────────────────────────────────────────────────────────────

CSS = r"""
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
details.panel>summary{cursor:pointer;color:var(--mut);font-size:.8rem;font-family:ui-monospace,monospace}
details.panel[open]>summary{margin-bottom:.4rem}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:3px;vertical-align:-1px}
#strips{display:grid;gap:5px}
.stripwrap{position:relative}
.striplab{position:absolute;left:6px;top:3px;font:600 11px ui-monospace,monospace;color:#fff;text-shadow:0 0 4px #000;pointer-events:none}
.miss{padding:6px 8px;border:1px dashed var(--line);border-radius:4px;color:var(--mut);font:12px ui-monospace,monospace}
.warn{color:var(--acc);font-weight:600}
"""

BODY = r"""<div class="panel"><div class="row">
<span><label>microphone channel</label> <select id="chan"></select></span>
<span><label>rotor-speed series (carrier)</label> <select id="car"></select></span>
<span><label>transform (spectrogram + cut)</label> <select id="tf"><option value="stft">STFT</option><option value="sst">synchrosqueezed</option></select></span>
<span><label><input type="checkbox" id="alt" checked> other carriers (dotted)</label></span>
<span><label>scale factor <b id="sfv" class="mono">1.00000</b></label><br><input type="range" id="sf" min="0.985" max="1.015" step="0.00002" value="1"><button id="sfr" title="scale factor back to 1 and every other-carrier overlay off">reset overlays</button><button id="sfc">0.99458</button></span>
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
<span><label>strip segment</label> <select id="seg"></select></span>
<span><label>strip bandwidth &plusmn;<b id="bwv" class="mono">1.50</b> rev/s</label><br><input type="range" id="bw" min="0.15" max="6" step="0.05" value="1.5"></span>
<span class="mono" id="stripnote" style="color:var(--mut)"></span>
</div></div>
<div id="strips"><p class="sub">decoding strips&hellip;</p></div>

<details class="panel" id="provbox"><summary>what this shows, and where every number came from</summary>
<p class="sub">Spectrogram with per-rotor harmonic overlays. Solid = the SELECTED carrier <span class="mono">k&middot;g(t)</span>; dashed = the selected carrier &times; the scale factor; dotted = every other carrier the page carries. Drag on the spectrogram to move the time marker. Strips are the demodulated envelope of each selected harmonic, frequency axis rescaled to shaft-rate offset <span class="mono">(f&minus;kg)/k</span> in rev/s, so the carrier is the centre line. A pure SCALE error is one slider setting that lands every dashed line on every tooth at every time; a SHAPE error is not.</p>
<div class="prov" id="prov"></div></details>
"""

SCRIPT = r"""<script>
/* One instance per document.  The whole page lives inside an IIFE keyed by a
   unique id, so several explorers can share a document (a notebook cell embeds
   this in its own iframe, but a bare fragment must not collide either). */
(function(){
"use strict";
const UID = "__UID__";
const D = __PAYLOAD__;
const M = D.meta, NROT = M.n_rotors, KS = D.ks, KMIN = KS[0], KMAX = KS[KS.length-1];
const CH0 = M.channel;
const NF = D.spec[CH0].nf, NT = D.spec[CH0].nt;
const COL = ["#4da3ff","#ff6b6b","#57d68b","#c48cff","#ffb347","#4dd0e1","#f06292","#aed581"];
const CARCOL = ["#ffd166","#8bd3dd","#f78fb3","#b5e48c","#e0aaff"];
const DEC = {};            /* channel id -> {stft, sst, strips:{key:Uint8Array}} */
let chan=CH0, ready=false, rotOn=[], ksPer=[], stripRot=0, tf="stft",
    carrier=D.carriers[0].id, sf=1, fl=0, fh=D.fmax, tIdx=Math.floor(NT/2), segIdx=0, bw=1.5,
    alt=true;   /* draw every OTHER carrier as a dotted overlay */
const KI={}; KS.forEach((k,i)=>{KI[k]=i;});
const defK=[...new Set([2,4,Math.round(KMAX*0.70),Math.round(KMAX*0.75)])].filter(k=>KI[k]!==undefined);
for(let r=0;r<NROT;r++){ rotOn.push(r==0); ksPer.push(defK.length?defK.slice():[KS[0]]); }

const ROOT = (typeof document.currentScript!=="undefined" && document.currentScript &&
              document.currentScript.parentNode) || document;
function $(id){ return document.getElementById(id); }
const chanLabel = ch => (D.chans.find(c=>c.id==ch)||{}).label || ch;

/* ---- provenance: this figure is evidence, it must describe itself ---- */
function renderProv(){
  const prov=$("prov"); prov.innerHTML="";
  const chanWarn = M.rps_channel=="motors_command" ? ' <span class="warn">(commanded, NOT measured)</span>' : "";
  const rows=[["recording", M.dataset+" / "+M.recording],
   ["window", M.t0.toFixed(5)+" s + "+M.dur.toFixed(2)+" s"],
   ["RPS channel", M.rps_channel+chanWarn],
   ["RPS note", M.rps_note],
   ["audio", M.sr+" Hz, "+M.n_mics+" mic, "+M.n_rotors+" rotor"],
   ["mic channel", chanLabel(chan)],
   ["rotor mean rates", M.rates.map(x=>x.toFixed(2)).join(" / ")+" rev/s"],
   ["spectrogram", "nfft "+M.nfft+", "+NF+" x "+NT+" bins, STFT + reassigned"],
   ["envelope", M.env_rate_hz+" Hz, "+M.cache],
   ["harmonics", M.k_note],
   ["carriers", D.carriers.map(c=>c.label).join(" | ")],
   ["strips", M.budget],
   ["channel trade", M.channel_trade],
   ["code", M.code_version+" @ "+M.git+", built "+M.built]];
  D.carriers.forEach(c=>{ if(c.note) rows.push(["carrier "+c.label, c.note]); });
  if(M.refined) rows.push(["refined labels", M.refined]);
  rows.forEach(([k,v])=>{const d=document.createElement("div");d.innerHTML="<b>"+k+"</b> "+v;prov.appendChild(d);});
}
renderProv();

/* ---- controls ---- */
const chanSel=$("chan");
D.chans.forEach(c=>{const o=document.createElement("option");o.value=c.id;
  o.textContent=c.label+(D.spec[c.id]&&!(D.strips[c.id]&&Object.keys(D.strips[c.id]).length)?"  (no strips)":"");
  if(c.id==chan)o.selected=true;chanSel.appendChild(o);});
chanSel.value=chan;
chanSel.onchange=e=>{ const id=e.target.value, nav=D.chans.find(c=>c.id==id)||{};
  if(nav.file){ if(typeof location!=="undefined") location.href=nav.file; return; }
  setChannel(id); };
if(D.chans.length<2) chanSel.disabled=true;

const rc=$("rotchk");
D.traj.rates.forEach((r,i)=>{const d=document.createElement("div");d.style.cssText="display:flex;align-items:center;gap:6px;margin:2px 0";
 d.innerHTML=`<label style="min-width:104px"><input type="checkbox" ${i==0?"checked":""} data-i="${i}"><span class="sw" style="background:${COL[i%COL.length]}"></span>r${i} <span class="mono">${r}</span></label>`+
 `<input type="text" class="kin" data-i="${i}" value="${ksPer[i].join(",")}" style="width:200px"><button class="kview" data-i="${i}">in view</button>`;
 rc.appendChild(d);});
rc.addEventListener("input",e=>{ if(!e.target.classList.contains("kin"))return;
 const i=+e.target.dataset.i;
 ksPer[i]=e.target.value.split(/[,\s]+/).map(Number).filter(n=>n>=1&&n<=KMAX);
 draw(); drawStrips(); });
rc.addEventListener("click",e=>{ if(!e.target.classList.contains("kview"))return;
 const i=+e.target.dataset.i, g=D.traj.G[carrier][i][tIdx];
 const a=Math.max(1,Math.ceil(fl/Math.max(g,1e-6))), b=Math.min(Math.floor(fh/Math.max(g,1e-6)),KMAX); const out=[];
 for(let k=a;k<=b&&out.length<6;k++) out.push(k);
 ksPer[i]=out.length?out:[KS[0]];
 const el=rc.querySelector(`.kin[data-i="${i}"]`); if(el) el.value=ksPer[i].join(",");
 draw(); drawStrips(); });
rc.addEventListener("change",e=>{rotOn[+e.target.dataset.i]=e.target.checked;draw();});

const segSel=$("seg");
D.segs.forEach((s,i)=>{const o=document.createElement("option");o.value=i;o.textContent=(s*1000)+" ms";segSel.appendChild(o);});
segSel.onchange=e=>{segIdx=+e.target.value;drawStrips();};
const srotSel=$("srot");
D.traj.rates.forEach((r,i)=>{const o=document.createElement("option");o.value=i;o.textContent="r"+i+"  "+r+" rev/s";srotSel.appendChild(o);});
srotSel.onchange=e=>{stripRot=+e.target.value;drawStrips();};
const carSel=$("car");
D.carriers.forEach(c=>{const o=document.createElement("option");o.value=c.id;o.textContent=c.label;carSel.appendChild(o);});
carSel.onchange=e=>{carrier=e.target.value;draw();drawStrips();};
if(D.carriers.length<2) carSel.disabled=true;
const altEl=$("alt");
if(altEl){ alt=altEl.checked!==false;
  altEl.onchange=e=>{alt=!!e.target.checked;draw();drawStrips();}; }
if(D.carriers.length<2){ alt=false; if(altEl){altEl.checked=false;altEl.disabled=true;} }
const tfSel=$("tf");
tfSel.onchange=e=>{tf=e.target.value;draw();};
$("stripnote").textContent=
  "strips: all "+NROT+" rotors, "+M.k_note+", plain STFT at "+M.env_rate_hz+" Hz";
const flEl=$("fl"), fhEl=$("fh");
const fpre=$("fpre");
const hiK=Math.round(Math.min(D.fmax, 70*D.traj.rates[0]));
[[0,D.fmax],[Math.max(0,hiK-500),Math.min(D.fmax,hiK+500)],[0,Math.min(1200,D.fmax)]].forEach(([a,z])=>{
  const b=document.createElement("button");b.textContent=(a/1000).toFixed(1)+"-"+(z/1000).toFixed(1)+"k";
  b.onclick=()=>{flEl.value=a;fhEl.value=z;fr();};fpre.appendChild(b);});

const cv=$("spec"), cx=cv.getContext("2d");
const sc=$("slice"), sx=sc.getContext("2d");
function fit(c){c.width=c.clientWidth*devicePixelRatio;c.height=c.getAttribute("height")*devicePixelRatio;c.getContext("2d").setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);}
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fBin=hz=>(hz-D.spec[chan].f0)/(D.spec[chan].f1-D.spec[chan].f0)*(NF-1);
const yOf=(hz,H)=>H-(hz-fl)/(fh-fl)*H;

/* PNG -> Uint8Array.  Blobs ship as grayscale PNG (about half the bytes of raw
   base64, which is what buys the extra carriers); one decode per channel keeps
   every redraw on plain typed arrays. */
function decodeGray(b64){return new Promise((res,rej)=>{const im=new Image();
  im.onload=()=>{const c=document.createElement("canvas");c.width=im.width;c.height=im.height;
    const g=c.getContext("2d",{willReadFrequently:true});g.drawImage(im,0,0);
    const d=g.getImageData(0,0,im.width,im.height).data;const out=new Uint8Array(im.width*im.height);
    for(let i=0,j=0;i<d.length;i+=4,j++)out[j]=d[i];res(out);};
  im.onerror=rej;im.src="data:image/png;base64,"+b64;});}

function meanOf(a){let s=0;for(let i=0;i<a.length;i++)s+=a[i];return s/a.length;}

/* Channels are decoded LAZILY: a notebook page can carry every microphone, and
   decoding all of them at load would cost both the time and the memory of
   channels the reader never opens. */
async function ensure(ch){
  if(DEC[ch]) return DEC[ch];
  if(!D.spec[ch]) return null;
  const e={stft:null,sst:null,strips:{}};
  e.stft=await decodeGray(D.spec[ch].stft);
  e.sst=await decodeGray(D.spec[ch].sst);
  if(Math.abs(meanOf(e.stft)-D.spec[ch].mean)>0.5) console.warn("COMB_DECODE_MISMATCH spec",ch,meanOf(e.stft),D.spec[ch].mean);
  if(Math.abs(meanOf(e.sst)-D.spec[ch].mean_sst)>0.5) console.warn("COMB_DECODE_MISMATCH sst",ch,meanOf(e.sst),D.spec[ch].mean_sst);
  const S=D.strips[ch]||{};
  for(const key in S){ const b=S[key]; const a=await decodeGray(b.data);
    if(a.length!=b.nk*b.nr*b.nc) console.warn("COMB_SHAPE_MISMATCH",ch,key,a.length,b.nk*b.nr*b.nc);
    if(Math.abs(meanOf(a)-b.mean)>0.5) console.warn("COMB_DECODE_MISMATCH",ch,key,meanOf(a),b.mean);
    e.strips[key]=a; }
  DEC[ch]=e;
  console.log("COMB_CHANNEL_READY "+ch+" strips="+Object.keys(e.strips).length);
  return e;
}
async function setChannel(ch){
  if(!D.spec[ch]) return null;
  chan=ch; ready=false;
  const host=$("strips"); host.innerHTML='<p class="sub">decoding '+chanLabel(ch)+'&hellip;</p>';
  await ensure(ch);
  ready=true; renderProv(); draw(); drawStrips(); fitHeight();
  return DEC[ch];
}

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

/* The SELECTED carrier is drawn solid, plus its dashed scaled copy while the
   scale factor is off 1, plus — while the "other carriers" box is ticked —
   every other carrier dotted.  Both extra families are OVERLAYS: "reset
   overlays" clears the two of them together, so the page goes back to one
   solid comb per rotor and nothing else.  With a single carrier this is
   exactly the original telemetry-only page. */
function combLines(r){
  const out=[[D.traj.G[carrier][r],1,"solid",COL[r%COL.length]]];
  if(Math.abs(sf-1)>1e-9) out.push([D.traj.G[carrier][r],sf,"dash",COL[r%COL.length]]);
  if(alt) D.carriers.forEach(c=>{ if(c.id!=carrier) out.push([D.traj.G[c.id][r],1,"dot",COL[r%COL.length]]); });
  return out;
}

function draw(){
  const dec=DEC[chan]; if(!dec||!dec[tf])return;
  const SPEC=dec[tf];
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
    for(const [g,mult,style,colr] of combLines(r)){
      for(const k of ksPer[r]){
        cx.beginPath(); cx.strokeStyle=colr;
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
  $("tv").textContent=(M.t0+D.spec[chan].t0+(D.spec[chan].t1-D.spec[chan].t0)*tIdx/(NT-1)).toFixed(2)+" s";
  drawSlice();
}
function drawSlice(){
  const dec=DEC[chan]; if(!dec||!dec[tf])return;
  const SPEC=dec[tf];
  fit(sc); const W=sc.clientWidth,H=+sc.getAttribute("height");
  sx.clearRect(0,0,W,H);
  const i0=Math.max(0,Math.floor(fBin(fl))), i1=Math.min(NF-1,Math.ceil(fBin(fh)));
  sx.beginPath(); sx.strokeStyle=css("--fg"); sx.lineWidth=1;
  for(let fi=i0;fi<=i1;fi++){ const x=(fi-i0)/Math.max(i1-i0,1)*W, v=SPEC[fi*NT+tIdx]/255, y=H-v*(H-8)-4;
    fi==i0?sx.moveTo(x,y):sx.lineTo(x,y); }
  sx.stroke();
  for(let r=0;r<NROT;r++){ if(!rotOn[r])continue;
    for(const [gt,mult,style,colr] of combLines(r)){
      const g=gt[tIdx];
      const kk=Math.min(Math.ceil(fh/Math.max(g*mult,1e-6)), 4000);
      for(let k=1;k<=kk;k++){
        const hz=k*g*mult; if(hz<fl||hz>fh)continue;
        const x=(fBin(hz)-i0)/Math.max(i1-i0,1)*W;
        sx.beginPath(); sx.strokeStyle=colr;
        sx.globalAlpha=style=="solid"?0.4:0.95; sx.lineWidth=style=="solid"?1:1.2;
        sx.setLineDash(style=="dash"?[3,3]:(style=="dot"?[1.5,3]:[]));
        sx.moveTo(x,0); sx.lineTo(x,H); sx.stroke(); } } }
  sx.globalAlpha=1; sx.setLineDash([]);
}
function missing(host,msg){const p=document.createElement("div");p.className="miss";p.textContent=msg;host.appendChild(p);}
function drawStrips(){
  const host=$("strips"); host.innerHTML="";
  const dec=DEC[chan];
  if(!ready||!dec){host.innerHTML='<p class="sub">decoding strips&hellip;</p>';return;}
  const key=stripRot+"|"+carrier+"|"+segIdx, meta=(D.strips[chan]||{})[key], arr=dec.strips[key];
  if(!meta||!arr){
    const withStrips=Object.keys(D.strips).filter(c=>Object.keys(D.strips[c]||{}).length);
    missing(host,`no strips for ${chanLabel(chan)}, rotor ${stripRot}, carrier ${carrier}, `+
    `${D.segs[segIdx]*1000} ms — demodulated strips are carried for `+
    `${withStrips.map(chanLabel).join(", ")||"no channel"} `+
    `(every channel has its own spectrogram; rebuild with strip_channels= for more)`); fitHeight(); return; }
  const cname=(D.carriers.find(c=>c.id==carrier)||{}).label||carrier;
  for(const k of ksPer[stripRot]){ const kidx=KI[k];
    if(kidx===undefined){ missing(host,`k=${k} not available — this page carries k=${KMIN}..${KMAX}`+
      (KS.length==KMAX-KMIN+1?" (contiguous)":" (with gaps)")); continue; }
    const wrap=document.createElement("div"); wrap.className="stripwrap";
    const c=document.createElement("canvas"); c.height=110; wrap.appendChild(c);
    const lab=document.createElement("div"); lab.className="striplab";
    const g0=D.traj.G[carrier][stripRot][tIdx];
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
    /* every OTHER carrier, as an offset curve against the selected one: with
       the telemetry carrier a refined track should follow the ridge; with the
       refined carrier the ridge should be flat at 0 and telemetry should be
       the one that wanders.  Same overlay switch as the dotted combs, so
       "reset overlays" clears these too. */
    const span=Math.max(D.spec[chan].t1-D.spec[chan].t0,1e-6);
    let ci=0;
    for(const c2 of (alt?D.carriers:[])){ if(c2.id==carrier) continue;
      const a=D.traj.G[c2.id][stripRot], b=D.traj.G[carrier][stripRot];
      g.strokeStyle=CARCOL[ci%CARCOL.length]; ci++;
      g.globalAlpha=0.7; g.setLineDash([]); g.lineWidth=1; g.beginPath();
      for(let px=0;px<=W;px++){ const tt=meta.t0+(meta.t1-meta.t0)*(px/W);
        const ti=Math.max(0,Math.min(NT-1,Math.round((tt-D.spec[chan].t0)/span*(NT-1))));
        const y=yr(a[ti]-b[ti]); px?g.lineTo(px,y):g.moveTo(px,y); }
      g.stroke(); g.globalAlpha=1; }
    g.setLineDash([]);
  }
  fitHeight();
}
/* In a notebook this page lives in a srcdoc iframe; size the frame to the
   content instead of leaving the reader a 400 px porthole.
   The height is taken from the BODY box and never from the scroll height of
   the document element: that value is at least the viewport, which inside an
   iframe IS the frame's own height, so writing it back into the frame added
   its padding again on every redraw and the notebook cell grew without a stop.  The result is idempotent (a no-op when nothing moved) and
   clamped, so a tall strip stack scrolls inside the frame instead of pushing
   the cell off the screen. */
const MINH=240;
function frameEl(){ try{ return window.frameElement||null; }catch(e){ return null; } }
function maxH(){ const fe=frameEl();
  const v=fe&&fe.dataset?+fe.dataset.maxHeight:0; return v>0?v:2200; }
function contentHeight(){
  const b=document.body;
  if(!b) return MINH;
  const r=b.getBoundingClientRect?b.getBoundingClientRect():null;
  const h=(r&&r.height)||b.scrollHeight||b.offsetHeight||0;
  return Math.ceil(h)+2;
}
function fitHeight(){ const fe=frameEl(); if(!fe||!fe.style) return;
  const want=Math.max(MINH,Math.min(maxH(),contentHeight()));
  if(Math.abs(parseFloat(fe.style.height||"0")-want)>2) fe.style.height=want+"px"; }
const provBox=$("provbox"); if(provBox&&provBox.addEventListener) provBox.addEventListener("toggle",fitHeight);

const sfEl=$("sf");
sfEl.oninput=e=>{sf=+e.target.value;$("sfv").textContent=sf.toFixed(5);draw();drawStrips();};
/* Reset clears EVERY overlay this page draws on top of the solid comb: the
   dashed scaled copy AND the dotted other-carrier traces (and their offset
   curves in the strips).  Clearing only the slider left the dotted lines on
   the figure with no control that removed them. */
$("sfr").onclick=()=>{ sfEl.value=1; sf=1; $("sfv").textContent=(1).toFixed(5);
  alt=false; const a=$("alt"); if(a) a.checked=false; draw(); drawStrips(); };
$("sfc").onclick=()=>{sfEl.value=0.99458;sfEl.oninput({target:sfEl});};
flEl.max=fhEl.max=Math.round(D.fmax); fhEl.value=Math.round(D.fmax);
function fr(){ fl=Math.min(+flEl.value,+fhEl.value-100); fh=Math.max(+fhEl.value,fl+100);
  $("flv").textContent=(fl/1000).toFixed(2)+" kHz";
  $("fhv").textContent=(fh/1000).toFixed(2)+" kHz"; draw(); }
flEl.oninput=fr; fhEl.oninput=fr;
const bwEl=$("bw");
bwEl.oninput=e=>{bw=+e.target.value;$("bwv").textContent=bw.toFixed(2);drawStrips();};
let drag=false;
const move=e=>{const r=cv.getBoundingClientRect();tIdx=Math.max(0,Math.min(NT-1,Math.round((e.clientX-r.left)/r.width*(NT-1))));draw();drawStrips();};
cv.addEventListener("pointerdown",e=>{drag=true;if(typeof window!=="undefined")window.__combActive=UID;move(e);});
addEventListener("pointermove",e=>{if(drag)move(e);}); addEventListener("pointerup",()=>drag=false);
if(typeof window!=="undefined" && !window.__combActive) window.__combActive=UID;
addEventListener("keydown",e=>{ if(typeof window!=="undefined" && window.__combActive!=UID) return;
  if(e.key=="ArrowLeft"){tIdx=Math.max(0,tIdx-1);draw();drawStrips();}
  if(e.key=="ArrowRight"){tIdx=Math.min(NT-1,tIdx+1);draw();drawStrips();}});
addEventListener("resize",()=>{draw();drawStrips();});

/* Test seam: the verification harness drives these instead of guessing at
   scope, so a render path that only a human could reach is still exercised. */
const api={draw,drawStrips,drawSlice,ensure,setChannel,fitHeight,uid:UID,root:ROOT,
  set:o=>{if("tf"in o)tf=o.tf; if("carrier"in o)carrier=o.carrier; if("stripRot"in o)stripRot=o.stripRot;
    if("segIdx"in o)segIdx=o.segIdx; if("bw"in o)bw=o.bw; if("sf"in o)sf=o.sf; if("tIdx"in o)tIdx=o.tIdx;
    if("ks"in o)ksPer[stripRot]=o.ks; if("fl"in o)fl=o.fl; if("fh"in o)fh=o.fh;
    if("rotOn"in o)rotOn=o.rotOn; if("alt"in o)alt=!!o.alt;},
  state:()=>({tf,chan,carrier,stripRot,segIdx,bw,sf,tIdx,fl,fh,ready,alt,
    strips:DEC[chan]?Object.keys(DEC[chan].strips).length:0,
    channels:Object.keys(DEC).length,
    /* the identity of the pixel block on screen: it MUST change with the
       microphone channel, or the selector is decoration */
    specMean:DEC[chan]&&DEC[chan][tf]?meanOf(DEC[chan][tf]):null}),
  D};
if(typeof window!=="undefined"){ window.__comb=api;
  window.__combs=window.__combs||{}; window.__combs[UID]=api; }

(async()=>{
  await ensure(chan);
  ready=true;
  fr(); draw();
  console.log("COMB_READY strips="+Object.keys(DEC[chan].strips).length+" chan="+chan);
  drawStrips(); fitHeight();
})();
})();
</script>
"""

HTML = "<title>Comb explorer — __TITLE__</title>\n<style>" + CSS + "</style>\n" + BODY + SCRIPT

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


def page_title(payload: dict) -> str:
    """The h1 / browser title.  It must not name one channel when the page
    carries several and the reader can switch between them."""
    m = payload["meta"]
    n = len(payload["spec"])
    label = (
        (m.get("channel_label") or channel_label(m["channel"], m["n_mics"]))
        if n == 1
        else f"{n} mic channels"
    )
    return f"{m['recording']}  t={m['t0']:.3f} s  +{m['dur']:.1f} s  —  {label}"


def render_html(payload: dict, *, uid: str | None = None) -> str:
    """The full page fragment for one payload (title + style + body + script)."""
    uid = uid or ("c" + uuid.uuid4().hex[:10])
    return (
        HTML.replace("__TITLE__", page_title(payload))
        .replace("__UID__", uid)
        .replace("__PAYLOAD__", json.dumps(payload))
    )


def write_page(payload: dict, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(payload))
    return out.stat().st_size


def write_index(pages: list[tuple[Path, dict, int]], out: Path) -> None:
    rows = []
    for p, payload, size in pages:
        m = payload["meta"]
        label = m.get("channel_label") or channel_label(m["channel"], m["n_mics"])
        rows.append(
            f"<tr><td>{m['dataset']} / {m['recording']}</td>"
            f"<td class='mono'>{m['t0']:.3f} + {m['dur']:.1f} s</td>"
            f"<td class='mono'>{label}</td>"
            f"<td class='mono'>{m['rps_channel']}</td>"
            f"<td class='mono'>{' / '.join(f'{x:.1f}' for x in m['rates'])}</td>"
            f"<td class='mono'>{size / 1e6:.1f} MB</td>"
            f"<td><a href='{p.name}'>open</a></td></tr>"
        )
    out.write_text(INDEX.replace("__N__", str(len(pages))).replace("__ROWS__", "\n".join(rows)))
