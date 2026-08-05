# `scripts/displacement/` — comb displacement measurement and the explorer page

Measurement code and reports for the DREGON comb-displacement work (see
`docs/experiments/dregon-comb-displacement.md`). This file is about the
interactive explorer only; the analysis scripts around it document themselves in
their own docstrings.

## `comb_explorer.py` — the interactive page

The page — the payload builder and the one HTML/JS template — lives in
`src/plots/comb_page.py`. There are two front ends over it and no duplicated JS:

- `comb_explorer.py` (this directory) — the CLI, which resolves DREGON and
  Michael's recordings from disk and writes `.html` files;
- `plots.comb_widget.comb_explorer(frame, ...)` — the notebook widget, which
  takes a `tdseries` Frame the caller already has (see "In a notebook" below).

`comb_explorer.py` builds one self-contained HTML page (no external host, no
fetch, no CDN) for one recording, one time slice and one microphone channel:

- The spectrogram of the slice, as a plain STFT or a synchrosqueezed
  (frequency-reassigned) copy. The toggle changes the spectrogram AND the linked
  spectrum cut. It never changes the strips: reassignment moves energy along the
  frequency axis, which is the axis a strip exists to measure.
- Per-rotor harmonic combs. Solid = telemetry `k*g(t)`, dashed = telemetry times
  a free scale factor (0.985 to 1.015), dotted = the refined trajectory when
  `--refined` supplies one. Each rotor has its own k list and an "in view"
  button.
- Demodulated strips, one for each selected harmonic. A strip is the short-time
  spectrum of harmonic `k` after heterodyning by `exp(-i k phi_r)`, with the
  frequency axis rescaled to a shaft-rate offset `(f - k g)/k` in rev/s. The
  carrier is telemetry, or the refined trajectory when the page carries one.

`hk_core.py` holds the shared loaders (`available_channels`, `load_raw`,
`phase`, `demod_spec`) for DREGON. Michael's recordings (FLY124 / FLY125) come
from `data_processing.sources.michaels`.

### CLI

```bash
# what can be built
python scripts/displacement/comb_explorer.py --list

# the reference slice (DREGON free flight, w01 cruise), mic-averaged
python scripts/displacement/comb_explorer.py \
    --recording free-flight_nosource_room1 --t0 22.56481 --dur 16 \
    --out F0v2.html

# every microphone channel of the same slice, plus the refined comb
python scripts/displacement/comb_explorer.py \
    --recording free-flight_nosource_room1 --t0 22.56481 --dur 16 \
    --channels all --refined refined_labels.npz --out-dir pages/

# a Michael's cruise slice, mic-averaged and mic 2
python scripts/displacement/comb_explorer.py \
    --recording FLY124 --t0 52 --dur 16 --channels avg,2 --out-dir pages/

# several slices in one run, with an index page
python scripts/displacement/comb_explorer.py \
    --recording free-flight_nosource_room1 --t0 22.56481 --dur 16 \
    --add hovering_nosource_room2:10:16 --out-dir pages/
```

Nothing assumes 4 rotors or 8 microphones: both counts come from the data.
DREGON uses `motors_measured` where the recording has it, else `motors_command`.
The page header names the channel it used, and marks `motors_command` as a
commanded value, not a measurement.

### Microphone channels — the payload trade

Each page holds ONE microphone channel, named in the header: a single mic, or
the incoherent average. `--channels` takes `avg`, `all`, or a list such as
`0,3,avg`, and writes one page for each. A selector at the top of the page moves
between the sibling pages, and an `index.html` lists them all.

The channel must be first class, because the common phase-noise term measured in
WP18 is predominantly per-mic (cross-channel coherence 0.065 / 0.237 against
0.81-0.94 for a common-mode control). But 8 mics times 100 harmonics times 4
rotors times 3 segment lengths times 2 carriers does not fit the size cap, and
the alternative — fewer harmonics — would leave holes in k. A hole in k is the
one failure mode this page must not have, so the split is by channel. Each page
states this trade in its header.

### In a notebook — `plots.comb_widget`

```python
from data_processing import sources
from plots.comb_widget import comb_explorer, discover

frame = next(
    f for f in sources.iter_recording_frames("DREGON", splits=["in_flight_noise"])
    if f["meta"]["recording_id"] == "free-flight_nosource_room1"
)
print(discover(frame).describe())          # what it found, before it builds
comb_explorer(frame, t0=22.56481, dur=16.0)
```

`notebooks/comb_explorer_demo.ipynb` runs exactly this, then the same call on
FLY124. `frame` is any `tdseries` Frame, sliced however you like
(`frame.slice["mic", 0:1]`, `frame.slice["rotor", 0:2]`, `frame.time[a:b]`).

Two things are discovered from the Frame instead of being configured:

- **the audio** — the uniformly-sampled entry with a `mic` / `channel` axis (or
  a bare mono `("time",)` entry) at the highest sample rate. 1 microphone or
  many; nothing assumes 8.
- **every rotor-speed track** — each `(rotor, time)` entry, uniform or
  event-sampled, sampled slower than the audio, with a plausible rev/s median.
  On a DREGON frame that is `motors_measured`, `motors_command` and
  `motors_command_raw`; on Michael's it is `rps`; a refined track you added with
  `frame.with_entry("rps_refined", ...)` appears by itself. The `rotor` dim name
  is the strong signal — it is what keeps Michael's `motor_volts`,
  `motor_esctemp` and `motorctrl_pwm` blocks, all `(channel, time)` with a
  median inside the rev/s range, out of the carrier list.

**Every discovered track becomes a selectable carrier in one page**, named by
its frame key. That is the point of the widget: `motors_measured` vs
`motors_command` vs a refined track becomes a by-eye comparison, in one page,
without a rebuild. The selected carrier is drawn solid (plus its dashed scaled
copy) and every other carrier dotted, and it is the carrier the strips are
demodulated by; the strips also draw every other carrier as an offset curve.

**Every requested microphone channel is in the SAME page**, switched in-page.
The CLI splits channels across sibling files because a written page has a 9 MB
budget; a notebook has none, so `max_mb` defaults to `None` and the only cost is
the size of the cell output — which is printed, and warned about above
`warn_mb` (30 MB), because notebook outputs are saved into the `.ipynb`.
`channels=` takes `"auto"` (the default: `avg` plus the loudest single mic),
`"all"`, `"avg"`, `"0,3,avg"` or `[0, 3]`.

Event-sampled telemetry is linearly interpolated onto the audio grid, but NOT
across a hole: a gap longer than `gap_tol` (0.5 s) inside the window, or a
window edge outside the track's coverage, is an error naming the gap — not a
straight line the data does not support. Sub-tolerance gaps are interpolated and
the largest one is named in the page's provenance panel.

Other keywords pass straight through to the builder: `ks`, `k_max`, `segs`,
`decim`, `ylim`, `strip_rows` / `strip_cols`, `nfft`, `spec_cols`, `fmax`,
`jobs`, `cache` / `cache_dir`, and `rps_keys=` to restrict or reorder the
carriers (the first one sets the k ceiling).

The widget renders inside a `srcdoc` iframe. That is not decoration: JupyterLab
does not execute `<script>` tags in HTML output it inserts into its own
document, and the iframe also makes every element id, global and event listener
private to the instance, so two widgets in two cells cannot collide. The page
script is additionally wrapped in an IIFE keyed by a unique instance id, and
registers itself under `window.__combs[uid]`.

### The envelope cache

The expensive product is the demodulated envelope: for each rotor, the audio
multiplied by `exp(-i k phi_r)` and decimated, for k = 1..100 and every mic
(about 35 s of CPU per rotor per carrier, 141 MB in memory for a 16 s window).
Strips for every channel and every segment length are cut from it, so more
channels cost only the cheap strip FFTs.

- `--cache` writes the envelopes to `--cache-dir` (default
  `.cache/comb_explorer`, or `$COMB_CACHE_DIR`). Each file is about 141 MB for a
  16 s window at k = 1..100, so the cache is for a slice you come back to, not
  for a sweep.
- A cache entry is read back only when the recording, the window, the k ceiling,
  the decimation AND a hash of the carrier trajectory all agree. The carrier
  hash is what keeps a refined-carrier run from silently reusing the telemetry
  envelopes.
- Without `--cache` the envelopes are always computed live. That is the default.
- The old `hk_cache.py` cache (`cache/manifest.json`) is deliberately NOT read.
  Its entries are decimation 100, that is a 441 Hz envelope, which gives only
  +-2.2 rev/s of offset at k = 100. The strips must cover the full +-6 rev/s
  bandwidth slider, so this tool demodulates at decimation 32 (1378 Hz,
  +-7.25 rev/s even at k = 95) and ignores anything else.

### Size budget

`--max-mb` (default 9 MB) is a hard cap per page. When the strips do not fit,
the builder shortens the TIME axis of the strips and tries again. Rotors,
harmonics, channels and carriers are never cut. Strip images ship as grayscale
PNG, which is about half the bytes of the raw base64 the hand-built page used.
The widget passes `max_mb=None` — a notebook has no page to move around — and
reports the payload size instead.

### Verification — `verify_page.js`

```bash
node scripts/displacement/verify_page.js pages/*.html widget_output.html
```

`node --check` is not sufficient. It parses without executing, and a
temporal-dead-zone `ReferenceError` parses correctly and then kills the whole
page at load. `verify_page.js` runs the page's script against a stubbed DOM,
decodes the PNG payload with a real inflate (so the page's own mean and shape
self-checks are meaningful), and then drives every render path:

- every MICROPHONE CHANNEL the payload carries — a channel that is in-page state
  has never been rendered until it is selected, so it gets the whole sweep too,
  through `api.setChannel(id)`
- every rotor x every carrier x every segment length x k = 1..k_max
- both spectrogram transforms, at several frequency ranges and time markers
- both bandwidth extremes, the sliders, the presets, the selects, the per-rotor
  k field and the "in view" button
- an out-of-range k, which must leave a VISIBLE placeholder
- the k set must be contiguous; every carrier must carry a trajectory of the
  right shape for every rotor; every channel option must either be in the
  payload or point at a sibling file that exists

It takes a widget's output as well as a CLI page: an `srcdoc="..."` iframe is
unwrapped and the page inside it is driven the same way. Write the widget's HTML
to a file with

```python
from plots.comb_widget import build_widget_payload, widget_html
payload, found = build_widget_payload(frame, t0=22.56481, dur=16.0)
open("widget_output.html", "w").write(widget_html(payload))
```

The exit code is 0 only if nothing threw, nothing warned, and every assertion
held.
