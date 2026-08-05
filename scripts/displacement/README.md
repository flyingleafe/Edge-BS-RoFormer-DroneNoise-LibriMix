# `scripts/displacement/` — comb displacement measurement and the explorer page

Measurement code and reports for the DREGON comb-displacement work (see
`docs/experiments/dregon-comb-displacement.md`). This file is about the
interactive explorer only; the analysis scripts around it document themselves in
their own docstrings.

## `comb_explorer.py` — the interactive page

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
harmonics and carriers are never cut. Strip images ship as grayscale PNG, which
is about half the bytes of the raw base64 the hand-built page used.

### Verification — `verify_page.js`

```bash
node scripts/displacement/verify_page.js pages/*.html
```

`node --check` is not sufficient. It parses without executing, and a
temporal-dead-zone `ReferenceError` parses correctly and then kills the whole
page at load. `verify_page.js` runs the page's script against a stubbed DOM,
decodes the PNG payload with a real inflate (so the page's own mean and shape
self-checks are meaningful), and then drives every render path:

- every rotor x every carrier x every segment length x k = 1..k_max
- both spectrogram transforms, at several frequency ranges and time markers
- both bandwidth extremes, the sliders, the presets, the selects, the per-rotor
  k field and the "in view" button
- an out-of-range k, which must leave a VISIBLE placeholder
- the k set must be contiguous, and every channel option must point at a file
  that exists

The exit code is 0 only if nothing threw, nothing warned, and every assertion
held.
