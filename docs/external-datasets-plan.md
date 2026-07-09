# External harmonic-noise datasets — sourcing, formatting, analysis

**Status:** in progress (branch `refactor/dload-omnirun`).
**Goal (C1):** broaden the on-R2 corpus of *rotating-source / harmonic* acoustic
noise beyond drones — industrial machines, propeller aircraft, more drones,
bench motors — each stored as streamable `tdframe-v1` frames with rich
provenance + per-sample metadata, so we can afterwards *measure how harmonic*
each source is and judge its value for ultra-low-SNR speech enhancement.

## Workflow (user directive)

1. **Download + format correctly first** — reproducible fetch → `td.Frame`s with
   rich metadata (what is present, how collected, which real-world system,
   *how observed* incl. source↔observer geometry where recoverable) → commit as
   `tdframe-v1` on R2 → pin.
2. **Then analyze** — measure **harmonicity** (prominence of harmonic peaks) on
   every audio-bearing sample; aggregate per dataset; decide suitability.

## Scope (decided 2026-07-09)

**In (8, cleanly scriptable, audio present):**

| dload name | Source | Size | Audio | License | Observation |
|---|---|---|---|---|---|
| `MIMII` | Zenodo 3384388 | ~100 GB (all 3 SNR tiers) | 8-ch 16 kHz WAV, 10 s | CC BY-SA 4.0 | fixed 8-mic ring (r=10 cm) beside stationary machine |
| `MIMII-DG` | Zenodo 6529888 | 4.4 GB | mono 16 kHz WAV, 10 s | CC BY-NC-SA 4.0 | fixed mic, bench; domain-shift sections |
| `AeroSonicDB` | Zenodo 8371595 | 1.7 GB | mono 22.05 kHz WAV | CC BY-NC 4.0 | ground mic, **aircraft flyover** (moving source) |
| `DroneAudioSet` | HF `ahlab-drone-project/DroneAudioSet` | 42.6 GB | multi-ch (17 mics), verify sr | MIT | **rig-mounted static drone**, arrays above/below @25/50 cm; clean drone-only + source-only stems |
| `drone-detection-samples` | HF `geronimobasso/drone-audio-detection-samples` | 6.8 GB | mono 16 kHz WAV | MIT | mixed provenance; binary drone/no-drone |
| `HornBase` | Mendeley `y5stjsnp8s` v2 | 146 MB | stereo 44.1 kHz WAV, 1 s | CC BY 4.0 | vehicle horns in traffic (tonal, *not* rotating — kept, flagged) |
| `KAIST-rotating-acoustic` | Mendeley `ztmf3m7h5x` v5, `acoustic.zip` only | 47 MB | mic pressure 51.2 kHz in `.mat`, 0 Nm only | CC BY 4.0 | fixed mic, bench; RPM/fault labels in filename |
| `HUSTmotor` | Google Drive folder | ~tens MB | 25.6 kHz `.txt`→wav | none stated (research-only) | bench fault simulator; 6 health × 4 speeds |

**Deferred (audio present, not API-scriptable — needs a manual/granted fetch):**
- `AeroSonicDB-3K` — Zenodo 12775560, ~78 GB, access-gated (files hidden from
  anonymous API; needs a granted Zenodo token).
- `UOEMD-VAFCVS` (UOttawa) — Mendeley `msxs4vj48g`, mic = column 2 of CSV/`.mat`
  @42 kHz; Mendeley file-list API returns empty → browser/mirror only.
- `BLDC-motor-sound` — Mendeley `j4yr5fmhv4`, 43 wav @16 kHz; same empty-API.

**Dropped:**
- Kaggle `zoya77/rotating-equipment-multi-sensor-fault-dataset` — **no audio**
  (scalar sensor CSV).
- Kaggle `ziya07/motor-control-performance-dataset` — **no audio** (control-loop
  telemetry).
- AudioSet *Gears* — no audio shipped; 616 YouTube-ID segments, ~11% rater
  accuracy (Google), third-party copyright hazard → not a reproducible source.

## Architecture (registry-driven; reuses `tdframe-v1`)

Mirrors `derivations.py` + `publish_frame_datasets.py`. Off-the-shelf for
everything generic; net-new only the registry + per-dataset builders + the
harmonicity metric.

- **`src/data_processing/downloaders.py`** — fetch helpers over mature tools:
  `zenodo_fetch` (`GET /api/records/<id>` → `files[].links.self`), `http_fetch`,
  `mendeley_fetch` (`GET /public-api/datasets/<id>/files?...` →
  `content_details.download_url`), `hf_fetch` (`huggingface_hub.snapshot_download`),
  `gdrive_fetch` (`gdown`). Each populates a raw dir; resumable, checksum-logged.
- **`src/data_processing/harmonicity.py`** — torch-free (numpy/scipy/librosa)
  harmonicity metrics (below); usable in a DataLoader worker and on the CPU box.
- **`src/data_processing/external_datasets.py`** — `EXTERNAL_SPECS` registry:
  per dataset a `DownloadSpec` (pinned version + license + provenance) and a
  `builder(raw_dir) -> Iterator[(key, td.Frame)]`. Torch-light (numpy/soundfile/
  librosa/scipy only) so registry integrity + a synthetic build round-trip test
  run on the small box.
- **`scripts/publish_external_datasets.py`** — `download / build / publish / pin`
  driver. `publish` streams `builder` → `repo.commit(name, ...,
  meta={layout: tdframe-v1, ...provenance}, recipe=<script src>)`; one frame in
  memory at a time.
- **Run:** apocrita-cpu omnirun job (`--gpus 0`, compute partition). R2 + HF +
  Kaggle creds ride in `.env`; big downloads land on `/gpfs/scratch`.

## Metadata schema (baked into each frame's `meta`)

Per-sample nested `meta` frame:
- `system` — `category` (`industrial_machine` | `drone` | `aircraft` | `motor` |
  `vehicle_horn`), `make`/`model`, `health`/`fault`, unit id.
- `observation` — `type` (`fixed_array_bench` | `fixed_mic_bench` |
  `rig_mounted_static` | `ground_flyover`), `source_motion` (`static`|`moving`),
  `mic_to_source_m` (when documented), `relative_trajectory`
  (`none`|`scalar_altitude`|`full`). Geometry restored as real `mic_pos`/
  `source_pos` arrays (via `make_recording_frame`) where documented — MIMII
  (8-mic ring r=10 cm) and DroneAudioSet (arrays @25/50 cm above/below).
- `operating` — `throttle`/`rpm`/`load`/`snr_db`/`distance_m`, as available.
- `label` — `class`, `normal_vs_anomaly`, `subclass`.

Dataset-level provenance (manifest `meta`): `source_url`, `doi`, `license`,
`citation`, `collection_method`, `equipment`, `observation_type`,
`sample_rate`/`channels`, `description`.

## Harmonicity metric (`harmonicity.py`)

Per clip (per channel, averaged) on a capped analysis window:
- long-window Welch PSD → estimate fundamental `f0` (cepstrum / harmonic-product
  spectrum, restricted to the plausible rotating range);
- fit the best integer harmonic comb; report
  - `f0_hz`
  - `harmonic_energy_ratio` ∈ [0,1] — energy in ±band around `k·f0` / total
  - `harmonic_to_noise_db` — comb energy vs interpolated broadband floor
  - `n_prominent_harmonics` — count of `k` with peak > floor + 6 dB
  - `spectral_flatness` — Wiener entropy (complementary tonality; low = tonal)

Computed in the **analysis** stage (kept out of the initial publish so the
metric can iterate without re-uploading ~150 GB); cross-checked later against
the project's `multif0`/`salience` models. Output: a per-sample table +
dataset-level aggregate feeding the "is it useful" report.
