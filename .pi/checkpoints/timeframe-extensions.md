# TimeFrame extensions + dataset class migration

**Goal:** Add tags, shift, map_track, rename, resample_audio_tf to TimeFrame/data library, then build DREGONDataset/MichaelsDataset/LibriSpeechDataset classes that produce HF Datasets of TimeFrames.

**Status:** plan approved, pending test-first implementation  
**Last touched:** 2026-06-01  
**Resume on:** local

## Done
- `src/utils/data/` library — UniformSeries, EventSeries, SegmentSeries, TimeFrame with slice/concat algebra. 35 Hypothesis tests passing.
- Root `utils.py` migrated to `src/utils/__init__.py` (backward-compat preserved).
- Design discussion resolved: tags, shift, map_track, rename, resample_audio_tf approved.

## Pending (ordered)

### Phase A: New primitives on existing types (test-first)
1. Write `tests/utils/data/test_tags.py` — 8 tests, invariants T1-T6
2. Write `tests/utils/data/test_shift.py` — 12 tests, invariants S1-S10
3. Write `tests/utils/data/test_map_track.py` — 5 tests, invariants M1-M6  
4. Write `tests/utils/data/test_rename.py` — 4 tests, invariants N1-N4
5. Write `tests/utils/data/test_resample_audio.py` — 5 tests, invariants R1-R5
6. Verify all ~34 new tests FAIL (features don't exist yet)
7. Commit: "test: add failing tests for tags, shift, map_track, rename, resample"
8. Implement: tags on TimeFrame (field + propagation in all ops)
9. Implement: shift on TimeSeries base + all 3 concrete types + TimeFrame
10. Implement: map_track on TimeFrame
11. Implement: rename on TimeFrame
12. Implement: resample_audio_tf (standalone, librosa dep)
13. Verify all ~69 tests PASS (34 new + 35 existing)
14. Commit: "feat: tags, shift, map_track, rename on TimeFrame; resample_audio_tf"

### Phase B: Dataset classes (after Phase A approved)
15. Write `data_processing/datasets.py` with abstract `TimeFrameDataset` base
16. Implement `DREGONDataset(download, load)` using new loaders
17. Impl `MichaelsDataset(download, load)` 
18. Impl `LibriSpeechDataset(download, load)`
19. Remove old record classes: `DREGONRecord`, `MichaelsRecord`, `IMUData`, `MotorData`, `SourcePositionData`
20. Update all consumers (`noise_rps_dataset.py`, `create_dregon_librimix.py`, `external_recordings.py`, `evaluate_rps_predictor.py`)

## State
- Working tree: clean (all changes committed to `better-utils` branch)
- Branch: `better-utils` (merged from `main`)
- `src/utils/data/` files exist, tests passing

## Decisions (do not relitigate)

1. **Tags are `frozenset[str]`** on TimeFrame, not tags-per-track. Simpler, sufficient for split-level filtering.
2. **Tags preserved through ALL ops**: slice, concat, merge, select, drop, with_track, map_track, rename, shift. Concat/merge union tags.
3. **`shift(dt)` is on every type** (base + 3 concrete + TimeFrame), not just TimeFrame. Needed for splicing (align a speech segment to a noise midpoint by shifting both).
4. **`map_track` takes a `TimeSeries → TimeSeries` fn**. Domain must match. Replaces ad-hoc `resample_audio` mutation pattern.
5. **`rename(old, new)` is atomic** — fails if old missing or new occupied.
6. **`resample_audio_tf` is a standalone function**, not a TimeFrame method. Has librosa dependency not suitable for the base library. Relies on `map_track` internally.
7. **Dataset classes: `load(data_dir) → Dataset`** returns Arrow table of specs (NO TimeFrames). `set_transform` added by caller when they need to load. This keeps `load()` fast and `filter()` possible without WAV I/O.
8. **Splits become tags** — DREGON splits map to tags like `"in_flight_noise"`, `"speech"`, `"room1"` etc. Tags are stored as Arrow `list<string>` columns alongside file paths.
9. **`_load_tf` must be `@staticmethod`** (not instance method) so `set_transform` can pickle it by name for `save_to_disk`/`load_from_disk`.

## Open questions

- Where exactly to put dataset classes: new `data_processing/datasets.py` or inside format-specific modules?
- Download methods: what sources (original URLs vs R2)? Credentials story?
- LibriSpeech: which subset to download? train-clean-100? All?
- Mixing: does `mix_at_snr` belong in `data_processing/` as a utility, or is it a `TimeFrame`-level operation?

## Resume

```bash
cd /home/flyingleafe/Research/PhD/projects/harmonic-noise-suppression
git log --oneline -1  # should be on better-utils branch
uv run pytest tests/utils/data/ -q  # 35 existing tests should pass

# Phase A, step 1: write test_tags.py, verify it fails, commit
# Then proceed through steps 2-14
```
