# Work inventory since last report

- generated: 2026-07-15T22:48:50+01:00
- boundary artifact: writing/reports/2026-07-12_full-flight-sim2real-rps
- boundary commit: 6514627 2026-07-12 full-flight sim2real report: 3-condition results + mean-collapse refutation
- HEAD: 7e1771d 2026-07-14 writeup agent

> Numbers in docs below may predate later fixes — always cross-check the
> newest report/doc before quoting. Sync results before analysis (Rule 5).

## Commits (newest first)

```
7e1771d writeup agent
5c552db Slide-prep note: experiment inventory since the gp-rotor-noise deck
```

## Experiment configs (conf/experiment/)

```
```

## Docs (docs/) — excerpts for added files


## Writing artifacts created/updated in the window

### writing/slides/2026-07-13_rps-synthetic-data-status
```
= Where we left off (July 6)
= Generator fix 1: harmonic linewidth
= Generator fix 1, in detail: linewidth, per rotor (1)
= Generator fix 1, in detail: linewidth, per rotor (2)
= Does it actually look right? Real vs. generated
= Generator fixes 2--3: silence + full flight
= What makes interpolation work: two regularizers
= Interpolating drone textures (1/2)
= Interpolating drone textures (2/2)
= Time-warp augmentation
= But: still not better than the best real-data models
= What is the "analytic static comb" (E8)?
= Training data: what's actually in each recipe
= Per-regime evaluation: setup
= Per-regime results: where predictors fail
= What the predictions actually look like: cruise
= What the predictions actually look like: warm-up
= What the predictions actually look like: ground
= What the predictions actually look like: full flight
= Sim curriculum predictions are twitchier, not just wronger
= Mean-tracking sanity check
= Conclusion
= Bonus: RPS label refinement -- the idea
= Bonus: two ways to read the spectrogram
= Bonus: validating against a hidden truth
```

## Code changes (summary)

```
```

## Untracked candidates (not yet committed)

```
  (none)
```

## Prep notes found (read these fully — often a ready-made narrative seed)

