# Registered prediction — wide-anneal ladder row

Registered 2026-08-05, BEFORE the job result was seen.

- Job: `bandadm-wide-85732e` (omnirun, backend `uni-cpu`, 8 CPU, 64 GB, HEAD `10e7322`)
- First attempt `bandadm-wide-443291` failed in stage 1 (blind_fullrange init) with a
  `BrokenProcessPool` after 3 minutes. The cause is memory, not the `--band-b0` change,
  which only enters stage 2. The resubmission asks for 64 GB.
- Command: `python scripts/beatvk_flagship.py --jobs 8 --apps 4 --pi-variant k_anneal
  --band-b0 3.0 --no-synthetic --dregon-dir dload:DREGON --out results/beatvk_bandadm_wide`
- Reference rows: protocol `dregon_cruise` ~1.85 / `fly124_cruise` 2.26; narrow k_anneal
  `fly124_cruise` 2.50.

## Prediction

1. The wide early iterations capture the FLY124 twins. Thus `fly124_cruise` approaches
   the protocol row (~2.26), not the narrow-anneal 2.50.
2. On DREGON the wide early iterations lock onto the displaced low-k comb before the
   anneal narrows. Thus `dregon_cruise` degrades to the protocol level (~1.85). It does
   not stay neutral.

## Falsification

If BOTH `fly124_cruise` and `dregon_cruise` stay good, the wide-anneal variant wins
outright and this prediction is wrong.
