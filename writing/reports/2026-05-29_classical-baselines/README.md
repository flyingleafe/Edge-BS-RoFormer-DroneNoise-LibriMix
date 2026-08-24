# Classical baselines for multi-rotor RPS estimation (2026-05-29)

Archival restore (2026-08-24). This report predates the `writing/reports/`
convention. It lived in `papers/classical_baselines_report/` and was deleted
in the 2026-08 repo cleanup (`4b07042` removed the root implementation file,
`d578cb3` removed the notebook). Sources here are restored verbatim from
commit `00753c4`; `report.pdf` is the built PDF from the author's archive
(`~/Research/PhD/25 May.pdf`).

Content: five classical pitch estimators (PYIN, cepstral, HPS, matched-filter
comb bank, NMF with a harmonic dictionary) extended to four rotors with a
greedy suppression scheme, against SimpleConv on the DREGON-LM test set.
Headline: classical methods are 86-193x worse on the multi-pitch + speech
task, but PYIN and the matched filter are near-perfect on clean single-rotor
recordings — the task, not the methods, is the wall. SimpleConv shows the
mirror failure: it collapses on single-rotor input (structural prior, not
comb reading).

The implementation (`classical_rps_predictors.py`, 537 lines at repo root at
`00753c4`) is restored and adapted as `src/experiments/classical_rps/` for
the unified baseline evaluation on `DREGON-LM-V4-michaels-valid-full` — see
`docs/experiments/unified-baseline-eval.md`.
