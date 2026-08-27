---
experiment: xrig_dregon_only
training_config: conf/experiment/xrig_dregon_only.yaml
batch: docs/experiments/stochastic-transfer.md
---

# `xrig_dregon_only`

## Motivation

A fair cross-rig bar for the synthetic-only campaign.

`r4hb_scv2` is the campaign's target, but it fine-tunes on Michael's FLY125 and
is scored on FLY124 — same aircraft, same array, an adjacent flight — while on
DREGON it crosses a room. Its Michael's column is therefore near in-domain, and
that is precisely where the synthetic-only gap sits. A synthetic stream may be
losing to an in-domain advantage rather than to anything it could ever close.

This run is `r4hb_scv2` with Michael's removed from the real pool and nothing else
changed: same warm start, same optimizer, same validation. Its Michael's column is
a TRUE cross-rig real number — what a model trained on real drone noise is worth
on an aircraft it has never met. That is the bar a synthetic-only model should
be read against, and it is the only way to tell "synthetic data is weak" from
"real data does not cross rigs either".

Train: `python train.py experiment=xrig_dregon_only`.

## Conclusion

PENDING — the run has not finished.
