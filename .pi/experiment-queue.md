# Experiment Queue — vast-server GPU runs

**Last updated:** 2026-05-30 01:12 UTC  
**All 9 experiments complete ✓**

## Final Leaderboard

| # | Model | MSE↓ | RMSE | MAE↓ | R²↑ | Impr% | Params | Time |
|---|-------|------|------|------|-----|-------|--------|------|
| 1 | **simple_conv_v2** | 2.61 | 1.61 | 0.76 | **0.951** | 99.3% | 1.50M | 70m |
| 2 | simple_conv_bigru_v2 | 2.67 | 1.64 | 0.78 | 0.948 | 99.3% | 1.44M | 60m |
| 3 | simple_conv_bigru | 2.74 | 1.66 | 0.80 | 0.945 | 99.2% | 663K | 63m |
| 4 | simple_conv_tcn | 3.09 | 1.76 | 0.83 | 0.936 | 99.2% | 1.38M | ~55m |
| 5 | simple_conv_magphase_bigru | 3.16 | 1.78 | 0.96 | 0.917 | 99.1% | 666K | 43m |
| 6 | simple_conv_attn_pool | 4.87 | 2.21 | 1.25 | 0.860 | 98.7% | 563K | 58m |
| 7 | simple_conv_wide | 5.04 | 2.24 | 1.32 | 0.847 | 98.6% | 3.94M | 91m |
| 8 | simple_conv_multiscale | 5.15 | 2.27 | 1.31 | 0.840 | 98.6% | 1.36M | ~45m |
| 9 | simple_conv (baseline) | 5.21 | 2.28 | 1.36 | 0.837 | 98.6% | 538K | 65m |
| 10 | simple_conv_se_next | 7.30 | 2.70 | 1.86 | 0.688 | 98.0% | 1.41M | 64m |

## Key Findings

1. **BiGRU temporal head dominates** — every top-5 model has it. Single most impactful component.
2. **v2 wins** (SE + attention + BiGRU + deeper encoder) — best R²=0.951, only marginally ahead of bigru_v2 (0.948) and bigru (0.945).
3. **TCN is best non-BiGRU architecture** — competitive at 0.936 but ~0.015 behind the BiGRU family.
4. **SE-Next is actively harmful** — causes training instability, 18% worse R² than baseline.
5. **Phase information** (magphase) helps vs baseline (+8% R²) but underperforms plain log-mag BiGRU.
6. **Bigger models ≠ better** — wide (3.94M) barely beats baseline. Small dataset (6000 samples) favors compact architectures.
7. **v2 vs bigru_v2 gap is small** — extra components (SE + attention pool) in v2 add only +0.003 R² over bigru_v2 for similar param count.
8. **Multi-scale fusion** is unstable and delivers no gain over baseline.

## Implications

- **Pareto-optimal choice: simple_conv_bigru** (663K params, R²=0.945) — near-best performance with fewest params.
- **If marginal gains matter: simple_conv_v2** (1.50M params, R²=0.951).
- **Ablation next**: strip components from v2 to isolate which matter most.
