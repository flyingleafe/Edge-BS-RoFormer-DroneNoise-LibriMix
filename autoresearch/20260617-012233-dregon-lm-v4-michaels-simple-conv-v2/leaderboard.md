# Autoresearch Leaderboard — 20260617-012233-dregon-lm-v4-michaels-simple-conv-v2

Target metrics: PIT MSE (lower is better), R^2 (higher is better, 1.0 is max)

| Rank | Model | Commit | Status | Job ID | PIT MSE | RMSE | MAE frame | MAE clip | R² | Save path | Log | Notes |
|------|-------|--------|--------|--------|---------|------|-----------|----------|----|-----------|-----|-------|
| 1 | simple_conv_v2 | 04bfe18 | completed | 12513837 | 7.8920 | 2.81 | 2.08 | 1.62 | 0.8183 | `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2` | `/gpfs/scratch/acw592/logs/ar_012233_simplecv2.o12513837` | Baseline; early stopped at epoch 31, best epoch 21. |
| — | simple_conv_v2_transformer | TBD | running | 12521795 | TBD | TBD | TBD | TBD | TBD | `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_transformer` | `/gpfs/scratch/acw592/logs/ar_012233_v2trans.o12521795` | H1; smoke-tested `(2, 4, 94)`, initial submit status PENDING so no more candidate submissions this cycle. |
