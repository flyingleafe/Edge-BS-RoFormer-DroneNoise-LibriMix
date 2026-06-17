# Autoresearch Leaderboard — 20260617-012233-dregon-lm-v4-michaels-simple-conv-v2

Target metrics: PIT MSE (lower is better), R^2 (higher is better, 1.0 is max)

| Rank | Model | Commit | Status | Job ID | PIT MSE | RMSE | MAE frame | MAE clip | R² | Save path | Log | Notes |
|------|-------|--------|--------|--------|---------|------|-----------|----------|----|-----------|-----|-------|
| 1 | simple_conv_v2 | 04bfe18 | completed | 12513837 | 7.8920 | 2.81 | 2.08 | 1.62 | 0.8183 | `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2` | `/gpfs/scratch/acw592/logs/ar_012233_simplecv2.o12513837` | Baseline; early stopped at epoch 31, best epoch 21. |
| 2 | simple_conv_v2_transformer | 1e4314d | completed | 12521795 | 43.5184 | 6.60 | 5.03 | 4.58 | -0.6571 | `/gpfs/scratch/acw592/results/autoresearch/20260617-012233-dregon-lm-v4-michaels-simple-conv-v2/simple_conv_v2_transformer` | `/gpfs/scratch/acw592/logs/ar_012233_v2trans.o12521795` | H1; much worse than baseline; early stopped at epoch 21, best epoch 11. |
