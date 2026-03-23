import os
import pytest
from pathlib import Path

@pytest.fixture
def tmp_results_dir(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    return d

@pytest.fixture
def tmp_data_dir(tmp_path):
    d = tmp_path / "datasets"
    d.mkdir()
    return d

@pytest.fixture
def sample_postdoc_yaml(tmp_path, tmp_results_dir, tmp_data_dir):
    config = tmp_path / "postdoc.yaml"
    config.write_text(f"""\
backend: local
local:
  gpus: 2
  results_dir: {tmp_results_dir}
  data_dir: {tmp_data_dir}
wandb:
  project: test-project
  entity: test-user
""")
    return config

@pytest.fixture
def sample_base_config(tmp_path):
    config = tmp_path / "base_config.yaml"
    config.write_text("""\
audio:
  chunk_size: 131584
  dim_f: 1024
  dim_t: 515
  hop_length: 512
  n_fft: 2048
  num_channels: 1
  sample_rate: 16000
  min_mean_abs: 0.0
model: dcunet
training:
  batch_size: 12
  gradient_accumulation_steps: 1
  grad_clip: 0
  instruments:
    - vocals
    - noise
  lr: 5.0e-4
  patience: 2
  reduce_factor: 0.95
  target_instrument: vocals
  num_epochs: 1000
  num_steps: 200
  data_path:
    - datasets/test-dataset/train
  valid_path:
    - datasets/test-dataset/valid
  q: 0.95
  coarse_loss_clip: true
  ema_momentum: 0.999
  optimizer: adamw
  other_fix: false
  use_amp: true
  early_stop:
    enabled: true
    patience: 5
    metric: si-sdr
inference:
  batch_size: 10
  num_overlap: 4
""")
    return config

@pytest.fixture
def sample_experiment_yaml(tmp_path, sample_base_config, tmp_data_dir):
    exp = tmp_path / "experiment.yaml"
    exp.write_text(f"""\
model:
  type: dcunet
  base_config: {sample_base_config}
  overrides:
    training.lr: 3.0e-4
    training.batch_size: 4
dataset:
  name: test-dataset
wandb:
  tags: [test]
run:
  max_duration: 1h
  checkpoint_interval: 10m
""")
    return exp
