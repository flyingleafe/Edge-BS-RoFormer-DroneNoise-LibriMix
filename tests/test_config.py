from pathlib import Path
import pytest
from postdoc.config import PostdocConfig, load_config

def test_load_config_from_file(sample_postdoc_yaml):
    config = load_config(sample_postdoc_yaml)
    assert isinstance(config, PostdocConfig)
    assert config.backend == "local"
    assert config.local.gpus == 2
    assert config.wandb.project == "test-project"

def test_load_config_backend_env_override(sample_postdoc_yaml, monkeypatch):
    monkeypatch.setenv("POSTDOC_BACKEND", "cloud")
    config = load_config(sample_postdoc_yaml)
    assert config.backend == "cloud"

def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/postdoc.yaml"))

def test_load_config_defaults_to_local(tmp_path):
    config_path = tmp_path / "postdoc.yaml"
    config_path.write_text("""\
local:
  gpus: 1
  results_dir: results
  data_dir: datasets
wandb:
  project: test
  entity: test
""")
    config = load_config(config_path)
    assert config.backend == "local"
