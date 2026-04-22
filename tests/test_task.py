from postdoc.task import build_task


FAKE_SHA = "0123456789abcdef0123456789abcdef01234567"
FAKE_URL = "git@github.com:user/repo.git"


def _minimal(**overrides):
    kw = dict(command="python train.py --x 1", git_sha=FAKE_SHA, git_url=FAKE_URL)
    kw.update(overrides)
    return build_task(**kw)


def test_build_task_envs_contain_git_info():
    t = _minimal()
    assert t["envs"]["POSTDOC_GIT_SHA"] == FAKE_SHA
    assert t["envs"]["POSTDOC_GIT_URL"] == FAKE_URL
    # Default repo dir matches the existing vast-server convention.
    assert t["envs"]["POSTDOC_REPO_DIR"] == "~/harmonic-noise-suppression"


def test_build_task_resources_default():
    t = _minimal()
    assert t["resources"]["infra"].startswith("ssh/")
    assert t["resources"]["accelerators"] == "*:1"


def test_build_task_zero_gpus_drops_accelerators():
    t = _minimal(gpus=0)
    assert "accelerators" not in t["resources"]


def test_build_task_overrides():
    t = _minimal(
        name="foo",
        gpus=4,
        pool="other-pool",
        repo_dir="/srv/repo",
        envs={"WANDB_MODE": "online", "FOO": "bar"},
    )
    assert t["name"] == "foo"
    assert t["resources"]["accelerators"] == "*:4"
    assert t["resources"]["infra"] == "ssh/other-pool"
    assert t["envs"]["POSTDOC_REPO_DIR"] == "/srv/repo"
    assert t["envs"]["WANDB_MODE"] == "online"
    assert t["envs"]["FOO"] == "bar"
    # Git envs still present.
    assert t["envs"]["POSTDOC_GIT_SHA"] == FAKE_SHA


def test_setup_pins_sha_and_runs_uv_sync():
    t = _minimal()
    setup = t["setup"]
    assert 'git reset --hard "$POSTDOC_GIT_SHA"' in setup
    assert "uv sync" in setup
    assert "curl -LsSf https://astral.sh/uv/install.sh" in setup


def test_run_activates_venv_and_runs_command():
    t = _minimal(command="python train.py")
    run = t["run"]
    assert "source .venv/bin/activate" in run
    assert 'cd "$POSTDOC_REPO_DIR"' in run
    assert run.rstrip().endswith("python train.py")


def test_setup_pulls_dvc():
    t = _minimal()
    assert "dvc pull" in t["setup"]
