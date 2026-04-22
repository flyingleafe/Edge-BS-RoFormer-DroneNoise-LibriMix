from postdoc.task import build_bootstrap_task, build_exec_task


FAKE_SHA = "0123456789abcdef0123456789abcdef01234567"
FAKE_URL = "git@github.com:user/repo.git"


# ---------- bootstrap -------------------------------------------------------

def test_bootstrap_has_hostpath_mount():
    t = build_bootstrap_task()
    pod = t["config"]["ssh"]["pod_config"]["spec"]
    vols = {v["name"]: v for v in pod["volumes"]}
    mounts = {m["name"]: m for m in pod["containers"][0]["volumeMounts"]}
    # Project hostPath: repo mounted same-path, read-write.
    assert vols["project"]["hostPath"]["path"].endswith("/harmonic-noise-suppression")
    assert vols["project"]["hostPath"]["type"] == "Directory"
    assert mounts["project"]["mountPath"] == vols["project"]["hostPath"]["path"]
    # SSH creds hostPath: ~root/.ssh from host mounted into /root/.ssh in pod, read-only.
    assert vols["ssh-creds"]["hostPath"]["path"].endswith("/.ssh")
    assert mounts["ssh-creds"]["mountPath"] == "/root/.ssh"
    assert mounts["ssh-creds"].get("readOnly") is True


def test_bootstrap_runs_as_root():
    t = build_bootstrap_task()
    sec = t["config"]["ssh"]["pod_config"]["spec"]["securityContext"]
    assert sec["runAsUser"] == 0


def test_bootstrap_requests_cluster_gpus():
    t = build_bootstrap_task(gpus=2, gpu_type="H100")
    assert t["resources"]["accelerators"] == "H100:2"


def test_bootstrap_passes_pool():
    t = build_bootstrap_task(pool="vast-server")
    assert t["resources"]["infra"] == "ssh/vast-server"


def test_bootstrap_setup_installs_uv_and_syncs():
    t = build_bootstrap_task()
    s = t["setup"]
    assert "astral.sh/uv/install.sh" in s
    assert "uv sync --no-dev" in s


# ---------- exec ------------------------------------------------------------

def _exec(**kw):
    base = dict(command="python train.py", git_sha=FAKE_SHA, git_url=FAKE_URL)
    base.update(kw)
    return build_exec_task(**base)


def test_exec_envs_contain_git_info():
    t = _exec()
    assert t["envs"]["POSTDOC_GIT_SHA"] == FAKE_SHA
    assert t["envs"]["POSTDOC_GIT_URL"] == FAKE_URL
    assert t["envs"]["POSTDOC_REPO_DIR"].endswith("/harmonic-noise-suppression")


def test_exec_resources_only_accelerators_no_infra():
    """sky exec ignores infra; we must not set it."""
    t = _exec()
    assert "infra" not in t["resources"]
    # Default GPU type for vast-server; count from DEFAULT_JOB_GPUS=1.
    assert t["resources"]["accelerators"].endswith(":1")
    assert ":" in t["resources"]["accelerators"]


def test_exec_no_setup_or_workdir():
    t = _exec()
    # These are properties of the cluster, not of the exec'd job.
    assert "setup" not in t
    assert "workdir" not in t
    assert "config" not in t


def test_exec_run_pins_sha_and_syncs():
    t = _exec()
    run = t["run"]
    assert 'git reset --hard "$POSTDOC_GIT_SHA"' in run
    assert "uv sync --no-dev" in run
    assert "source .venv/bin/activate" in run
    assert "dvc pull" in run
    assert run.rstrip().endswith("python train.py")


def test_exec_gpus_zero_drops_accelerators():
    t = _exec(gpus=0)
    assert "accelerators" not in t["resources"]


def test_exec_env_overrides_merge():
    t = _exec(envs={"WANDB_MODE": "online"})
    assert t["envs"]["WANDB_MODE"] == "online"
    # Git envs still there.
    assert t["envs"]["POSTDOC_GIT_SHA"] == FAKE_SHA
