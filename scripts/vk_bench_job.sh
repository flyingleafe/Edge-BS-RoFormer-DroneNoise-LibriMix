#!/usr/bin/env bash
# Generic omnirun job wrapper for scripts/vk_bench.py — all args pass through
# (e.g. `bash scripts/vk_bench_job.sh --backend torch --device cpu
# --out-suffix _torch_c128`). Successor of vk_bench_opt_job.sh's hardcoded
# run list, sharing its PYTHONPATH pin:
#
# omnirun worktrees share the checkout's .venv, whose editable install of this
# project can point at a *stale* sibling worktree — job vkbench-opt-77725d
# (2026-07-20) silently benched old code that way. Putting the job worktree's
# own src/ on PYTHONPATH (which precedes site-packages) guarantees the code
# under test is the code that was pushed; the probe below fails fast if the
# import still resolves somewhere stale (checked against the newest VKConfig
# field, "backend", from the torch-inference work).
set -x
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import dataclasses; import data_processing.vk_tracking as m; print("vk module:", m.__file__); assert any(f.name == "backend" for f in dataclasses.fields(m.VKConfig)), "stale vk_tracking import"' || exit 1
python scripts/vk_bench.py "$@"
