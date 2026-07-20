#!/usr/bin/env bash
# One-shot omnirun job for the VK fast-inference re-bench (scripts/vk_bench.py):
# full 4-case x 2-config run on the new defaults + three single-case A/B runs
# attributing the win (solver, lp_mode, pruning).
#
# PYTHONPATH pin: omnirun worktrees share the checkout's .venv, whose editable
# install of this project can point at a *stale* sibling worktree — job
# vkbench-opt-77725d (2026-07-20) silently benched old code that way. Putting
# the job worktree's own src/ on PYTHONPATH (which precedes site-packages)
# guarantees the code under test is the code that was pushed; the probe below
# fails fast if the import still resolves somewhere stale.
set -x
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import dataclasses; import data_processing.vk_tracking as m; print("vk module:", m.__file__); assert any(f.name == "solver" for f in dataclasses.fields(m.VKConfig)), "stale vk_tracking import"' || exit 1
python scripts/vk_bench.py --out-suffix _opt
python scripts/vk_bench.py --cases free-flight_nosource_room1 --solver splu --out-suffix _ab_splu
python scripts/vk_bench.py --cases free-flight_nosource_room1 --lp-mode fir --out-suffix _ab_fir
python scripts/vk_bench.py --cases free-flight_nosource_room1 --no-prune --out-suffix _ab_noprune
