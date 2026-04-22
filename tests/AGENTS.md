# tests/ — Postdoc System Tests

Pytest suite for the `postdoc` CLI. Tests the thin surface we own: task-YAML construction and `sky` command-line invocation. Tests do **not** invoke SkyPilot — `subprocess.run` is mocked.

## Files

| File | Tests |
|---|---|
| `conftest.py` | `fake_sky` fixture — monkeypatches `subprocess.run` and `shutil.which`, records every argv |
| `test_task.py` | `build_task` — defaults, overrides, zero-GPU case |
| `test_cli.py` | every subcommand — verifies the `sky` argv we produce |

## Running

```bash
pytest tests/
```

## Not covered (by design)

- Live SkyPilot behaviour — exercised by real runs on vast-server.
- Logs/queue/status output parsing — we pass through to `sky` directly; nothing to test in isolation.
- `infer.py` — currently no unit test; its deps (torch, ml-collections) are heavy. Add a smoke test if/when the resolver logic grows.

## If adding a feature

1. If it builds argv for `sky`, add a `test_cli.py` case that asserts the argv.
2. If it shapes the task YAML, add a `test_task.py` case.
3. If it touches `sky` live behaviour only — don't add a unit test, document it and rely on a real run.
