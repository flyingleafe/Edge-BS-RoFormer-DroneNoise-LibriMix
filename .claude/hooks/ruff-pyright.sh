#!/usr/bin/env bash
# PostToolUse hook: enforce ruff + pyright on edited Python files.
# Auto-fixes lint/format issues, then blocks (exit 2) if ruff or pyright
# still report problems, feeding the errors back to Claude to fix.
# Uses the bare ruff/pyright binaries from the nix dev shell (same versions
# the project's pre-commit pins) — NOT `uv run`, which reinstalls each call.

f=$(jq -r '.tool_input.file_path // .tool_response.filePath // empty')
case "$f" in *.py) ;; *) exit 0 ;; esac
[ -f "$f" ] || exit 0

ruff check --fix -q "$f" >/dev/null 2>&1
ruff format -q "$f" >/dev/null 2>&1

err=""
if ! out=$(ruff check "$f" 2>&1); then err="${err}
--- ruff ---
${out}"; fi
if ! out=$(pyright "$f" 2>&1); then err="${err}
--- pyright ---
${out}"; fi

if [ -n "$err" ]; then
    printf 'Lint/type issues in %s — fix before continuing:%s\n' "$f" "$err" >&2
    exit 2
fi
