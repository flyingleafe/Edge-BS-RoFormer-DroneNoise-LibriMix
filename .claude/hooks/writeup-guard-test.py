#!/usr/bin/env python3
"""Standalone tests for writeup-guard.py. Run: python3 .claude/hooks/writeup-guard-test.py"""

import json
import os
import subprocess
import sys
import tempfile

GUARD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "writeup-guard.py")


def run_guard(
    root: str,
    tool: str,
    tool_input: dict,
    cwd: str | None = None,
    scope: str | None = None,
    agent_id: str | None = None,
) -> tuple[bool, str]:
    """Returns (allowed, reason)."""
    payload = {
        "session_id": "test",
        "cwd": cwd or root,
        "hook_event_name": "PreToolUse",
        "tool_name": tool,
        "tool_input": tool_input,
    }
    if agent_id is not None:
        payload["agent_id"] = agent_id
    env = dict(os.environ, CLAUDE_PROJECT_DIR=root)
    argv = [sys.executable, GUARD]
    if scope is not None:
        argv.append(f"--scope={scope}")
    proc = subprocess.run(
        argv,
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.stdout.strip():
        out = json.loads(proc.stdout)
        hso = out.get("hookSpecificOutput", {})
        if hso.get("permissionDecision") == "deny":
            return False, hso.get("permissionDecisionReason", "")
    return True, ""


def main() -> None:
    # NOT under /tmp — the jail whitelists /tmp, which would mask every deny case
    base = os.path.expanduser("~/.cache/writeup-guard-tests")
    os.makedirs(base, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as root:
        tdir = "writing/slides/2026-07-13_test-deck"
        os.makedirs(os.path.join(root, tdir, "assets"))
        os.makedirs(os.path.join(root, "writing/reports/2026-07-06_other/assets"))
        os.makedirs(os.path.join(root, ".claude/writeup"))
        os.makedirs(os.path.join(root, "src"))
        with open(os.path.join(root, ".claude/writeup/target.json"), "w") as f:
            json.dump({"dir": tdir, "kind": "slides"}, f)

        abs_t = os.path.join(root, tdir)
        cases = [
            # (expect_allowed, tool, tool_input, label)
            (True, "Write", {"file_path": f"{abs_t}/slides.typ"}, "write inside"),
            (
                True,
                "Write",
                {"file_path": f"{abs_t}/workflow/creator-log.md"},
                "write new subdir file",
            ),
            (False, "Write", {"file_path": f"{root}/src/train.py"}, "write outside"),
            (
                False,
                "Write",
                {"file_path": f"{root}/writing/reports/2026-07-06_other/report.typ"},
                "write sibling artifact",
            ),
            (False, "Edit", {"file_path": f"{abs_t}/../other.typ"}, "dotdot escape"),
            (True, "Write", {"file_path": "/tmp/scratch/x.md"}, "write tmp"),
            (True, "Bash", {"command": f"make -C {tdir} check"}, "make -C inside"),
            (
                False,
                "Bash",
                {"command": "make -C writing/reports/2026-07-06_other"},
                "make -C outside",
            ),
            (
                True,
                "Bash",
                {"command": f"typst compile --root {root} {tdir}/slides.typ"},
                "typst compile",
            ),
            (True, "Bash", {"command": f"python3 {tdir}/prepare.py"}, "run prepare.py"),
            (True, "Bash", {"command": "git log --oneline -20"}, "git log"),
            (True, "Bash", {"command": "git diff HEAD~3 --stat"}, "git diff"),
            (False, "Bash", {"command": "git add -A && git commit -m x"}, "git commit"),
            (False, "Bash", {"command": "git checkout main"}, "git checkout"),
            (False, "Bash", {"command": "git -C src reset --hard"}, "git -C reset"),
            (True, "Bash", {"command": f"mkdir -p {tdir}/assets"}, "mkdir inside"),
            (False, "Bash", {"command": "mkdir -p results/foo"}, "mkdir outside"),
            (False, "Bash", {"command": "rm -rf src"}, "rm outside"),
            (True, "Bash", {"command": f"rm -f {tdir}/check/page-1.png"}, "rm inside"),
            (
                True,
                "Bash",
                {"command": f"cp writing/reports/2026-07-06_other/assets/f.png {tdir}/assets/"},
                "cp outside->inside",
            ),
            (
                False,
                "Bash",
                {"command": f"cp {tdir}/assets/f.png writing/reports/2026-07-06_other/"},
                "cp inside->outside",
            ),
            (False, "Bash", {"command": "echo hi > /etc/motd"}, "redirect outside abs"),
            (False, "Bash", {"command": "echo hi > notes.md"}, "redirect outside rel"),
            (True, "Bash", {"command": f"echo hi > {tdir}/notes.md"}, "redirect inside"),
            (True, "Bash", {"command": "python3 x.py 2>&1"}, "fd dup ok"),
            (
                True,
                "Bash",
                {"command": f"pdftoppm -png {tdir}/slides.pdf {tdir}/check/page"},
                "pdftoppm",
            ),
            (False, "Bash", {"command": "sed -i 's/a/b/' conf/config.yaml"}, "sed -i outside"),
            (True, "Bash", {"command": f"sed -i 's/a/b/' {tdir}/slides.typ"}, "sed -i inside"),
            (False, "Bash", {"command": "curl -o /tmp/x https://example.com"}, "curl denied"),
            (False, "Bash", {"command": "sudo ls"}, "sudo denied"),
            (False, "Bash", {"command": "pip install requests"}, "pip install denied"),
            (False, "Bash", {"command": "uv sync"}, "uv sync denied"),
            (True, "Bash", {"command": "uv run python -c 'print(1)'"}, "uv run ok"),
            (False, "Bash", {"command": "bash -c 'rm -rf src'"}, "bash -c escape"),
            (False, "Bash", {"command": "find . -name '*.png' -delete"}, "find -delete outside"),
            (False, "Bash", {"command": "ls | xargs rm"}, "xargs rm"),
            (False, "Bash", {"command": "tee conf/config.yaml"}, "tee outside"),
            (True, "Bash", {"command": f"ls {root}/src && cat conf/config.yaml"}, "reads fine"),
            (True, "Bash", {"command": "git merge-base --is-ancestor abc def"}, "git merge-base"),
            (True, "Bash", {"command": f"cd {tdir} && rm check/page-1.png"}, "cd inside, rm rel"),
            (False, "Bash", {"command": "cd src && rm train.py"}, "cd outside, rm rel"),
            (
                True,
                "Bash",
                {"command": f"cd {tdir} && echo hi > notes.md"},
                "cd inside, redirect rel",
            ),
            (False, "Bash", {"command": f"cd {tdir} && cd .. && rm -rf x"}, "chained cd escape"),
            (
                True,
                "Bash",
                {"command": f"python3 a.py 2>&1 | tee {tdir}/log.txt"},
                "fd dup + tee inside",
            ),
            (False, "Bash", {"command": "python3 a.py 2>&1 | tee log.txt"}, "fd dup + tee outside"),
            (
                True,
                "Bash",
                {"command": 'grep -n "fig_clipped\\|def clipped" src/prepare.py'},
                "grep with escaped pipe in quotes",
            ),
            (
                False,
                "Bash",
                {"command": 'grep "a\\|b" x.py && rm -rf src'},
                "parse-hard segment + rm outside",
            ),
        ]

        failures = 0
        for expect_allowed, tool, tool_input, label in cases:
            allowed, reason = run_guard(root, tool, tool_input)
            ok = allowed == expect_allowed
            status = "PASS" if ok else "FAIL"
            if not ok:
                failures += 1
                print(
                    f"[{status}] {label}: expected {'allow' if expect_allowed else 'deny'}, "
                    f"got {'allow' if allowed else 'deny'} {('— ' + reason) if reason else ''}"
                )
            else:
                print(f"[{status}] {label}")

        global_cases = [
            # (expect_allowed, tool, tool_input, label, scope, agent_id)
            (
                False,
                "Write",
                {"file_path": f"{root}/src/train.py"},
                "global: manifest+agent_id, write outside -> deny",
                "global",
                "agent-1",
            ),
            (
                True,
                "Write",
                {"file_path": f"{abs_t}/slides.typ"},
                "global: manifest+agent_id, write inside -> allow",
                "global",
                "agent-1",
            ),
            (
                True,
                "Write",
                {"file_path": f"{root}/src/train.py"},
                "global: manifest present, no agent_id, write outside -> allow",
                "global",
                None,
            ),
        ]

        for expect_allowed, tool, tool_input, label, scope, agent_id in global_cases:
            allowed, reason = run_guard(root, tool, tool_input, scope=scope, agent_id=agent_id)
            ok = allowed == expect_allowed
            status = "PASS" if ok else "FAIL"
            if not ok:
                failures += 1
                print(
                    f"[{status}] {label}: expected {'allow' if expect_allowed else 'deny'}, "
                    f"got {'allow' if allowed else 'deny'} {('— ' + reason) if reason else ''}"
                )
            else:
                print(f"[{status}] {label}")

        # creator scope, manifest ABSENT -> deny (fail closed, unchanged default behavior)
        os.remove(os.path.join(root, ".claude/writeup/target.json"))
        allowed, reason = run_guard(root, "Write", {"file_path": f"{abs_t}/slides.typ"})
        if allowed:
            failures += 1
            print("[FAIL] no-manifest (creator scope): expected deny, got allow")
        else:
            print("[PASS] no-manifest (creator scope) denies")

        # global scope, manifest ABSENT, agent_id present, write outside -> allow
        allowed, reason = run_guard(
            root,
            "Write",
            {"file_path": f"{root}/src/train.py"},
            scope="global",
            agent_id="agent-1",
        )
        if not allowed:
            failures += 1
            print(f"[FAIL] global: no manifest, write outside: expected allow, got deny — {reason}")
        else:
            print("[PASS] global: no manifest, write outside allows")

        total = len(cases) + len(global_cases) + 2
        print(f"\n{total - failures}/{total} passed")
        sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
