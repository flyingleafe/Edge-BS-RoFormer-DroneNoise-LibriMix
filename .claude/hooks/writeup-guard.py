#!/usr/bin/env python3
"""PreToolUse guard for the writeup-creator agent (see .claude/agents/writeup-creator.md).

Confines all writes to the active target directory declared in
.claude/writeup/target.json (written by the /writeup orchestrator before the
creator agent is spawned). Reads are unrestricted.

Policy:
  Write/Edit/NotebookEdit  -> file must resolve inside the target dir (or /tmp).
  Bash                     -> deny mutating git subcommands, package-manager /
                              system mutations, network fetches, and write-verb
                              commands (rm, mv, cp, tee, redirects, sed -i, ...)
                              whose targets resolve outside the target dir.
  everything else          -> allowed (this hook is only attached to the four
                              tools above via the agent frontmatter matcher).

Deny = JSON permissionDecision on stdout + exit 0. Silent allow = exit 0, no output.
Fails closed: any internal error denies the call with the error message.
"""

import json
import os
import re
import shlex
import sys

TMP_PREFIXES = ("/tmp/", "/var/tmp/")
CONTROL_TOKENS = {"&&", "||", ";", "|", "&"}

# commands where every path argument must be inside the target dir
ALL_INSIDE = {"rm", "mv", "ln", "truncate", "touch", "mkdir", "rmdir", "shred", "unlink"}
# commands where only the last path argument (the destination) must be inside
DEST_LAST = {"cp", "rsync", "install"}
# always denied outright
FORBIDDEN = {"sudo", "chmod", "chown", "mount", "umount", "curl", "wget", "npm", "npx", "cargo"}
READONLY_GIT = {
    "log",
    "show",
    "diff",
    "status",
    "blame",
    "rev-parse",
    "ls-files",
    "ls-tree",
    "grep",
    "shortlog",
    "describe",
    "cat-file",
    "name-rev",
    "branch",
    "reflog",
    "show-ref",
    "rev-list",
    "diff-tree",
    "merge-base",
}


def deny(reason: str) -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )
    sys.exit(0)


def project_root(cwd: str) -> str:
    root = os.environ.get("CLAUDE_PROJECT_DIR")
    if root:
        return root
    d = cwd
    while d != "/":
        if os.path.isdir(os.path.join(d, ".git")):
            return d
        d = os.path.dirname(d)
    return cwd


def load_target(root: str) -> dict:
    manifest = os.path.join(root, ".claude", "writeup", "target.json")
    if not os.path.isfile(manifest):
        deny(
            "writeup-guard: no active target manifest (.claude/writeup/target.json). "
            "The orchestrator must write it before spawning the creator agent; "
            "all writes are blocked until then."
        )
    with open(manifest) as f:
        return json.load(f)


class Jail:
    def __init__(self, root: str, target: dict, cwd: str):
        self.cwd = cwd
        self.dirs = [os.path.realpath(os.path.join(root, target["dir"]))]
        for extra in target.get("allow_extra", []):
            self.dirs.append(os.path.realpath(os.path.join(root, extra)))
        self.display = target["dir"]

    def allows(self, path: str, cwd: str | None = None) -> bool:
        p = os.path.expanduser(path)
        if not os.path.isabs(p):
            p = os.path.join(cwd or self.cwd, p)
        # realpath the deepest existing ancestor so symlinks can't escape,
        # while paths that don't exist yet (new files) still resolve.
        p = os.path.normpath(p)
        probe, tail = p, ""
        while probe != "/" and not os.path.exists(probe):
            probe, last = os.path.split(probe)
            tail = os.path.join(last, tail) if tail else last
        rp = os.path.normpath(os.path.join(os.path.realpath(probe), tail))
        if rp == "/dev/null":
            return True
        if any(rp.startswith(t) for t in TMP_PREFIXES):
            return True
        return any(rp == d or rp.startswith(d + os.sep) for d in self.dirs)

    def reason(self, what: str) -> str:
        return (
            f"writeup-guard: {what} is outside your allowed directory "
            f"({self.display}). You may only create/modify files inside it "
            f"(or /tmp). If this blocks something you legitimately need, do NOT "
            f"work around it — record the need in workflow/creator-log.md and "
            f"use an in-directory alternative."
        )


def strip_env_assignments(tokens: list[str]) -> list[str]:
    i = 0
    while i < len(tokens) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[i]):
        i += 1
    return tokens[i:]


def path_args(tokens: list[str]) -> list[str]:
    """Non-flag tokens (candidate path arguments)."""
    return [t for t in tokens if t not in CONTROL_TOKENS and not t.startswith("-")]


def check_git(tokens: list[str]) -> str | None:
    """Return a denial reason if this git invocation mutates state."""
    rest = tokens[1:]
    i = 0
    while i < len(rest):
        t = rest[i]
        if t in ("-C", "-c", "--git-dir", "--work-tree"):
            i += 2
            continue
        if t.startswith("-"):
            i += 1
            continue
        sub = t
        if sub not in READONLY_GIT:
            return (
                f"writeup-guard: 'git {sub}' mutates repository state and is not "
                f"allowed for the creator agent. Committing is the orchestrator's "
                f"job. Read-only git (log, show, diff, status, blame, ...) is fine."
            )
        if sub == "branch":
            flags = set(rest[i + 1 :])
            if flags & {"-d", "-D", "-m", "-M", "-c", "-C", "--delete", "--move", "--copy"}:
                return "writeup-guard: mutating 'git branch' invocation is not allowed."
        return None
    return None


def _tokenize(cmd: str) -> list[str] | None:
    """Quote-aware tokenization with shell operators as separate tokens.

    posix mode first; on failure (e.g. backslash-in-double-quotes like
    grep "a\\|b") retry non-posix and strip the quote characters it keeps.
    """
    for posix in (True, False):
        lex = shlex.shlex(cmd, posix=posix, punctuation_chars=True)
        lex.whitespace_split = True
        try:
            toks = list(lex)
        except ValueError:
            continue
        if not posix:
            toks = [t.strip("\"'") for t in toks]
        return toks
    return None


def check_bash_command(cmd: str, jail: Jail, depth: int = 0, cwd: str | None = None) -> str | None:
    """Return a denial reason, or None if the command is acceptable.

    Tokenizes quote-aware (operators split out via shlex punctuation mode),
    walks the token stream into simple-command segments, tracks `cd` across
    segments, and checks redirect targets with the segment's effective cwd.
    """
    if depth > 3:
        return "writeup-guard: command nesting too deep to verify; simplify it."
    cur: str = cwd or jail.cwd

    toks = _tokenize(cmd)
    if toks is None:
        return "writeup-guard: could not parse command; simplify quoting."

    seg: list[str] = []
    redirects: list[str] = []
    i, n = 0, len(toks)

    def flush() -> str | None:
        nonlocal cur
        words = strip_env_assignments(seg)
        red = list(redirects)
        seg.clear()
        redirects.clear()
        if words and os.path.basename(words[0]) == "cd":
            arg = next((t for t in words[1:] if not t.startswith("-")), None)
            cur = (
                os.path.normpath(os.path.join(cur, os.path.expanduser(arg)))
                if arg
                else os.path.expanduser("~")
            )
            return None
        for tgt in red:
            if not jail.allows(tgt, cur):
                return jail.reason(f"redirect target '{tgt}'")
        if not words:
            return None
        return check_simple(words, jail, cur, depth)

    while i < n:
        t = toks[i]
        is_op = t and all(c in "|&;()<>" for c in t)
        if is_op and ">" not in t and "<" not in t:
            r = flush()
            if r:
                return r
            i += 1
            continue
        if is_op and ("<" in t or ">" in t):
            if ">" not in t:
                # pure input redirect / heredoc: skip the source/delimiter word
                i += 2
                continue
            if ">&" in t:
                # fd duplication (2>&1 / >&2): drop the fd digit, skip the dup target
                if seg and seg[-1].isdigit():
                    seg.pop()
                i += 2
                continue
            if seg and seg[-1].isdigit():
                seg.pop()
            tgt = toks[i + 1] if i + 1 < n else ""
            if tgt and not tgt.startswith("&"):
                redirects.append(tgt)
            i += 2
            continue
        seg.append(t)
        i += 1
    return flush()


def check_simple(seg: list[str], jail: Jail, cwd: str, depth: int) -> str | None:
    """Policy checks for one simple command (no operators)."""
    cmd0 = os.path.basename(seg[0])

    if cmd0 in FORBIDDEN:
        return (
            f"writeup-guard: '{cmd0}' is not allowed for the creator agent "
            f"(no system/package/network mutations). Record the need in "
            f"workflow/creator-log.md if it was legitimate."
        )
    if (
        cmd0 in ("pip", "pip3", "uv")
        and len(seg) > 1
        and seg[1] in ("install", "uninstall", "add", "remove", "sync", "lock")
    ):
        return "writeup-guard: modifying the python environment is not allowed."
    if cmd0 == "git":
        r = check_git(seg)
        if r:
            return r
    if cmd0 in ("bash", "sh", "zsh") and "-c" in seg:
        payload = seg[seg.index("-c") + 1] if seg.index("-c") + 1 < len(seg) else ""
        r = check_bash_command(payload, jail, depth + 1, cwd)
        if r:
            return r
    if cmd0 in ("xargs", "parallel") and any(
        os.path.basename(t) in (ALL_INSIDE | DEST_LAST | {"tee", "sed"}) for t in seg[1:]
    ):
        return (
            "writeup-guard: piping into bulk write commands can't be verified; "
            "use explicit per-file commands instead."
        )
    if cmd0 == "find" and ("-delete" in seg or "-exec" in seg):
        for p in path_args(seg[1:2]):
            if not jail.allows(p, cwd):
                return jail.reason(f"find -delete/-exec over '{p}'")
    if cmd0 == "make" and "-C" in seg:
        d = seg[seg.index("-C") + 1] if seg.index("-C") + 1 < len(seg) else ""
        if d and not jail.allows(d, cwd):
            return jail.reason(f"make -C '{d}'")
    if cmd0 in ALL_INSIDE:
        for p in path_args(seg[1:]):
            if not jail.allows(p, cwd):
                return jail.reason(f"'{cmd0} {p}'")
    if cmd0 in DEST_LAST:
        args = path_args(seg[1:])
        if args and not jail.allows(args[-1], cwd):
            return jail.reason(f"'{cmd0}' destination '{args[-1]}'")
    if cmd0 == "tee":
        for p in path_args(seg[1:]):
            if not jail.allows(p, cwd):
                return jail.reason(f"tee target '{p}'")
    if cmd0 == "sed" and any(t.startswith("-i") and not t.startswith("--") for t in seg[1:]):
        # file args = non-flag tokens minus the sed script (first one, or -e operands)
        rest, files, script_seen = seg[1:], [], False
        i = 0
        while i < len(rest):
            t = rest[i]
            if t == "-e":
                script_seen = True
                i += 2
                continue
            if t.startswith("-"):
                i += 1
                continue
            if not script_seen:
                script_seen = True
            else:
                files.append(t)
            i += 1
        for p in files:
            if not jail.allows(p, cwd):
                return jail.reason(f"sed -i on '{p}'")
    if cmd0 == "dd":
        for t in seg[1:]:
            if t.startswith("of=") and not jail.allows(t[3:], cwd):
                return jail.reason(f"dd output '{t[3:]}'")

    return None


def parse_scope(argv: list[str]) -> str:
    for arg in argv:
        if arg.startswith("--scope="):
            return arg.split("=", 1)[1]
    return "creator"


def main() -> None:
    scope = parse_scope(sys.argv[1:])

    data = json.load(sys.stdin)
    cwd = data.get("cwd") or os.getcwd()
    root = project_root(cwd)

    if scope == "global":
        # Global registration fires for every tool call in the session (main
        # thread + every subagent). It must stay a no-op unless a writeup is
        # actually armed (manifest present) and the caller is a subagent
        # (agent_id present) — main-thread calls are never jailed.
        manifest = os.path.join(root, ".claude", "writeup", "target.json")
        if not os.path.isfile(manifest):
            sys.exit(0)
        if "agent_id" not in data:
            sys.exit(0)

    tool = data.get("tool_name", "")
    tool_input = data.get("tool_input") or {}
    target = load_target(root)
    jail = Jail(root, target, cwd)

    if tool in ("Write", "Edit", "NotebookEdit"):
        path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
        if not path or not jail.allows(path):
            deny(jail.reason(f"'{path}'"))
        sys.exit(0)

    if tool == "Bash":
        reason = check_bash_command(tool_input.get("command", ""), jail)
        if reason:
            deny(reason)
        sys.exit(0)

    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # fail closed: every hooked tool is write-capable
        deny(f"writeup-guard: internal error ({e.__class__.__name__}: {e}); denying to be safe.")
