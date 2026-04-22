---
name: record-and-remember
description: Closing step for any interaction where something non-trivial happened — solution, pattern, mistake, or gotcha. Ensures the knowledge base is monotonically improved and nothing is re-learned the hard way.
---

# Record and Remember

Closing step. Runs **iff** something non-trivial happened.

## Gate 1 — did something non-trivial happen?

Ask: did this interaction teach me something, solve something, reveal a mistake, or expose a documentation gap?

- **No** (purely trivial, nothing new) → skip. Done.
- **Yes** → continue to Gate 2.

## Gate 2 — does this learning deserve a new artifact?

Before creating ANY new file (skill, doc, AGENTS.md section), run the
worthiness check from the user-wide `research-before-build` skill (Phase B).
In brief:

1. Who opens the new artifact? (Concrete persona, not "someone".)
2. What concrete event triggers them to open it?
3. What do they do differently — and is the time saved > 5 minutes?

**Default: do not create. Edit an existing file, or do nothing.**

The correct output of most successful tasks is silence. Producing something
to "capture the lesson" is a bias, not a virtue. Most lessons either (a)
are already in an existing skill/doc, or (b) are too project-specific to be
worth recording.

## What to Record

### 1. Concrete details → AGENTS.md or code comments

If you solved a specific problem or learned about a specific area:
- **Update the relevant subdirectory AGENTS.md** — what changed and why
- **Add/update code comments** if the insight is about a specific function/class
- **Update root AGENTS.md** only if project structure or key facts changed

### 2. Transferable patterns → Skills (high bar)

A new skill is justified only if ALL of:
- The pattern will recur ≥ 3 times across different tasks/projects in the foreseeable future.
- No existing skill covers it (even partially — prefer refining over creating).
- It describes a *behavior* (workflow, debugging approach), not a one-time setup.

**Anti-examples** (do NOT create skills for these):
- Setup guides for tools the project already uses.
- "Lessons from session X" or retrospective summaries.
- Content that's already well-covered in a project doc.

When in doubt: **refine an existing skill** rather than create a new one.

### 3. Mistakes and gotchas → Both

- **Add to relevant AGENTS.md** under `Gotchas`
- **Add to relevant skill** as a pitfall or warning

## How to Record

- **AGENTS.md**: concise. Add *why*, not just *what*. Respect existing structure.
- **Skills**: keep under ~2000 tokens. If a skill grows too large, decompose.
- **Code comments**: brief, explain intent not mechanics. Only where understanding is non-obvious.

## Self-check

- Would a new agent find this on first read? If not, make it more prominent.
- Still under 2000 tokens per skill?
- Root AGENTS.md updated if structure or key facts changed?

## Core Principle

> The cost of re-learning is real, but so is the cost of knowledge clutter.
> Prefer editing to creating, one line to one file, and silence to both.
>
> The correct output of most successful tasks is no new artifact at all.
