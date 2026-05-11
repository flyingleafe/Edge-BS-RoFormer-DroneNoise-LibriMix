---
name: solve-problem
description: Open-ended problem-solving loop for debugging, implementing features, improving models, or any creative/analytical work. Use whenever the current task requires fixing, adding, improving, or building something.
---

# Solve a Problem

Structured loop for attacking any open-ended task in this repository.

> The Bootstrap and `record-and-remember` closing are global invariants — see root AGENTS.md. This skill only adds task-specific substance.

## Steps

1. **Define the problem.** One or two sentences: the gap between current state and desired state. **State it without project jargon** — if it sounds generic, it probably is, and a tool exists.

   Then **decompose before prescribing.** Ask repeatedly:
   - *"What would it mean to achieve this goal?"* (what is the verification procedure?)
   - *"What would it take to achieve it?"* (what are the open questions?)
   
   Repeat until you have a verifiable specification and a clear decomposition. Only then form a plan or delegate to sub-tasks. Never jump from goal → implementation without this step.

2. **Research-before-build gate (MANDATORY for non-trivial tasks).** Run the
   user-wide `research-before-build` skill. Do not skip. Do not rationalize
   skipping. Concretely: if you are about to write >200 LOC, create new
   abstractions, or reinvent anything in a mature domain (versioning, sync,
   caching, scheduling, config, auth), **open a search tool before opening
   an editor**. Summarize the 2-3 obvious alternatives and surface them to
   the user if the choice isn't trivial.

3. **Check existing knowledge before acting.**
   - Relevant subdirectory `AGENTS.md` for the areas you'll touch
   - Skills index for applicable project skills
   - Similar problems solved before leave traces in `Gotchas` sections

4. **Hypothesize 2–3 approaches.** Briefly. Pick the most promising. At least
   one approach must be "use tool X off the shelf with thin glue" unless step
   2 ruled that out.

5. **Execute iteratively.** Implement → test → verify. If stuck:
   - Re-read relevant AGENTS.md — the answer may already be documented
   - Try the next approach
   - Use `answer-question` if you need to understand existing code first

6. **Verify.** Run tests, check metrics, confirm the behavior actually changed.

## Principles

- **Search before write.** Check if the tool/knowledge exists before reinventing. LLMs have builder bias — counteract it with discipline.
- **Small steps, verify often.** 1 change → verify → repeat beats 10 changes → debug.
- **If confused, AGENTS.md first.** If the doc doesn't answer your question, that's a gap — record it.
- **Prefer composition.** Can existing models/skills/tools be composed, or does something new need to be built? Default: composition. Custom build is a last resort that needs justification.
- **A scaffolding stub is not a directive.** If you find `NotImplementedError` or a `# TODO` in someone else's code, first question whether the scaffold should exist at all, not just how to fill it in.
- **Surface tradeoffs, don't silently downgrade.** If the user asked for X and you're about to do X' because X is hard, say so first.
- **Never jump from opinion → implementation plan.** When a stakeholder gives feedback or a goal, decompose by asking "what would it mean?" and "what would it take?" until you have verifiable sub-goals. Prescribing a sprint before knowing the gap is builder bias.
