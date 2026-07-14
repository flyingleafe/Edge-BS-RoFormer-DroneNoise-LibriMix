---
name: writeup
description: >
  End-to-end workflow that produces a polished Typst slide deck or report:
  deterministic work inventory since the last artifact, narrative proposal with
  a user checkpoint, then the writeup-build dynamic workflow (jailed creator
  agent + adversarial critic loop), user review, and a final friction report.
  Use whenever the user asks for a new presentation/deck or a new report on
  recent work. Supersedes calling create-typst-slides / create-typst-report
  directly.
argument-hint: "slides|report [title hint] [--since <ref>] [--rounds N] [--no-user]"
---

# /writeup — orchestrated deck & report creation

You are the **orchestrator**. You never write artifact content yourself. The
build loop is codified in the dynamic workflow `.claude/workflows/writeup-build.js`
(creator = `writeup-creator` subagent, jailed to the artifact dir by
`.claude/hooks/writeup-guard.py`; critic = read-only `writeup-critic`, fresh
per round). Your job: prepare inputs, run the human checkpoints that a workflow
cannot (no mid-run user input), launch the build, verify, and deliver the final
friction report.

**Single-writer invariant:** during the workflow only the creator edits files
under the target dir. You intervene by messaging the creator, never by editing.

## Arguments

- `slides` or `report` (required; if absent, infer from the request, else ask)
- optional title hint (free text)
- `--since <ref>`: override the inventory boundary commit
- `--rounds N`: max critic rounds (default 3)
- `--no-user`: skip user checkpoints (implied when no interactive user is
  present, e.g. when invoked by another agent)

## Phase 0 — resume check

If `.claude/writeup/target.json` exists, a previous run did not finish. Read it,
inspect the target dir's `workflow/` state (`narrative.md`? `check/` rendered?
`critique-round-*.md`?), report the situation, and resume at the first
incomplete phase. All durable state lives on disk, so every phase is resumable.
Two writeups in parallel are not supported — refuse the second.

## Phase 1 — inventory (deterministic)

```bash
bash .claude/skills/writeup/inventory.sh --kind <slides|report> \
    [--since <ref>] --out <scratchpad>/inventory.md
```

Read the output, then the sources it points to: added docs under
`docs/experiments/`, headings of new reports, and any `NEXT-*` prep note under
the kind's directory (often a ready-made narrative seed — use it, then fold it
into the deck dir as the prep note itself instructs). Do NOT skip the newest
report of the window: it usually contains the verified numbers. Also read the
**Discussion/Limitations sections** of the window's reports, not just results —
that's where the causal links live ("X failed *because* Y, which is why we
built Z"), and a narrative without them reads as a list of disconnected wins.

## Phase 2 — narrative proposal

Bucket the window's work into idea/direction groups (an experiment chain is one
group; infra/tooling is a group only if it changes the story). Write the
proposal in exactly this format:

```markdown
# Narrative — <working title>
kind: slides|report
audience: <who + what they already know from previous artifacts>
through-line: <2–4 sentences: the story arc, including the punchline>

## Sections (ordered)
1. <section title> — message: <one sentence> — evidence: <figures/tables + source paths> — sources: <docs/reports to draw from>
...

## Cut (considered, excluded)
- <item> — <why>

## Open questions for the user
- <anything genuinely ambiguous>
```

Rules: every section names its evidence; the punchline gets its own section;
secondary messages after the punchline, not before; 8–14 sections for slides
(≈1–2 slides each), 5–8 for a report.

## Phase 3 — user checkpoint (~10 min timeout)

Skip in `--no-user` mode (proceed with the proposal as-is).

1. Post the full narrative proposal in chat. Say explicitly: *"I'll proceed
   with this narrative in ~10 minutes unless you want changes —
   add/remove/reorder/reprioritize in free form."*
2. Start the timeout: `Bash(command="sleep 600", run_in_background=true)`, then
   end your turn.
3. If the user replies: kill the timer, apply their edits, re-post the changed
   section list (one line per section), and — only if the edits were
   substantial — offer one more (~5 min) window; otherwise proceed.
4. If the timer fires with no reply: proceed with defaults, and note in the
   final report that the narrative was not user-reviewed.

## Phase 4 — arm

1. Decide `<date>_<slug>` (today's date, kebab-case). Target dir:
   `writing/<slides|reports>/<date>_<slug>/`.
2. `mkdir -p <target-dir>/workflow` and write `workflow/narrative.md` (final
   version) and `workflow/inventory.md` (copy from scratchpad).
3. Snapshot the tree baseline **inside the dir** (jail-readable/writable):
   `git status --porcelain > <target-dir>/workflow/baseline-status.txt`
4. Arm the jail — write `.claude/writeup/target.json`:
   `{"dir": "writing/slides/<date>_<slug>", "kind": "slides", "allow_extra": []}`

## Phase 5 — build (the writeup-build workflow)

**Preferred path** — launch the dynamic workflow (RunWorkflow tool, or the user
runs `/writeup-build` in a session where the runtime exists):

```
workflow: writeup-build
args: {"dir": "<target-dir>", "kind": "<kind>", "rounds": <N>,
       "prevArtifacts": ["<2-3 most recent same-kind dirs>"]}
```

The script runs: preflight (arming check) → creator round 1 → per-round
[fresh critic (structured verdict) → tree verification → fresh creator revise
round] → collect (friction log, TODO markers, render freshness). It returns a
JSON summary; the critic rounds and verdicts are in `rounds[]`.

**Fallback path** — if no workflow runtime is available in this session,
execute the *same* loop inline; the script is the specification — read it and
mirror it exactly, using the Agent tool (`subagent_type: writeup-creator/-critic`,
background spawn + TaskOutput wait). Revise rounds spawn a **fresh** creator
whose prompt says "continue, don't restart" (all state is on disk) — exactly
like the script. Do not rely on SendMessage continuation of a finished creator:
custom-agent transcripts are often not resumable ("No transcript found");
treat a successful resume as a lucky optimization, never as the plan.

After the build finishes (either path):
1. Persist critiques: write each round's issues to
   `<target-dir>/workflow/critique-round-<n>.md` if the creator didn't already.
2. **Restore violations** reported by the verification stages: for each tracked
   file changed outside the target dir, `git checkout -- <path>`; move
   untracked strays to your scratchpad (never silently delete). Record them
   for the final report.
3. **Trust but verify the render**: if `renderFresh` came back false (or
   `check/` page mtimes predate the `.typ`), bounce it back for one rebuild
   before proceeding.

## Phase 6 — user review

Skip in `--no-user` mode.

Present: the PDF path, a 3–6 line tour, outstanding critic issues (if the loop
ended on REVISE), remaining `[TODO verify]` markers. Then: *"Review and give me
comments — I'll forward them to the creator. Say 'done' when satisfied."*

Each batch of user comments = one more creator round, exactly like a critic
round (prefix: `User review round <k>. These override any critic opinion.`).
Run it via SendMessage to the existing creator, or one more workflow/agent
round if that instance is gone. Verify the tree after each round. No timeout
here — wait for the user.

## Phase 7 — finalize + friction report

1. Disarm: delete `.claude/writeup/target.json`; kill any `typst watch`
   background task.
2. Final message to the user (or return value to the calling agent):
   - artifact path + one-line description
   - rounds run, verdicts, violations caught (if any)
   - **creator friction report**: the collected `creator-log.md` content
     (BLOCKED entries + workarounds) verbatim enough to act on, plus your own
     observations on where the workflow dragged — this is the input for tuning
     the workflow; don't sand it down
   - remaining `[TODO verify]` markers
   - do NOT commit; suggest a commit message the user can use.

## Failure modes to guard

- **Creator claims success without rendering** → the collect stage checks
  `check/` freshness; bounce back once before accepting.
- **Critic dies / malformed verdict** → the workflow schema-validates it; in
  the inline fallback, re-prompt the same round once, then extract issues
  yourself and continue.
- **typst @preview packages missing offline** → pre-warm by compiling any
  existing artifact once yourself before Phase 5.
- **Jail architecture** (context for debugging): frontmatter hook on
  `writeup-creator` enforces always; a global settings.json hook additionally
  jails *every* subagent while `target.json` exists (main thread exempt) —
  covers workflow-spawned agents even if frontmatter hooks don't fire there.
  Registered in `.claude/settings.json`; scope logic in `writeup-guard.py`.
