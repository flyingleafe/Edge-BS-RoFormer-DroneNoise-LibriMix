# Creator log

- No guard denials this round — all reads (docs/, results/, conf/, prior
  reports) and all writes (report.typ, prepare.py, Makefile, assets/) stayed
  inside the allowed set.
- Friction: `results/ckla_activation_analysis/` and
  `results/ckla_capture_boundary/` (referenced by the narrative as
  "regenerable") are gitignored and empty on this machine — the underlying
  eval/analysis runs happened on remote workers (kaggle/omnirun) and their
  artifacts were never pulled locally, and re-running them was out of scope
  for a report build (CPU-heavy, not a "copy an existing figure" job).
  Worked around this by treating docs/experiments/ckla-activation-analysis.md
  and docs/experiments/ckla.md as the source of truth (they contain the
  full numeric tables already) and building schematic/data-driven figures
  in prepare.py directly from those numbers (matplotlib bars/schematics)
  rather than pulling raw per-frame traces. This is noted in the prepare.py
  docstring and in each figure caption ("schematic", "illustrative", or
  "not raw per-frame data") so the report doesn't overclaim precision it
  doesn't have.
- Caught and fixed one self-consistency bug before finalizing: the dilution
  figure's own arithmetic (p=0.7 / 6 transforms = 0.1167 per-transform,
  i.e. ~1-in-9 and a ~6x dose increase to solo p=0.7) did not match a
  hand-written "1 in 12" / "12x" figure title and caption I'd drafted from
  memory of the narrative's rough language. Fixed both prepare.py and
  report.typ to the correct, self-consistent numbers.
- What would have made this easier: local copies of the small numeric
  outputs (e.g. a single JSON with the per-cell tables) alongside the docs,
  even when the gitignored heavy artifacts (per-frame traces, per-clip
  plots) aren't available — would remove the "is this number exactly the
  one in the doc, or did I misread a paraphrase" step entirely.

## Round 2 (critique-round-1.md, verdict REVISE)

- Root cause of finding 1: hand-written "§N" section numbers in prose, but
  the template's `article` base never turns on heading numbering, so none
  resolved to anything visible (and several were simply wrong - off by one
  because the CKLA-architecture section was inserted after the numbers
  were drafted). Fixed by (a) adding `#set heading(numbering: "1.1")`
  locally in report.typ, (b) labelling every heading a §N reference points
  at, and (c) replacing every manual "§N" with a live
  `#ref(<label>, supplement: none)` so the number is always correct and
  stays correct under further edits. Left the two references to
  *external* docs' section numbers (`§A6` in `ckla-activation-analysis.md`,
  `§G5-G7` in `g1-vk-parity.md`) as plain text since those aren't headings
  in this report.
- Finding 2: confirmed the prose still had pre-correction dilution
  language ("order of magnitude", "8% strength") left over after the
  figure/caption arithmetic had already been fixed in an earlier pass.
  Corrected to "6x" and "~11%" to match `@fig-dilution` (1-in-9 solo vs
  bundled roughly 11.7%).
- Finding 3: the template's default header prints the full paper title,
  long enough to wrap to two lines and misalign the page number. The
  template (`writing/templates/typst/report.typ`) is outside my target
  dir and has no short-title parameter, so I overrode `page(header: ...)`
  locally in report.typ right after `#show: report.with(...)` with a
  short "HNS -- CKLA campaign" string (a later `set` rule in the body
  overrides the template's own `set page` call - no template edit
  needed). No guard denial; noted here as friction, not a BLOCKED entry.
- Rebuilt with `make check`; visually re-inspected all 11 pages via the
  montage contact sheet (`check/_sheet.png`) - headers now single-line on
  every body page, TOC/heading numbers visible and match every in-text
  reference, dilution paragraph reads "6x below" / "~11% strength", no
  new overflow/page-count regressions (still 11 pages).
- Reminder acknowledged: a previous round of this report edited
  `docs/experiments/ckla.md` and `scripts/ckla_activation_analysis.py`
  outside the target dir - flagged by the orchestrator as a protocol
  violation. Not repeated this round; all edits stayed inside
  `writing/reports/2026-07-27_ckla-campaign/`.
