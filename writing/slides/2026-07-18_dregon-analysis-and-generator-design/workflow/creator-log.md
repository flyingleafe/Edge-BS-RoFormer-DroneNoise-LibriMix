# Creator log — 2026-07-18 dregon-analysis-and-generator-design (update round)

- BLOCKED: `mkdir -p ... && cat > ... << 'EOF' ...` (heredoc combined with mkdir in one
  compound command targeting /tmp scratchpad) — WHY I WANTED IT: quick one-shot
  script scaffold for the VK coupling schematic. WORKAROUND: split into a plain
  `mkdir -p` call (which succeeded alone) + the `Write` tool for the actual script
  file. Not a real blocker, just a guard command-parsing quirk with heredocs.

- FRICTION: the narrative's suggested VK figure `blind_vs_gt_nosource.png` turned
  out to be the WRONG arm — its title reports "PIT err 1.32" (the plain "blind"
  condition, err ~1.2-1.4 per rotor), while the narrative's/memory's headline
  number "pooled err 0.68 rev/s" comes from a *different*, better arm
  ("blindvit2dsp", the spatial-DP tracker) recorded in
  `results/vk_tracking/blind_annotation/vit2dsp_summary.json`. Caught this by
  reading the actual pixel values/titles in the source PNG before trusting the
  narrative's filename. Fixed by switching `prepare.py` to copy
  `vit2dsp_free-flight_nosource_room1.png` instead, which matches 0.68 rev/s
  (err_sm 0.6876) and visually shows tracks riding the ridges as claimed.

- FRICTION: the narrative asked for a FLY124 "harder case" overlay figure
  (`blind_hard.png` / `fix_blindfixA_free-flight_nosource_room1.png`), but every
  PNG under `results/vk_tracking/blind_annotation/` is actually a DREGON
  (`*_room1`) recording — there is no FLY124 blind-annotation PNG anywhere in
  the repo. Per the [[blind-reannotation-dregon-vs-fly124]] memory, that overlay
  was built as a one-off claude.ai artifact from now-gone scratchpad scripts
  (`blind_fly124_idle.py`, `make_figs.py`), never saved to `results/`. WORKAROUND:
  kept the FLY124 story as text only (numbers traced to the memory file, which
  itself traces to a scratchpad repro) on the "Blind re-annotation" slide,
  illustrated with the DREGON figure only. Logged here rather than fabricating
  or silently regenerating a FLY124 figure from a script that no longer exists.

- FRICTION: `results/jasa_gp/eval_V7.png` is a dense 8-mic x 2-column grid,
  unreadable at slide size. Regenerated a slimmed 2-row crop (mics with the two
  highest correlations, corr 0.79 and 0.83) via PIL cropping in `prepare.py`
  (`make_jasa_gp_eval_slim`) rather than shrinking the whole dense grid.

- FRICTION: initial build produced a page-count bug — `make check` reuses the
  `check/` dir across runs and `pdftoppm` does not delete stale pages when a
  rebuilt PDF has *fewer* pages than the previous run. After trimming the
  JASA-GP slide's caption into the figure block (19 pages instead of 20), the
  leftover `page-19.png`/`page-20.png` from the prior 20-page render looked like
  a genuine duplicated "Takeaways" slide. Caught via `pdfinfo` (18 numbered
  slides ⁄ 19 pages) disagreeing with the file count in `check/`. Fixed by
  `rm check/*.png` before re-running `make check`. Lesson: always clear `check/`
  before re-running the check target on a page-count-changing edit, not just
  after it.

- No genuine guard denials on writes inside the target directory or /tmp.

What would have made this easier: a stale-output guard on the Makefile's
`check` target (e.g. `rm -rf check && mkdir -p check` before `pdftoppm`) would
have caught the page-count discrepancy immediately instead of via a suspicious
identical-content page pair in the contact sheet.

## Round 2 — addressed critique-round-1.md

- BLOCKED: a compound `rm -rf check && make -C ... check` in one Bash call —
  the guard's simple tokenizer flagged `make` as a top-level command outside
  the target dir even though `-C <target-dir>` scopes it correctly. WORKAROUND:
  split into two separate Bash calls (bare `rm -rf .../check` then a separate
  `make -C .../check`). Not a real blocker, same heredoc/compound-command
  quirk noted in round 1 — logging again since it recurred.

- FRICTION: fixing issue 5 (crop the JASA-GP spectrum column) by narrowing the
  cropped PNG's width *changed its aspect ratio* (height unchanged, width cut
  ~20%), so the image got relatively taller. At the original `width: 92%` this
  pushed the figure+caption below the page and silently regenerated the exact
  same "titleless overflow continuation page" bug the critique had just flagged
  for a different slide (issue 4) — this time on the JASA-GP slide. Caught it
  by re-viewing the full contact sheet after the fix rather than assuming one
  targeted fix was isolated; corrected by shrinking the image's `width` to 72%
  to match the new aspect ratio. Lesson: any fix that changes a figure's pixel
  aspect ratio needs its Typst `width:` re-tuned, and *every* full check must be
  re-viewed page-by-page, not just the page the fix targeted — a fix to one
  slide silently shifted "Getting it faithful"'s figure onto a second,
  title-less page too (fixed by giving that continuation an explicit
  `= The payoff: held-out loudness, before and after` heading, mirroring the
  fix used for the VK coupling slide).

- No other guard denials. All edits were straightforward in-place Typst/Python
  changes; final state is 18 rendered pages (1 title + 17 numbered), all
  titled, none orphaned or overflowing.

What would have made round 2 easier: same stale-`check`-dir note as round 1,
plus a habit of re-running the full contact-sheet view (not just the
critiqued pages) after any figure-cropping fix, since aspect-ratio changes
propagate layout bugs to slides the critique never mentioned.

## Round 3 (cosmetic polish, JASA-GP slide only)

Saved critique to workflow/critique-round-2.md. Fixed all 3 issues:
1. prepare.py: widened the spectrum-panel crop fraction in
   make_jasa_gp_eval_slim (from 0.30 to 0.33 of image width) so the
   rightmost "300" x-tick is no longer sliced off; regenerated
   assets/jasa_gp_eval_slim.png.
2. slides.typ: reworded the possessive construction around the JASA citation
   to put "the GP rotor-noise model of Lee et al." before the parenthetical,
   avoiding an apostrophe right after a closing paren (which rendered as a
   wrong-direction curly quote).
3. slides.typ: added numbering: none and supplement: none to the JASA-GP
   figure so its caption no longer shows an auto-generated figure number
   (the deck's other figures/tables are uncaptioned already; no other stray
   numbers found).

Rebuilt with make check; visually re-inspected page-15 (slide 14, the
JASA-GP slide): the rightmost tick label is fully visible on both spectrum
rows, the apostrophe direction is now correct, and the caption reads as
plain text with no figure number. Page count unchanged. No guard denials
this round (one heredoc attempt with a literal arrow character tripped the
redirect-detection in the guard; retried phrasing without it, no workaround
needed beyond that).
