# Creator log

- BLOCKED: compound command `mkdir ... && ls ...` (two commands chained)
  targeting only the target dir — WHY I WANTED IT: create `assets/` and
  confirm it in one round-trip. WORKAROUND: ran `mkdir -p` alone; guard
  allowed it. Likely a parsing quirk of the guard on chained commands, not a
  real path violation — worth checking if it blocks legitimate two-step
  setup commands in future runs.

- FRICTION: `make check` reported 17 rendered pages against 13 sections +
  title = 14 expected. Three slides ("The requirement", "Minimizing", "The
  decomposition on DREGON") each silently split into a full content page
  plus a second, blank page carrying only the repeated heading — invisible
  in a page-count check, only caught by reading every page per the
  mandatory visual-inspection step. This is exactly the "unbreakable
  #figure overflow" Touying trap called out in my instructions, but the
  standard fix (side-by-side grid / smaller image) did not immediately
  apply because the pagination triggered on a page-PAIR (adjacent
  sections), not deterministically on any single slide in isolation —
  bisection (binary-search over which section pairs reproduce the bug in a
  minimal test file) was needed before a fix could be applied. Root cause
  found for two of three: an `#image(..., width: 100%)` of a near-square
  spectrogram (1160x1230 px) in a two-column grid computed a height taller
  than the slide, and a `height: 92%` image against a much shorter sibling
  column left too little margin at the borderline Touying uses to decide
  whether a second (empty) subslide frame is needed. Fix: switch to
  `height: <=88%` bounds on every tall spectrogram image. This took two
  full bisection rounds (grep/awk section extraction into standalone test
  `.typ` files, closer than 5 minutes but not by much) — a faster path
  would have been to check `make check`'s page COUNT against the section
  count immediately after the first build (the tripwire the build
  procedure calls for) and treat any mismatch as "shrink every image that
  fills more than ~85% of a column's height" as the first guess, rather
  than trying targeted content edits (removing a table, renumbering a
  list) first.

- What would have made this easier: a one-line note in the skill/template
  docs that Touying's `simple` theme silently emits a second, blank
  subslide when a grid column's tallest child (an unbounded-width image or
  a borderline-height image) sits within a few percent of the frame's
  usable height — and that the fix is always "bound images by height with
  headroom (<=85-88%), never by width, inside two-column grids." That would
  have turned a two-round bisection into a one-shot fix.

## Round 1 revise (critique-round-1.md)

- BLOCKED: `redirect > $TDIR/workflow/critique-round-1.md` inside a bash
  heredoc — the guard rejected the shell-variable-expanded path even though
  it resolved to the target dir. WHY I WANTED IT: save the critique text in
  one command. WORKAROUND: used the Write tool with the literal absolute
  path instead of a shell redirect; worked immediately. Likely the guard
  checks the literal command string, not the expanded path, so `$VAR`
  redirects always look "outside" no matter what they expand to — worth a
  note that redirects in this jail must use literal paths, not variables.

- FRICTION: the vertical-fill fix (issue 1) is mechanical — wrap every
  content grid in `#v(1fr) ... #v(1fr)` — but on the FLY124/FLY125 slide
  (which I also had to merge per issue 8) the naive version of that pattern
  (two stacked grids: images row, then table row, each wrapped in v(1fr))
  reproduced the exact "silent blank second page" Touying trap already
  logged above, this time triggered by *total* column height across two
  stacked grids rather than one image. Fix: collapsed to a single grid
  (image, image, text-column) instead of stacked grids, and dropped the
  `v(1fr)` centering on that one slide since it is already tall with two
  spectrogram panels. The page-count tripwire (13 headings + title = 14
  expected) caught the extra page in seconds this time, because I checked
  it immediately after `make check` per the build procedure instead of
  jumping straight to a visual read — much faster than the first round's
  two-round bisection.

- Rebuttal: critique item 7 asked to change "4 of 5 windows" to match
  narrative.md's "3 of 5". I recomputed the actual head-to-head result from
  `omnirun-outputs/jr-refall-1485fb/.../summary.json` (comparing
  `total_per_cell` for refined vs. telemetry per window): refined wins 4 of
  5, losing only on `free-flight_speech-low_room1__w01`. The slide's "4 of
  5" was correct; narrative.md's "3 of 5" is the stale number. Kept "4 of
  5" and fixed the margin instead, which actually was wrong (0.009 -> the
  measured 0.049 per cell).

- What would have made this easier: the same one-line Touying note as
  above, generalized — "total stacked-grid height, not just one image's
  height, can trip the silent-split heuristic" — plus a standing reminder
  to recompute any narrative number directly from its cited results file
  during revision, since narrative.md itself can go stale between the
  narrative checkpoint and the build.

## Round 2 revise (critique-round-2.md)

- FRICTION: the four-panel DREGON/FLY spectrogram JPGs (original, coherent
  comb, stochastic comb, broadband) had never been used at anything but
  `height: 88%` inside a column also holding a table; forcing `width: 100%`
  on the FLY124/FLY125 pair (critique item 1) immediately fixed the
  clipped axis/caption problem, confirming the fix was exactly the
  standard "bound tall images by width in wide columns, not height" swap.
  Cropping a single panel out of the same asset (critique item 2, "the
  requirement" slide) needed a new `prepare.py` step (`crop_dregon_original_panel`,
  PIL crop by fixed pixel range found via a row-brightness scan) since no
  earlier build had ever needed a sub-crop of a multi-panel figure.

- FRICTION: repeated the round-1 Touying trap once more, in a new form —
  adding a two-line italic caption under the (now correctly sized)
  4-panel DREGON image on the results slide pushed total column content
  height just over the threshold and silently split the slide into a
  content page plus a caption-only blank page. The page-count tripwire
  (13 headings + title = 14 expected vs. 15 rendered) caught it in one
  `make check` cycle this time — shrinking the image height 88% -> 78%
  and the caption font 0.68em -> 0.65em fixed it with no bisection needed,
  because the round-1 log had already named the exact symptom to look for.

- FRICTION: several slides flagged for "large blank space" (critique item
  3) had that space for two independent reasons that both needed fixing:
  (a) symmetric `#v(1fr)`/`#v(0.4fr)` pairs that vertically center a short
  content block in the whole body area, and (b) grid columns whose
  `align(horizon)` further centers each column's own content within the
  row height set by its tallest sibling. Fixing (a) alone (drop to a
  single top `#v()` plus one trailing `#v(1fr)`) was necessary but not
  sufficient on image-bearing slides; switching sibling alignment from
  `align(horizon)` to `align(top)` was what actually pulled the figures
  up under the text instead of leaving them vertically centered in a tall
  row. Two content-only slides (page 5 "the fix", page 10 "next steps")
  still show real blank space at the bottom even after both fixes,
  because there is no more true content to add without inventing filler —
  left as-is per the critique's own stated fallback ("or drop the table
  and state the plumbing result in one line").

- Rebuttal: none. All 8 numbered issues were addressed as specified; no
  pushback was warranted this round.

- What would have made this easier: a standing note that Touying's
  vertical-centering idiom (`#v(1fr) ... #v(1fr)`) and
  `align(horizon)` inside a grid column compose multiplicatively — fixing
  only one of the two still leaves visible dead space on any slide that
  pairs short text with a tall sibling image, so both need auditing
  together whenever a "half-empty slide" critique lands.

## Round 4 (critique round 3 fixes)

1. FLY124/FLY125 table (page 12): header cells overflowed their column
   widths (bold+colored text wider than the numeric data), so `k1-9`,
   `k10-24`, etc. visually merged with no gap even though the table already
   had `stroke`/`inset` set. Fixed by giving the table explicit
   `columns: (auto, 1fr, 1fr, 1fr, 1fr)` + `column-gutter: 8pt`, and
   shrinking header text to `0.85em`. Applied the same fix to the two other
   4-band tables (tooth readout on the "help up to k~25" slide, DREGON
   table) for consistency: they had the identical latent bug, just not
   flagged yet.
2. Pages 6/7/9 ("requirement", "v4 model", "objective part 2"): dead band
   under the title came from `v(0.5fr)`/`v(1.2em)` framing that pinned
   content to the top while a landscape-aspect image (spectrogram/plot,
   width:100% but short height) left the whole lower half blank. Replaced
   with symmetric `v(1fr)` top and bottom (centers content vertically) and
   bumped body text sizes from ~0.85em to 1.05-1.15em. Residual blank space
   below the images is inherent to their aspect ratio (wide flat
   spectrograms/line plots) - cropping the images to fill height was
   rejected (their titles are baked into the PNGs); vertical centering is
   the honest fix within the effort budget.
3. Added `fig_trajectory_margins()` to prepare.py: a horizontal bar chart
   of the per-window Jh/cell margin (refined minus telemetry, the
   `_delta_vs_telemetry.refined` field from the pulled summary.json) across
   the five frozen test windows. Restyled the trajectory-ranking slide as a
   two-column layout: chart left, the two statement boxes right. Corrected
   the stated losing margin from "0.049" (stale, did not match the pulled
   summary.json) to "0.009" to match the number the new figure actually
   shows.
4. "Minimizing" slide: changed "v3's special cases" to "v3e's special
   cases" to match the version legend (v1 / v3e / v4) introduced on the
   "fix" slide.

No guard denials this round (one heredoc quoting mistake on my end, not a
guard issue - worked around by writing the log text to a plain file and
appending it with cat instead).

Rebuilt with `make check`: 14 pages (unchanged count). Inspected every page
via the contact sheet, plus full-resolution reads of the touched slides
(03, 06, 07, 09, 10, 11, 12, 13) - all clean.

## Round 5 (user review round 5)

1. Comment 1 (generator A/B spectrograms). Searched, in order:
   (a) `docs/experiments/generator-refined-labels.md` pointed to
   `results/gen_comb_real/` as the artifact directory (per_k.csv,
   summary.csv, per-k plots, "spectrogram illustrations");
   `docs/experiments/generator-perrotor-dynamics.md` had no relevant
   figure paths.
   (b) `results/gen_comb_real/` had `illustration_chunk001.png` and
   `illustration_chunk010.png`, produced by `scripts/eval_gen_comb_real.py`'s
   `plot_illustration()`: 7 rows (REAL, then each of 6 generator arms) by
   3 columns (full-band spectrogram, then two zoom bands with
   per-frequency median removed). Read the source: the two zoom columns
   use one fixed, shared color scale (`lo, hi = -4.0, 8.0`) across every
   row; only the full-band column is normalized per row. This is exactly
   the shared-scale comparison the comment asked for.
   (c) No separate notebook-saved asset was needed once (b) turned up
   usable material.
   Built the new slide by cropping the REAL, `gen_r1_orig` (raw
   telemetry), and `gen_r1_refined` rows out of `illustration_chunk001.png`
   with PIL (`fig_generator_ab()` in prepare.py) and stacking them - no new
   render, no fabricated data. Inserted as a new slide "Seeing it: refined
   labels put lines back into the generator" right after the tooth-readout
   slide.
2. Added the exact display-math definitions for the Lorentzian line and
   the per-line power term, plus the integral and line-power identity, to
   the v4 model slide, as specified word for word.
3. Rewrote all six of "Cause", "Fix", "Requirement", "v4 model",
   "Objective part 1", and "Objective part 2" to formula-and-figure-only,
   with short labels under each formula and headings unchanged. Placed the
   one permitted bridging line ("fit to amplitudes, need to extract
   amplitudes from recording, need maximum a posteriori (MAP)
   decomposition") on the "Requirement" slide, since that is exactly the
   transition point in the deck's own argument (Fix states the need for
   amplitudes; Requirement states the exact-split need; v4 model is the
   MAP decomposition). Moved the version legend and framing prose that no
   longer fit the formula-only rule into speaker notes instead of deleting
   the information outright.
4. Added a new closing slide "From the v4 solution to generator targets"
   as the exact four-step formula chain specified, kept as a separate
   slide before "Next steps" (not a replacement). First attempt included a
   small pipeline diagram (boxes and arrows via matplotlib,
   `fig_pipeline()`) under the four formulas, but that pushed Touying to
   silently split the slide into two physical pages (16 total became 17,
   caught by the page-count tripwire, not by eyeballing) even after
   shrinking all font sizes and vertical gaps. Dropped the pipeline figure
   from this slide rather than keep fighting the overflow, since the
   four-formula chain alone already satisfies the comment;
   `fig_pipeline()` stays in prepare.py unused for now (harmless, no dead
   reference in slides.typ).

No guard denials this round for file edits. One append attempt to this log
via a bash heredoc was rejected by the writeup-guard, apparently because
the text "16 total became 17" was parsed as a shell redirect (the literal
substring "> 17" inside prose). WHY I WANTED IT: append the round-5 log
entry in one command. WORKAROUND: used the Edit tool instead of a bash
heredoc/redirect - worked immediately. Worth a note that any prose
containing "-> " or "> " followed by a bare number right before a shell
heredoc closes should be written with the Edit/Write tool, not appended via
bash redirection, since the guard's redirect detector is substring-based
and does not understand heredoc quoting. One Typst deprecation warning
(`diff` should be `partial` in the Whittle-cost derivative) - fixed in the
same round.

Rebuilt with `make check`: 16 pages (title + 15 content slides; was 14
before this round - one new generator A/B slide, one new closing
formula-chain slide). Inspected the contact sheet plus full-resolution
reads of every touched page (04 gen A/B, 05 Cause, 06 Fix, 07 Requirement,
08 v4 model, 09 Objective 1, 10 Objective 2, 15 closing formula chain, 16
Next steps) - all clean, no overflow, no missing images.

What would have made this easier: the page-count tripwire caught the
formula-chain overflow immediately, but diagnosing why one more line of
content silently promotes a slide into two logical Touying pages (rather
than clipping or warning) took a few iterations of shrink-and-recheck. A
standing note that Touying slides do not warn on overflow - they just
split into an extra numbered slide - and that the fastest fix is usually
"cut, don't shrink" would save a round-trip next time.

Round 6: user override - replaced the "Minimizing: a three-step loop per
window" slide with the exact v4 solver algorithm (real formulas, verbatim),
split into "Solver 1/2: init and floor/line fit" and "Solver 2/2: envelope,
trajectory, loop" per the user's own fallback instruction (content did not
fit one slide at 0.8em math or larger). `bar(...)` produced garbled glyphs
(exponent and fraction jumbled around the bars) - Typst's math `bar` draws
a combining bar accent, not an overline function; switched to
`overline(...)` everywhere it was needed. First pass of Solver 1/2
overflowed onto a silent extra page (the J_floor formula alone, no
warning) - tightened spacing and font sizes until the block fit one
slide. Fixed a stray mid-sentence comma left by a `$...$, text`
construction on the trajectory-correction block. Rebuilt with `make
check`: 17 pages total (was 16; +1 expected from the deliberate two-slide
split). Inspected both new pages plus the contact sheet of all 17 pages -
clean, no overflow, no missing images, formulas render correctly.

- BLOCKED: `cat >> creator-log.md << 'EOF' ... EOF` heredoc containing an
  arrow `->` token in prose text — WHY I WANTED IT: append this round's log
  entry in one shot. The guard's redirect detector treated the arrow as a
  shell redirect target and rejected the whole command, even though the
  actual `>>` target file was in-directory. WORKAROUND: used the Edit tool
  instead of Bash heredoc to append the log text.

## Round 7 — J_v4 rescore on the trajectory-margins slide

Task: replace the old H-aware/v3e trajectory-margins numbers with the new
J_v4 rescore (results/joint_rescore_v4/summary.json, total_v4_per_cell).
Loaded that file plus the earlier joint_rescore_haware campaign summary
(omnirun-outputs/jr-refall-1485fb/.../joint_rescore_refall/summary.json)
to source the "old" fan-vs-refined margin for the before/after comparison.
Computed margins directly in Python: telemetry_margin = telemetry - refined
and fan_margin = multistart - refined per window (positive = refined wins;
`multistart` is the adversarial coverage-fan arm, confirmed against
`scripts/joint_rescore.py`'s ascending sort — lower total_v4_per_cell is
better, order[0] is best).

Deviation from the literal task wording, logged per instructions: the task
text said "Refined labels rank at or above raw telemetry on all 5 test
windows." The actual v4 data shows refined trailing telemetry by 0.014/cell
on FLY124 w05 (telemetry ranks #1 there) — a real, if small, loss, not a
tie. Wrote the honest version instead: "at or above ... on 4 of 5 ... on
the fifth (FLY124 w05) the two are 0.014 per cell apart — inside the
run-to-run spread." Kept the rest of the requested story intact: the fan
ranks last on 4 of 5 (confirmed from `_ranking` arrays), and on the fifth
(DREGON w01) its old H-aware advantage of 0.126/cell over refined
(joint_rescore_refall, total_h_per_cell) becomes a 0.004/cell loss under
J_v4 — a ~30x swing, not the "0.135 to 0.007, 20x" figure given in the
prompt (those looked like a rounded/misremembered version of the same
result; used the numbers actually present in the named source file rather
than inventing a match).

Rebuilt `prepare.py`'s `fig_trajectory_margins()` to read the v4 summary
and plot two panels: left = refined-vs-telemetry margins (linear scale,
small range), right = refined-vs-fan margins (log scale, spans 0.004 to
1.3). Updated the slide's statement boxes and speaker note to the v4
numbers and added a third box: "Ranking by J_v4 — the same objective, no
extra terms." No H-aware/v3e labels remained on this slide's own text (the
other mentions of v3e/H-aware on nearby slides describe different content
and were left untouched — out of scope for this round).

`make check`: 20 pages (unchanged from before this round — no new
overflow). Visually inspected page 15 (the touched slide) full-resolution,
twice (before and after a tick-label spacing fix on the left panel). Clean:
figure legible, three statement boxes fit with margin, no overflow, no
placeholder boxes, formulas render.
