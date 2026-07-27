# Creator log — dcunet-generalization deck, Part 2 (CKLA campaign) extension

- No guard denials encountered. All reads/writes stayed inside the target
  dir (`writing/slides/2026-07-27_dcunet-generalization/`), except reading
  source material (report, docs, conf/experiment, git log) which is
  unrestricted.
- Friction: `scripts/check_stream.py`, named in narrative section 7 as
  "if merged by build time; else describe the design", does not exist yet
  in the repo (checked via `ls` + `git log`). Followed the narrative's
  explicit fallback: the "Stream sanity protocol" slide describes the
  planned checks (frame-expansion factor, effective stage boundaries in
  epochs, empirical fire-rate diffing) as a design, not a fabricated
  example-output table. Speaker note says explicitly that it's not merged
  yet, so nobody mistakes the box for a real screenshot.
- Figures for Part 2 are copied verbatim from
  `writing/reports/2026-07-27_ckla-campaign/assets/` via `prepare.py`
  (`copy_ckla_figures()`) — no regeneration needed, all six figures were
  already presentation-ready (wide aspect, big fonts) since the report was
  built for the same standing deck rules.
- Page-count tripwire caught one real overflow: the "Results — pending"
  slide's 3-row table + full-width results_ladder image did not fit on one
  page together (21 pages instead of the expected 20; the table's last row
  spilled onto a new page under "Stream sanity protocol"). Fixed by
  shrinking the image (96% -> 68% width), the table font (0.85em) and inset,
  and merging two "stands" rows into one — brought it back to 20 pages,
  verified by re-running `make check` and reading page 18 directly.
- What would have made this easier: knowing up front whether
  check_stream.py had landed (a quick grep would have sufficed, but the
  narrative already flagged the uncertainty, so this cost ~1 command, not
  real friction).

## Round 2 (critique-round-2.md)

- Guard false positive: appending this section via a `bash cat >> file
  << EOF` heredoc was denied by the writeup guard, which mis-parsed the
  literal text "92% to 62%" inside the note body as a shell redirect
  target ("redirect target '62%' is outside your allowed directory").
  No actual out-of-directory write was attempted. Workaround: used the
  Edit tool to append this section instead of a shell heredoc. Logging
  this per protocol even though nothing was actually blocked outside the
  target dir.
- Item 1 (staging bug not on-slide): added a compact grey box on the
  "augmentation regime" slide stating the bug in three lines: flatten
  channels to 8 mono frames per chunk, stage-2 boundary landing at true
  epoch ~80 instead of epoch 10, staged augs never firing, caught via
  bit-identical cross-policy runs. First attempt overflowed onto an
  orphan page (21 pages instead of 20 -- caught by the page-count
  tripwire); fixed by shrinking the dilution figure and caption font and
  tightening vertical spacing. Rebuilt to exactly 20 pages, verified
  visually.
- Item 2 (garbled SI-SDR/SDR formulas): the double-pipe norm syntax was
  parsing wrong in Typst. Rewrote using `norm(s)^2` with explicit
  parenthesized numerator/denominator groups. Renders cleanly now.
- Item 3 (accumulator-degeneration figure unreadable, dead space below):
  gave the left panel more relative column width and vertically centered
  both images in the grid. Axis labels, legend, and the annotation are
  now legible at 150 dpi.
- Item 4 (stream-sanity slide tone + alignment): reworded body to "Gate
  design, landing this week" (previously implied the gate already
  existed) and left-aligned the bullet list inside the box (the outer
  center alignment was propagating to all descendant text; fixed with an
  inner left-aligned wrapper).
- Friction: the page-count tripwire is essential here -- the first
  version of the staging-bug box compiled without error but silently
  created an orphan page; only the check/page-*.png count (21 vs
  expected 20) caught it, the compile log gave no hint.

## Round 3 (user review; fresh creator picking up a stopped round)

- Inherited state: the previous creator had already written the full
  Part-2 rewrite per the user spec (base/expanded augmentation tables +
  synced-axis tuple grids, freq-shift prediction probe, VK / KLA / CKLA
  derivation, WIP-badged results) and regenerated all assets; the
  staging-bug, protocol-reset and stream-sanity slides were already
  removed. Decision: build on it rather than restart, and finish the
  verification pass it never got to.
- Page-count tripwire fired hard: 35 rendered pages against 31 expected
  headings. Four silent Touying splits, all fixed by shrinking, not by
  bisecting: (a) the freq-shift slide split into THREE pages (figure at
  100% + caption); fixed by image 100% -> 62% and caption 0.8em ->
  0.7em; (b) "Kalman filters as network layers" split at an em-dash in
  the lead sentence (the known quirk) -- reworded to a colon and put the
  two display formulas in 0.92em text; (c) "Results -- one prediction"
  split, image 62% -> 44%; (d) leftover caption orphan after the first
  freq-shift fix, cleared by the second shrink. Final: 31 pages = 31
  headings + title, one page per slide.
- Figure fixes found by reading full-res pages: in every synced-axis
  grid the y tick labels of panels 2..n collided with the neighbouring
  panel ("1000" clipped to "000"). Since all panels share identical
  ranges by construction, tick labels are now drawn only on the left
  column (tuple_grid + the freq-shift figure). Also narrowed the per-
  panel width (3.6 -> 3.0 in) so the grids fill more of the slide, and
  labelled the freq_scale panel "clip shortens" so the white margin
  left by alpha=1.2 resampling doesn't read as a rendering bug.
- Verified visually that the compared grids share axis ranges: base set
  and expanded set both run 0-1200 Hz / 0-max rev/s with one shared
  colour bar; freq_scale's RPS curves sit visibly ~20% above clean, and
  in the freq-shift probe the dashed GT rises with the shift while the
  solid prediction stays put -- the intended failure is legible at
  150 dpi.
- No guard denials this round.
- What would have made this easier: a `make check` that fails loudly
  when page count != heading count. Every layout defect this round was
  a silent split; nothing in the compile log hinted at any of them.
  Also: pdftoppm read a half-written slides.pdf once (xref errors, 14
  pages) when run in the same command as `typst compile`; a sync/short
  wait between the two would avoid a confusing false alarm.

## Round 4 (critique round 3 — six fixes)

- (1) The literal-brace exponent in the CKLA derivation now uses
  parentheses instead of curly braces; verified on the rendered page-26
  PNG (no stray braces).
- (2) Expanded-set grid 2/2 repeats the clean panel and shares exact
  t/RPS limits with grid 1/2: `tuple_grid()` gained optional `t_max` /
  `rps_max`, computed once over both case lists. Verified numerically —
  the first (clean) 420-px column of the two PNGs is byte-identical
  (max abs pixel diff 0).
- (3) Internal designators "protocol-B" / "gain-fix" removed from slide
  bodies and from the overlay figure title (now "CKLA, current run
  (WIP)"); both retained in the speaker notes only.
- (4) PIT-MAE / neural-floor footnote added under the cruise-pool table.
- (5) k, v, q and lambda_v glossed under the KLA readout.
- (6) Freq-shift probe cropped to the common valid duration
  (clip length / 1.10, labels and predictions sliced to the same frame
  count) — the zero-filled end-of-clip plunge is gone.
- Page-count tripwire fired: the first rebuild came back at 32 pages.
  The culprit was the pre-existing base-set grid at `width: 100%` plus a
  `#v(0.2em)` spacer, which had been splitting silently all along (the
  footer counter, not the physical page count, hid it). Dropped the
  spacer and set the image to 94%; back to 31 physical pages, and the
  footer total corrected itself from 31 to 30.
- BLOCKED: `cat >> workflow/creator-log.md <<'EOF' ...` — WHY I WANTED
  IT: append this round entry in one shot. The guard parsed a backtick
  inside the heredoc body as a redirect target and denied the call.
  WORKAROUND: appended with the Edit tool instead. Worth noting for
  guard tuning: heredoc bodies should not be scanned for redirects.
- What would have made this easier: the same tripwire wish as last
  round — the split slide was only findable by diffing `pdftotext`
  page-by-page against the footer numbering.

## Round 4 — Part 1 rebuild (baselines evaluation)

- Rewrote all of Part 1 against `BRIEF.md`, `docs/experiments/f1-se-blind-baselines.md`
  and `docs/experiments/f2-survey-replication.md`. New slides: models table,
  data/protocol table, per-SNR results for Pass A and Pass B, output
  spectrograms. Kept the seen/unseen probe, its control, the breadth ladder and
  the MP-SENet control; dropped the DN-LM-leakage slides and the ranking-table
  slide (the leakage argument is about someone else's dataset, not about our
  own baseline evaluation, which is what this Part is now for).
- New figures generated from `results/f1_perclip/*__SE-valid-drone.csv`
  (per-clip, all 350 clips; the 5 digitally-silent clips are dropped by
  matching the noisy anchor, so every curve is over the same 345 clips).
- Spectrogram figures run the four F1 Pass-A `best.ckpt` files on CPU through
  `scripts/eval_se_perclip.py::_estimates_model`. Two frictions:
  (1) that helper resolves `conf/model/*.yaml` and `results/*/best.ckpt`
  relative to the CWD, so prepare.py has to pass absolute overrides;
  (2) chunked inference leaves the final partial chunk of DCUNet's output as
  exact zeros, which renders as a black band and reads like a bug — every
  panel is therefore cropped to the region all models actually produced.
- SGMSE+ has no spectrogram panel: no local checkpoint, and its reverse-SDE
  sampler is ~30 min/utterance on CPU. It appears in the Pass-B curve only
  (Pass A was never evaluated), labelled as a compute-bounded negative control.
- Two silent Touying page-splits appeared and were fixed by splitting the
  "Why is DCUNet so bad here?" slide in two (contradiction / replication) and
  by shrinking the MP-SENet grid. Final deck is 32 pages, verified by page
  count and by reading the rendered pages.
- No guard denials this round.

## Round 5 (user review round 3 — four items)

- **Item 1 (8 s output-spectrogram clips).** The published `SE-valid-drone`
  set stores 350 clips of *exactly* 2.0 s (verified by loading every clip and
  counting samples); there is no longer clip to pick. Rather than rebuild and
  publish a dataset (which would also write outside the target dir), the new
  `_long_drone_clips()` in prepare.py re-runs the builder's own mixing code
  — `scripts/build_se_valid.py`'s drone noise pool, held-out-speaker speech
  pool, silence rejection and `_scale_source_to_snr` — at `duration_s=8.0`,
  in memory. Same distribution, same held-out speakers, longer clip; nothing
  written outside `assets/`. `_estimates_model` only touches `ds[i]["mixture"]
  .data`, so a 6-line `_Arr` shim stands in for the tdseries frame.
- **Item 2 (colour bars off, grids bigger).** The colour bar was attached to
  the top row only, which is exactly why the spectrogram panels were narrower
  than the RPS panels beneath them. Removed it; also dropped
  `bbox_inches="tight"` in favour of an explicit `subplots_adjust`, so the
  two rows share the identical horizontal extent by construction. Panels
  widened 3.0 -> 3.4 in, slide widths 94/86% -> 100/96%, and the RPS y-range
  now starts near the data (shared floor across both expanded-set halves,
  ignoring freq_scale's zero-filled tail) so the α=1.2 rescale is finally
  visible instead of being squashed against a 0-baseline.
- **Item 3 (old vs new regime).** Two new slides after the freq-shift probe:
  a score table (63.7 vs 37.6, WIP-badged) and a both-models version of the
  probe (`fig_freqshift_both`, 3x3: spectrogram row + one prediction row per
  model, shared axes, cropped to the commonly-valid duration).
  **Outcome, reported as measured:** the new-regime model does *not* follow
  the shift. Mean prediction over the clip, truth in brackets:
  no shift 79.5 (80.4), 2% 76.2 (82.0), 10% 74.7 (88.4); the old regime gives
  80.2 / 77.4 / 70.6. Both drift the wrong way; the new model is merely less
  wrong at 10%. The slide title, caption and speaker note say so plainly
  rather than claiming the intended result.
- **Item 4 (CKLA sequence redone).** Six slides, one idea and one visual each:
  comb sketch -> VK alternating-loop diagram + the VK objective -> KLA
  pipeline block + the recursion triple -> phasor pair + ā = e^((−γ+iω)Δt) ->
  phase-error panels -> the existing CKLA block diagram + the |ā|² precision
  fact. Every formula has a one-line plain-words reading directly beneath it.
  Five new matplotlib schematics live in `fig_ckla_diagrams()`; they are
  drawings, and each slide that carries one says so ("illustration, not a
  measurement") so nobody reads a number off them.
  *Deviation, logged:* the standalone "why it trains fast / Möbius scan"
  slide was dropped (no visual, and it is a detail for a general audience);
  the parallel-prefix point moved into the KLA slide's speaker note.
- prepare.py now takes figure names as argv (`python prepare.py diagrams
  freqshift_both`), because a full run costs ~6 min of CPU inference and a
  review round only ever touches two or three figures.
- Page-count discipline: 33 physical pages = title + 32 slides, one page per
  heading, confirmed by diffing `pdftotext` headings page-by-page against the
  heading list after every rebuild (no silent splits at any point this
  round). The footer total updated itself to 32.
- No guard denials this round. One non-guard friction: `make check` reruns
  the whole of prepare.py, so the verification loop used
  `typst compile` + `pdftoppm` directly and regenerated only the figures that
  changed.

### What would have made this easier

- A `make check` that does not rebuild figures (or a `make check-fast`).
  Six minutes of CPU inference per verification pass dominated this round.
- The SE-valid clip length being a build parameter rather than baked into the
  published set: wanting "the same thing, 8 seconds" should not require
  re-entering the builder's internals from a figure script.

`scripts/eval_se_perclip.py` is the right piece of machinery to reuse for
figure generation, but it assumes it is run from the repo root. If its
`_estimates_model` resolved `conf/` and `results/` from a module-level repo
root instead of the CWD, a prepare.py in any subdirectory could call it
directly with no overrides.

## Round 6 (user review round 5 — historical VK/KLA rework, overriding critic)

- Retitled "Vold–Kalman is not a Kalman filter" -> "Vold–Kalman began life as
  a Kalman filter". Rewrote it around the true 1993 history given verbatim
  by the user: sequential KF + RTS smoother, integrated-random-walk state,
  tachometer carrier in the measurement row `H(n) = [e^(iφ(n)), 0, ...]`,
  the three abandonment reasons (order-at-a-time leakage, ill-conditioned
  Riccati recursion in the heavy-smoothing regime, no joint multi-order
  recursion), and VK-2 = joint batch banded solve = Kalman smoother of the
  same model. Kept `dia_kf_vs_vk.png`, just relabelled its two panel titles
  ("1993: recursive chain, step by step" / "today: one global batch fit")
  since it already drew exactly the recursion-vs-batch contrast needed. The
  3-row table's "process assumption" row (previously implying two different
  models) was replaced by a lead-in sentence stating the shared model, and
  the surviving rows recast as "1993 recursive solver" vs "modern batch
  solver (what we run)". Added the requested punchline line verbatim.
- New slide added: "One stochastic model underneath: the Ornstein–Uhlenbeck
  process", immediately after the retitled slide. New asset `dia_ou.png`
  (`_dia_ou()` in prepare.py): left panel a real-OU sample path (fades to
  zero, `e^{-γΔt}` annotated), right panel a complex-OU sample path drawn as
  a spiral in the plane (fades and turns, start/end marked). Both real and
  complex recursion equations shown, plus a compact 3-column strip (VK-2
  batch | Vold 1993 | CKLA) with the exact one-phrase factorizations given
  in the task. This is the one new heading requested (37 -> 38).
- KLA slide ("A Kalman filter over token evidence") got the requested
  framing sentence verbatim, right below the pipeline figure.
- CKLA "The network steers the filter" slide reframed: dropped the old
  single caption line, replaced the image with a smaller (98% -> 88%) copy
  of the same block diagram, and added a 3-row table mapping each of the
  three 1993 failure modes to its CKLA answer, exactly as specified
  (tachometer -> learned ω_t; fixed λ/ill-conditioned Q-R -> per-token
  learned gains; single-order leakage -> many complex channels + learned
  query readout).
- "Why the state should rotate" got one added footnote line tying it back
  to the new OU slide, as requested.
- "Why it must be steerable: phase error accumulates" got the requested
  closing line ("Recursive (1993) → batch (1997–today) → recursive again,
  with the unknowns learned.") added below the existing two caption lines,
  and the speaker note extended with the same arc in prose.
- Page-count tripwire fired on first rebuild: 41 physical pages for 36 "="
  headings + 2 section-divider slides + title = 39 expected. Two slides had
  silently split — the retitled VK slide (its own new content was too long
  at the original font sizes) and the new OU slide (image + both equations
  + table at first-draft sizes). Fixed by shrinking the images (84% -> 66%,
  88% -> 62%) and body text (0.82-0.9em -> 0.72-0.78em) on both slides, and
  compacting the OU equation block onto one line. Rebuilt to exactly 39
  physical pages; footer total updated itself to N / 38 automatically.
  Verified via `pdftotext -layout` page-by-page heading diff (no more
  splits) and by reading the rendered PNGs of every touched page at full
  resolution via a 3x3 contact sheet (pages 26-34) plus a spot check of
  title/section-divider/results/backup pages (1, 15, 24, 35-39) to confirm
  nothing outside the CKLA block moved.
- No guard denials this round; no deviation from the user's spec — every
  fact used was one of the eight given verbatim, no invented numbers.
- What would have made this easier: same wish as prior rounds — a
  `make check` variant that fails loudly on physical-page-count !=
  heading-count would have caught the two overflow pages in the compile
  step instead of requiring a manual pdftotext diff.

## Round 7 (targeted fix — OU slide figure only)

- Scope: page 27 / heading "One stochastic model underneath" only, per
  explicit instruction to touch nothing else.
- Regenerated `_dia_ou()` in prepare.py: left panel now starts at x0=1 with
  low noise (sigma 0.03) and gamma=0.55, overlaid with the deterministic
  +/- e^{-gamma t} envelope as dashed grey curves, the e^{-gamma Delta t}
  label moved off the title line; right panel now uses omega=2*pi*0.6 >> 
  gamma=0.15 with very low noise (sigma 0.003) so the path is a clean
  ~5-turn decaying spiral, start (outer, red) and end (near origin, black)
  marked with small text annotations instead of a legend, equal aspect,
  matching axis limits/size to the left panel. Both panels are now the
  same figure size.
- Added the missing on-slide phrase, previously only in the speaker note:
  "The complex OU process is the canonical stochastic model of a tone at
  frequency omega with linewidth gamma."
- First rebuild after adding that sentence overflowed to 40 pages (the OU
  slide split its table onto an orphan page). Fixed by shrinking the image
  (62% -> 50%), dropping vertical spacers, and reducing formula/table font
  sizes (0.72-0.78em -> 0.66-0.7em, table inset 4pt -> 3pt). Rebuilt to
  exactly 39 pages / 38 headings, footer N / 38 confirmed unaffected.
- Verified only page 27 changed by re-reading it and its neighbours (26,
  28) at full resolution — 25/26 (VK slide) and 27/28 (KLA slide) render
  identically to round 6, confirming the fix was fully local.
- One PostToolUse lint-hook catch during editing: a stray leftover
  `lw(1.6) if False else 1.6` fragment from an in-place edit tripped ruff's
  syntax check before the file was saved; fixed immediately, not a guard
  denial.
- No guard denials this round.

## Round 4 (user review)

- ITEM 1: added `table_regime_mae()` to `prepare.py`. Ran both transformer
  checkpoints over the 37-clip valid-full set on CPU (4 threads, `nice`), using
  the local copy `datasets/DREGON-LM-V4-michaels-full/valid` instead of
  `dload:DREGON-LM-V4-michaels-valid-full` — it is the same 37-clip set
  (`vk.load_clip_data` validates the clip-name list against the embedded CLIPS
  table and passed), so no network fetch was needed. Regimes bucketed per FRAME
  on the rotor-mean GT (<1 / 1–50 / >=50 rev/s). Numbers written to
  `assets/regime_mae.csv` and typed into the slide.
- ITEM 2: rewrote the CKLA section as 10 slides (section title, VK recap,
  KF-vs-VK contrast, 3x KLA, 4x CKLA). Two new diagrams drawn in `prepare.py`:
  `dia_kf_vs_vk.png`, `dia_block_swap.png`. The old comb + VK-loop figures are
  reused side by side on the recap slide.
- ITEM 3: added `fig_ckla_freqshift()` — same probe machinery as
  `fig_freqshift_both`, run on `ckla_p1_pnoise_pb`. Rendered honestly: the model
  does NOT follow shifts (pred ratio x1.006 vs ideal x1.10), slide is WIP-badged
  and titled as the criterion.
- Page-count tripwire: first rebuild gave 39 pages for 37 headings — the
  "A Kalman filter over token evidence" slide silently split. Fixed by shrinking
  the pipeline image (90% -> 74%) and the table (0.9em -> 0.82em). Now 38 pages
  = 37 headings + title, as expected.
- Formula-only slides (recursion, CKLA definition) were bottom-heavy with dead
  space; used `#v(1fr)` / `#v(1.4fr)` to balance instead of manual `#v(Xem)`.
- No guard denials this round.
- What would have made this easier: knowing up front that the published
  valid-full set has a byte-identical local copy under `datasets/` — I nearly
  paid for a dload pull. A one-line "local mirrors of pinned datasets" note in
  the target dir's workflow notes would save that lookup.

## Round 8 (targeted fix — "The bar" results slide only)

- Scope: page 38 / heading "The bar: a model that follows the comb" only,
  per explicit instruction.
- WIP badge text changed to "WIP — single-seed, 2% probe" (kept the badge
  itself). Existing 3-panel `ckla_freqshift.png` probe figure kept as-is,
  just shrunk (74% -> 56% width) to make room. Bottom caption/conclusion
  replaced with the specified 3-row table (training regime x transformer /
  CKLA scale-response ratios and % of shift followed) plus the two given
  sentences, lightly formatted with `$alpha in [0.7, 1.3]$` math.
  Speaker note rewritten to explain the "following" metric and walk through
  the three regimes in prose.
- Typst gotcha: a table cell starting with a literal `+` (`+ synthesis-
  first, ...`) was silently parsed as ordered-list markup and rendered as
  "1." instead of "+" — caught by reading the full-res page, not by the
  compile log. Fixed with `#sym.plus` instead of a literal `+`.
- Page-count tripwire fired once: adding the table without shrinking first
  overflowed to 40 pages (table spilled onto an orphan page). Fixed by the
  image/table/caption size reductions noted above; rebuilt to 39 pages,
  footer 37 / 38, confirmed by `pdftotext -layout` page-by-page diff and by
  reading the final page-38 PNG at full resolution.
- Anomaly noticed and NOT fixed (out of scope per explicit instruction to
  touch nothing else): the KLA slide immediately after the new OU slide
  currently carries the heading "Recent work (2026): Attention layer
  replacement based on Kalman filter" rather than "A Kalman filter over
  token evidence", which is what earlier rounds' `Read`s of this same file
  showed at that position. The diff shows this text was written by my own
  round-6 Edit call, meaning the pre-edit file on disk at that moment
  already differed from what an earlier `Read` in this session displayed —
  most likely a concurrent edit to `slides.typ` from outside this task
  (the working tree is uncommitted). Flagging for the coordinator/user to
  check intent; left untouched since it is unrelated to the two slides this
  round and round 7 were scoped to.
- No guard denials this round.

## Round 9 (user review — uniform-v2 checkpoints replace the "new regime" story)

- Swapped every "new regime" / gain-fixed-CKLA reference to the verified v2
  checkpoints: `g2_if_freqscale_v2` (transformer) and `ckla_pnoise_fs_v2`
  (CKLA), both loaded exactly like round 4's regime table via
  `vk.load_model(experiment, uri, "cpu")`. `prepare.py` constants `NEW_CKPT`
  / `NEW_EXP` / `CKLA_CKPT` / `CKLA_EXP` updated; `fig_freqshift_both`,
  `table_regime_mae`, `fig_ckla_freqshift` all re-point at the new arms and
  were rerun (`python prepare.py freqshift_both regime_table
  ckla_freqshift`, ~2 min on CPU).
- Slide 21 ("Fixing the augmentation regime moved the score"): both tables
  rewritten from the fresh `regime_mae.csv`. Total PIT-MAE 4.72 -> 3.83,
  envelope MSE 63.7 -> 42.3 (given). The per-regime table tells a different,
  more honest story than the old "new regime" pass: DREGON gets *worse*
  everywhere under uniform scaling (warm-up 11.0 -> 15.8 is the worst hit),
  FLY124 gets much better (idle 30.5 -> 5.0); the net PIT-MAE improvement is
  entirely carried by FLY124's larger gain outweighing DREGON's loss. Caption
  and speaker note now say this plainly instead of implying a clean win.
- Slide 22 (renamed "The uniform regime partially follows the shift"):
  regenerated `freqshift_both.png` with the v2 transformer as the second row.
  Found a real discrepancy: this single probe clip's own numbers (no shift
  80.7, 2% shift 80.8, 10% shift 78.5) show v2 moving the *right* direction
  at 2% but slipping back at 10%, which does not literally match the given
  12-clip verbatim 42%/71% following figures. Resolved by keeping the
  figure's honest single-clip numbers in the caption and explicitly stating
  the 12-clip probe average (42%/71%) is the number to trust, rather than
  overwriting what the regenerated figure actually shows. Same reconciliation
  applied on slide 37 ("The bar"), where the regenerated `ckla_freqshift.png`
  single clip undershoots (pred x0.981 at 10% shift) against the 12-clip
  average of x1.0992 — added a one-line grey caption pointing at the table
  as the source of truth instead of silently letting the two numbers on the
  same slide contradict each other.
- CKLA results block (slides 34-37): slide 34 (full-envelope) now compares
  the matched pair under the same v2 regime (transformer 42.3 vs CKLA 41.4).
  Slides 35 (cruise pools) and 36 (one-prediction) still show the old
  gain-fixed/protocol-B checkpoint's numbers/image — per the task's explicit
  instruction these were NOT rewritten as if they were fs_v2's; instead their
  WIP badges and captions were reworded to say plainly that fs_v2 has not
  been re-evaluated on cruise pools / does not have a fresh prediction plot
  yet. Slide 37 ("The bar") table extended from 1 to 3 alpha columns
  (x1.02/x1.05/x1.10) and rewritten with the exact verbatim numbers given in
  the task (not recomputed); the v3 corruption-ramp row was dropped, per the
  task's "if space is tight" option, since the 4-column table was already
  close to the slide width at readable font size.
- Deleted slide 38 ("Backup — the retracted reading") entirely per
  instruction. Deck is now 37 headings, 38 physical pages (title + 37),
  confirmed by a `pdftotext -layout` footer scan showing every "N / 37" from
  1 to 37 exactly once (no silent Touying splits).
- Typst gotcha (new this round): `#sym.dash.em` rendered as *nothing* — an
  em-dash written as `---` immediately after an inline `` `code` `` span
  (raw text) also silently vanished, even though the earlier em-dash
  workaround (literal "—" character or restructuring the sentence) has
  worked in every prior round. Root cause not fully isolated; worked around
  by removing the dash from that one sentence (rephrased with a colon)
  rather than spending more time bisecting, per the "don't bug-hunt for
  more than 5 minutes" rule. Caught only by `pdftotext -layout`, not by
  looking at the rendered PNG at normal reading size — the missing
  character left no visible gap.
- No guard denials this round.
- What would have made this easier: the single-clip vs 12-clip-average
  scale-response numbers disagreeing on both probe slides (22 and 37) cost
  real time to reconcile honestly rather than just overwriting the caption
  text with the given verbatim numbers. A note in the task handoff on
  whether the single illustrative clip was expected to reproduce the
  12-clip average, or whether disagreement was anticipated, would have
  saved that back-and-forth.
