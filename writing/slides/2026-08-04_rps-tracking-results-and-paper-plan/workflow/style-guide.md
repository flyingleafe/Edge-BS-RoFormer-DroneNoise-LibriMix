# Slide style guide — what this user wants in a deck

Compiled 2026-08-04 from: workflow critiques + narratives of the 3 newest decks
(2026-07-18, 2026-07-27, 2026-08-04), user turns in 17 session transcripts,
project memory (writeup-workflow, writeup-agent-gotchas, quiet-drones,
rps-plot-pit-alignment, fly124 eval, append-to-typst), the writeup-creator /
writeup-critic definitions, and the Touying template.

Confidence tags: [R] = seen repeatedly, [1] = seen once, [I] = inferred.

## (a) Text density and wording

- **As few words as possible on slides.** The binding formula (2026-08-04
  narrative, GLOBAL REQ 1): each slide = title + one figure/table + 1–3 short
  lines, nothing else. Creator rule of thumb: ≤ 40 words per slide. The user
  said this verbatim: "use as few words as possible on slides". [R]
- **All explanation goes into speaker notes, on EVERY slide.** Notes are the
  speaker script AND build instructions — the user hand-edits them and treats
  them as directives ("Treat those speaker notes as INSTRUCTIONS"). Never
  delete or thin them. (07-18 narrative; sessions da86, 44eb; /goal hook
  8c62.) [R]
- **The slide title states the takeaway**, not a label ("Removing one
  threshold beat the synthetic pipeline", not "Results 3"). (creator craft
  rules; critic rubric.) [R]
- **Plain human language, but supervisor-level — do not dumb it down.** When
  an agent over-simplified, the user pushed back: "not AS simple… SOME jargon
  is better than 'borrow a yardstick'". No cutesy metaphors; real technical
  terms are fine. (session 8c62.) [1, but load-bearing]
- **No hype words** — "novel", "leverage", "delve", "we posit", marketing
  tone. Short whiteboard sentences. (critic rubric, applied every round.) [R]
- **No internal designators on slides.** "protocol-B", "gain-fix", experiment
  handles → plain words on the slide, designator only in speaker notes.
  Expand acronyms like PIT-MAE in a footnote on first use. (07-27 critique
  round 3, items 3–4.) [R]
- **No `§N` cross-references** to narrative/report sections — the deck has no
  visible numbering; refer to threads by name. (07-18 critique round 1.) [1]
- **Every statement must be backed by a shown number or plot.** User verbatim:
  "statements which are not backed by demonstrated numbers (as numbers or
  plot figures) are not allowed". Numbers always need an anchor
  (before/after, baseline/ours). (session da86; critic rubric.) [R]
- **Method slides must have rebuild-from-slides fidelity**: a reader should be
  able to reimplement the algorithm from the deck. Pseudocode is explicitly
  welcome; the coupling condition / update equations must be ON the slide, not
  only in notes. (user in da86; 07-18 critique round 2 item 1.) [R]
- **Honest status labels, no drama.** Pending results: mark cells "WIP" /
  "updating" / "re-score in flight" with a footnote — never quote stale
  numbers as current, and never present unmerged tooling as existing. But
  also do NOT narrate retractions or bugs as plot twists: "no 'twists' like
  'validation set were lying'… we DON'T need to explain that we had incorrect
  results" — present the fixed view, mark WIP. (08-04 numbers policy; 07-27
  critique tone item; sessions 97a8, da86.) [R]
- Prose obeys the global Simplified Technical English rule (user config). [I]

## (b) Figures and plots

- **Figures + formulas dominate; text only anchors.** "Figures are better
  than text — the speaker knows what to say anyway." Every content slide is
  carried by a figure, table, or diagram. (user in da86; 07-18 narrative
  "Style law"; critic rubric.) [R]
- **Always show model outputs.** User rule of thumb, verbatim: "always show
  model outputs!" — prediction overlays on real samples, "you can do many
  slides like these its alright". Qualitative output slides are never a
  waste. (session 97a8.) [R]
- **RPS comparison plots have a house format**: RPS-vs-time, all 4 rotors,
  predicted solid + GT **dotted**, PIT-aligned via `align_rps_to_gt`
  (`src/tasks/rps_prediction.py`) — unaligned plots show fake rotor swaps.
  Spectrogram + RPS side-by-side = the `two_columns=True` renderer pattern.
  (memories rps-plot-pit-alignment, fly124-simpleconvv2-eval; 08-04 narrative
  GLOBAL REQ 3.) [R]
- **Spectrogram comparisons: real next to generated, side by side**, ideally
  with a mean-spectrum line plot under each ("under each spectrogram pls add
  mean spectrum plot"), and per-rotor / per-component panels when comparing
  generator variants. Overlaid spectrum line plots reveal what stacked
  spectrograms hide. (sessions 2aaf, 97a8, 7553.) [R]
- **Sync axes across comparison panels.** Identical xlim/ylim for every panel
  in a grid; include a clean reference panel as the first column; crop all
  panels to the common valid duration so artifacts don't compete with the real
  effect. ("x- and y-axis synced… so that absolute changes in RPS are
  apparent"; 07-27 critique round 3 items 2, 6.) [R]
- **Big figures — use the space.** Multi-row layouts so spectrograms are not
  tiny ("use multi-row figures… spectrograms are too small otherwise");
  enlarge instead of leaving the bottom half of a slide blank. Axis labels
  and legends must be readable at 150 dpi page renders — if not, regenerate
  the asset with fewer panels and larger fonts, don't just scale the PNG
  down. (session 97a8; 07-18 critique round 4 item 6; 07-27 critique item
  3.) [R]
- **Crop or drop uninformative panels** (e.g. a flat spectrum column) so the
  informative panels fill the slide. (07-18 critique round 1 item 5.) [1]
- **Difference images to localize a win**: when a metric improves, show WHERE
  on the spectrogram (e.g. |Δ log-mag| image next to the spectrogram).
  (session 97a8.) [1]
- **Stepper/slider style for algorithms**: one pipeline diagram repeated
  across ~5 slides, current step highlighted, plus one small real-data panel
  and at most ONE formula per step. The user asked for this by name ("in the
  same form as slider in the artifact explainer"). (session 44eb; 08-04
  narrative sections 3–7.) [R]
- **Bar charts for money comparisons** with a one-line takeaway under them
  ("running four rotors at once destroys phase coherence"). (08-04 narrative
  slides 12–13.) [1]
- **Plot hygiene** (improve-plot-visibility skill): percentile clipping
  (vmin/vmax 1–5 / 99–99.5), colormap trials (inferno/magma/hot/bone_r),
  per-sample clipping when SNRs differ, log/dB scaling; verify the final PNG
  at ≥150 dpi, never trust the PDF view alone. [R]
- **Copy-first figures**: reuse existing report/notebook PNGs when they serve;
  regenerate only when the narrative demands it. Assets are generated by a
  `prepare*.py` inside the deck dir; images are gitignored (only .typ/.csv/.py
  tracked). (creator rules; append-to-typst memory.) [R]
- **Sample choice matters**: check the sample actually shows the phenomenon
  (a "full recording" panel where GT is all-zero was called "plain wrong";
  FLY124 must be filtered to stable in-flight slices or idle inflates error).
  (session 97a8; fly124 memory.) [R]

## (c) Tables

- **Small, punchy tables are welcome — and often demanded.** A compact 3-row
  banded table beat a 110-word text wall (07-18 critique r1 item 3); the user
  asks for a headline results table early in the deck ("slide 3 should
  include current overall results table (MAE) for best VK tracker and best
  neural models"). [R]
- **MAE in rev/s is the results currency**; the user asked for "full results
  table in MAE, we can even drop the full table in MSE". (session 97a8.) [1]
- **Readable size**: if a wide table renders small, transpose it ("table smal
  make vert cols" — make vertical columns). (session 97a8.) [1]
- **One metric name + direction everywhere.** The same quantity must not
  appear as "MR-STFT ↑" on one slide and "msSTFT ↓" on another; state
  "lower = better" explicitly. (07-18 critique round 4 item 5.) [1]
- Pending cells: mark `*` + footnote "re-score in flight" rather than omit or
  quote stale values. (08-04 narrative numbers policy.) [R]

## (d) Structure and pacing

- **Scale**: 8–14 narrative sections, ≈1–2 slides each; recent decks are
  18–20 slides plus backups. One point per slide. (writeup SKILL; 08-04
  narrative.) [R]
- **Opening**: title, then ONE text-allowed overview slide ("the one
  text-allowed slide") mapping the week's threads by name. (07-18
  narrative.) [R]
- **Punchline gets its own slide** (the narratives literally mark a "MONEY
  SLIDE"); secondary messages come after it, not before. (writeup SKILL;
  08-04 narrative slide 12.) [R]
- **Endings**: significance in ~3 bullets of ≤6 words each, then the concrete
  near-term plan — the next paper / next 4 weeks, NOT the full thesis plan
  ("slides should NOT include the full plan until the end of the thesis").
  (session 44eb; 08-04 narrative.) [R]
- **Backup slides after a "Backup" divider** for Q&A depth (the Quiet Drones
  deck: 12 main + divider + 5 backups). Retracted/secondary material can live
  there. (quiet-drones memory; 07-27 deck "Backup — the retracted
  reading".) [R]
- **Part dividers are fine** when extending an existing deck; existing parts
  stay INTACT unless the user says otherwise ("Existing Part-1 slides must
  NOT be rewritten (light touch only)"). (07-27 narrative.) [R]
- **Consistent slide-family naming**: if slide 2 promises "Three assumptions",
  the follow-ups must all be "Assumption 1/2/3", not a mix with "Hypothesis".
  (07-18 critique round 4 item 7.) [1]
- **Standalone clarity for a cold reader** who has only seen the previous
  decks: introduce every acronym/experiment name before use; the deck must
  stand alone even if the supervisor skipped the reports. (critic persona;
  08-04 audience note "is lost in them — this deck must stand alone".) [R]
- **Training-setup slides are content, not filler**: when results depend on
  data/augmentation choices, the user wants them shown ("which data, how
  combined, how generated data is obtained, augmentations — none of that is
  there!"), preferably as a name+description table plus a visual
  before/after of one augmented sample. (sessions 97a8, da86.) [R]

## (e) Known irritants (repeatedly corrected)

1. **Orphan / continuation pages and dead whitespace.** The #1 recurring
   critique across every round of both workflow decks: title-less overflow
   pages, half-empty slides. User verbatim: "too much unused white space on
   the slides is bad." Fix by enlarging content, merging sparse slides, or
   splitting deliberately with a real title. [R]
2. **Content that exists only in the speaker note.** The central mechanism
   of a slide (a bug, a coupling condition) must be visible on the slide
   body; the note is script, not a hiding place. (both decks' critiques.) [R]
3. **Tiny/illegible figures** — shrunken axis text, blurry panels, clipped
   plot frames, missing axis labels. Regenerate at slide-friendly size. [R]
4. **Walls of bullets / text-only slides** (> ~40 words, no figure). [R]
5. **Unbacked claims and stale numbers** — every number verified or marked
   `[TODO verify]`/WIP; never invent, never quote pre-recalibration values. [R]
6. **Fake narrative drama** — invented twists, dwelling on retracted results,
   mentioning dead-end diversions the user considers noise ("Kalman kill —
   don't even mention it"). [R]
7. **Over-simplified language** for a supervisor audience. [1]
8. **Wrong or degenerate example samples** (all-zero GT, idle segments). [R]
9. **Slow builds.** Deck building is "low-cognition, high-perceptual-
   feedback": ≤20–30 min to first review, ≤5 min per iteration; many fast
   iterations beat few deliberate ones; ≤5-min cap on rendering-bug hunts —
   simplify the layout instead. (user directive 2026-07-13.) [R]
10. **Metric name/direction inconsistency** across slides. [1]

## (f) Typst/Touying mechanics that broke before (with fixes)

Template facts: Touying 0.7.4, simple theme, 16:9, 16 pt base text,
left-aligned, page counter footer, no per-section slides
(`new-section-slide-fn: none`). Import root-absolute
(`#import "/writing/templates/typst/slides.typ": hns-slides`); Makefile must
compile with `--root $(git rev-parse --show-toplevel)`. `make check` renders
pages via `pdftoppm -r 150`. Speaker notes via `#speaker-note[...]`
(`--input notes=1` renders them). (writing/AGENTS.md, template.) [R]

Known traps (writeup-agent-gotchas memory + creator logs — all confirmed on
this template):

- **Em-dash + `*bold*` in the same bare paragraph splits the slide** into
  spurious subslides (still present in Touying 0.7.4). Safe inside list
  items, grid/table cells, speaker notes. In bare paragraphs use `:` `,` `.`
  instead of `—`. [R]
- **Tall figures silently spill onto extra pages.** Size tall images by
  height: `#align(center, image("f.png", height: 64%))`, leave room for the
  keyline. [R]
- **Page-count tripwire**: physical pages must equal (count of `= ` headings)
  + 1 title. A surplus page = silent split or overflow; bisect per figure.
  Mechanical overflow check: rasterize and measure ink fraction (~0.75% =
  orphan page vs 2–5% real). [R]
- **`~` is a non-breaking space**, not "approximately" — it silently eats the
  symbol ("~0 eSTOI" → "0 eSTOI"). Use `$approx$` / `≈`; grep for `~` before
  shipping. [R]
- **Math**: `e^{...}` renders literal braces — write `e^((-gamma + i omega)
  Delta t)`; `||s||^2` mis-parses — use `norm(s)^2` with explicit grouping. [R]
- **`#let hl = text(fill:..)` fails** ("missing argument: body") — define as
  a function: `#let hl(body) = text(fill: .., weight: "bold", body)`. [1]
- **Touying splits a paragraph mid-sentence at an inline raw span**
  (`` `code` ``) onto a blank next slide — restructure (short prose first,
  table/figure last) rather than fight it. [1]
- **First compile may need network** for `@preview` packages — pre-warm by
  compiling an existing deck. [R]
- **prepare.py hygiene**: new scripts must pass pyright+ruff pre-commit
  (grandfathered old ones do not) — cast tdseries getitem results, `# ruff:
  noqa: E402` after `os.chdir(PROJECT_ROOT)`, no bare `json.dump(...,
  open(...))`. Assets write only into `<deck>/assets/`. [R]
- **Verification is non-negotiable**: `make check`, then actually look at
  every rendered page (contact sheet via `montage` first, full-res where
  suspicious). Critics auto-REVISE unrendered work. [R]
