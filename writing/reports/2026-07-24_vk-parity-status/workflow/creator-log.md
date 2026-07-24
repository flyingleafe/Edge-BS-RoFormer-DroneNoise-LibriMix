# Creator log — VK parity status report

- No guard denials during build (one incidental error: chained `mkdir && ls`
  in a single Bash call tripped the "outside target dir" check because the
  guard parses the whole compound command; fixed by issuing `mkdir` alone).
  Not a real blocker — no workaround needed beyond splitting the command.
- Figures: reused/regenerated recipes from the 07-18 deck's prepare.py
  (coupling schematic, speedup bars) verbatim; built two new figures
  (FLY124 blind-overlay from the r6 sweep npz, alias illustration, parity
  bars) from scratch since no report-quality equivalents existed yet.
- FLY124 npz gotcha confirmed from the narrative: `edge` array in the sweep
  npz is a KEEP mask, not an exclude mask — used correctly.
- Phase-B "best" numbers (2.87 DREGON / 1.90 FLY124) were reconstructed by
  scanning the omnirun report.json/summary.txt across the g1_transformer_4s
  and g1_transformer_8s arms/smoothing combos for the best entries; matches
  the narrative's quoted figures closely (best chmean med2 dregon 2.872,
  best ch0 dregon_cruise for 4s/none 3.619 vs FLY124 1.902 on 4s/ch0/none —
  picked the single best cell per pool across all arms, consistent with how
  phase A's "best" was quoted in docs/experiments/g1-vk-parity.md).
- What would have made this easier: a single canonical "phase B best" summary
  table already computed (like PHASE_A_SUMMARY.txt) rather than needing to
  scan raw JSON for the winning smoothing arm per pool.

## Round 1 critique fixes
- Table 2 "+ fix" cells: `+` at start of a table cell is parsed by Typst as
  an enum marker even inside `table()` cells — escaped as `\+` in both rows.
  Non-obvious Typst gotcha worth remembering for future decks/reports.
- vk_parity_bars.png: raised y-limit (1.35x max bar) and moved legend to
  upper right so the 2.87 value label clears the legend box.
- vk_alias_illustration.png: moved the scan_f_max annotation down near the
  x-axis (was colliding with the legend at top-right) and added ax.set_ylim
  headroom.
- Verdict section: reworded to quote the best neural lever (phase A: 2.62 /
  1.55) as the headline gap, with phase B (worse, 2.87 / 1.90) noted as the
  refuted alternative — consistent with the abstract.
- Both "[TODO verify ...]" bracket markers reworded to plain bold lead-ins
  ("In build, not yet measured:" / "In build now, no numbers yet:") — no
  content change, same pending-status semantics.
- No issues rejected this round; all 5 were valid and cheap to fix.
- Rebuilt via `make check`; page count held at 11; re-inspected pages 7, 8,
  9, 10 (all figure/table/text changes) — all render correctly now.
