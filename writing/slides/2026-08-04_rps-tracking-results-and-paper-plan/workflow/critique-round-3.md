# Critique round 3 — verdict: APPROVE (2 non-blocking nitpicks)

1. [figures] page 13 (lock ladder, assets/lock_ladder.png): the green
   threshold label "needed for lock (~0.7)" partly overlaps the single-motor
   blue bar, reducing contrast. Fix in prepare_figs.py: move the label anchor
   right of the first bar group (over the "4 motors" gap) or add a white
   bounding box behind the text.
2. [clarity] page 15 (displaced comb, hover row): "hover — same on-grid
   harmonics, 3–4× weaker than free flight" reads as amplitude-weaker; the
   finding is the DISPLACEMENT is 3–4× weaker in hover. Reword to
   "hover — displacement 3–4× weaker (needs translation)".
