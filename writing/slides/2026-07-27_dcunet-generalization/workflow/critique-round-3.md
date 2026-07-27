# Critique round 3 (post user-review rewrite) — VERDICT: REVISE

1. [figures] p25 CKLA derivation: `e^{(−γ+iω)Δt}` renders literal braces
   (Typst treats `^{...}` braces literally) — slides.typ:572; use
   `e^((-gamma + i omega) Delta t)`.
2. [figures] p19 expanded-set grid 2/2: no clean reference panel and RPS
   y-axis ~90 vs grid 1's ~110 — violates the identical-axes spec. Repeat
   the clean panel as first column and share exact xlim/ylim with grid 1.
3. [clarity] p27/p29: "protocol-B", "gain-fix" are internal designators —
   replace on-slide with plain words; keep in speaker notes.
4. [clarity] p28: expand "PIT-MAE" and "neural floor" in a footnote.
5. [clarity] p22: gloss k, v, q, λ_v under the readout line.
6. [figures] p20 freq-shift probe: crop all panels to common valid duration
   (end-of-clip plunge artifact competes with the real failure).
