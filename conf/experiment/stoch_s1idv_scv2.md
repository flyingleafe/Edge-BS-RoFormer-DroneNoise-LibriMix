# stoch_s1idv_scv2 — arm IDV: cover both line-visibility regimes

Arm ID with FOUR knobs widened and nothing else changed. It targets the two
cells that hold the whole remaining gap to the real-only target, Michael's ramp
and Michael's cruise, which no single model holds.

## Why

The per-cell bests come from two arms that differ mainly in one block:

| arm | line-visibility block | Michael's ramp | DREGON cruise |
|---|---|---|---|
| S | absent (defaults) | **8.14** | 15.26 |
| H / ID | present, tight | 20.71 | **2.16** |

Making the lines easy to see buys cruise and costs the ramp. The campaign's one
durable lesson is that coverage beats realism, so the move is not to pick a side
of the trade but to put both sides in one stream. The ranges are the union:

| knob | arm H (ID) | defaults (S) | arm IDV |
|---|---|---|---|
| `harm_coherence` | [0.6, 1.0] | [0.0, 1.0] | [0.0, 1.0] |
| `harm_gp_std_db` | [0.5, 3.0] | [0.5, 6.0] | [0.5, 6.0] |
| `floor_rel_db` | [-30, -8] | [-22, -2] | [-30, -2] |
| `min_lines_above_floor` | 0.5 | 0.30 | 0.30 |

## What the result means

- Keeps ID's cruise and takes some of arm S's ramp: the trade was a coverage
  artifact, and the same move should be tried on the other committed knobs.
- Lands between the two arms on BOTH cells: the trade is real, one stream
  cannot serve both, and this direction is closed. Routing or a per-clip
  conditioning signal would be the only way left to hold both.

## Note on the inert phase block

The flight-phase block sits one level above where the loader reads it, so it
never applies and the run uses `FlightPhaseRanges` defaults. That is arm ID's
documented accident, kept deliberately — the defaults fit the real ramp
distribution better than the written values (total variation 0.254 against
0.310), and arm IDV must differ from arm ID in the visibility block alone.

Policy: `conf/online_mix/stoch_s1idv_dload.yaml`.
Batch doc: `docs/experiments/stochastic-transfer.md`.
