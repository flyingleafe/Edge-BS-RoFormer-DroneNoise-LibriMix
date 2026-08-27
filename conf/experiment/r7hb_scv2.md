# r7hb_scv2 — the stochastic family as a permanent part of the pool

The joint counterpart of R6. Both put the stochastic family into the mixed
regime that produced this project's best result; they differ in *where*.

| | R6 | R7 |
|---|---|---|
| shape | two-stage curriculum | one stage |
| stochastic family used as | stage-1 initialization | 22.7% of the training pool |
| real data | stage 2 only | 45.5% of the same pool |
| sibling it modifies | `r4hb_scv2` (**2.67**, the target) | `r5hb_scv2` |
| the one changed thing | stage-1 checkpoint | the third noise source |

## The pool

Real 45.5% : silence 9.1% : stochastic 22.7% : static comb 22.7% — weights
2.0 : 0.4 : 1.0 : 1.0, exactly R5's ratio with the stochastic source standing
where the M3 neural generator stood. The real block, the honest silence arm,
the comb and the policy are all R5's verbatim; the stochastic block is arm ID's
verbatim.

Arm ID's own `kind: silence` arm (weight 0.2) is dropped, because the R5 real
block already carries the honest silence arm at 0.4 and keeping both would put
0.6 of silence in the pool and change the ratio this arm holds fixed.

## What the pair answers

Whether the synthetic family is better used as an **initialization** that real
data then overwrites, or as a **permanent part of the pool** that keeps
supplying states the real recordings do not contain. The whole stochastic
campaign has been optimizing the family as a stage-1 pre-training target; if
R7 beats R6, that framing was wrong and the family's value is coverage held
throughout training, which is also what the campaign's own durable lesson
("coverage beats realism") would predict.

Unlike R5 this stream runs on a CPU-only box — the stochastic source renders on
the CPU, whereas `kind: generated` starts a CUDA producer.

Stream check: PASS (determinism 4/4 bit-identical).
Batch doc: `docs/experiments/stochastic-transfer.md`.
