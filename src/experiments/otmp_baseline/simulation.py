"""The paper's Sec. VIII-A Monte-Carlo self-test (gross error rate).

Björkman & Elvander (arXiv:2508.02471), Sec. VIII-A: four pitches with nominal
fundamentals 176, 197, 240 and 272 Hz, perturbed at random per draw, each with
a random number of harmonics between 3 and 10, unit magnitudes and uniform
random initial phase, observed over ``N = 250`` samples at 8 kHz. Performance
is the *gross error rate*

    GER = (1/K) sum_k [ |1200 * log2(f_hat_k / f_k)| > 50 ]

i.e. the fraction of estimates off by more than a quarter note. The paper
reports roughly 8-10 % for ``Proposed_c`` at 5 dB SNR (Fig. 4).

Result of record for this reimplementation, 50 draws, seed 0, the defaults of
:func:`~experiments.otmp_baseline.estimate.simulated_config`::

    GER = 28.0 %,  median |deviation| = 9.2 cents,  18.4 s/draw

The median deviation says the pitches that *are* found are found accurately —
9 cents, well inside the 50-cent bar — so the 28 % is not a precision problem
but a detection one: in most failing draws one of the four pitches carries no
mass at all and a spurious candidate is reported in its place. The pitches
that go missing are the ones the draw gave few harmonics (the count is uniform
on 3..10). With ``eta = 0.1`` and ``zeta = 10`` at this normalization, opening
a new column for a 3-harmonic pitch costs ``zeta*eta*max_f M`` in the
group-sparsity term, which is more than the transport cost of absorbing those
three partials into a neighbouring column — so the estimator trades the weak
pitch away. That, rather than the octave ambiguity, is the gap to the paper.

Run it with::

    PYTHONPATH=src python -m experiments.otmp_baseline.simulation --draws 50

[choice] The paper's model (2) is complex-valued and its SNR definition
``10 log10(sum_k sum_l |alpha|^2 / sigma^2)`` counts one unit per partial, so
the draws here are generated directly as analytic signals with circular
complex Gaussian noise. Generating a real signal and Hilbert-transforming it
would shift the effective SNR by 3 dB.
[choice] The perturbation of the nominal fundamentals is unspecified; +-1 Hz
is used, which is a full step of the 1 Hz pitch grid and so removes the
grid-alignment bias the paper is guarding against.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from experiments.otmp_baseline.estimate import OTMPConfig, estimate_frame, simulated_config

__all__ = [
    "NOMINAL_PITCHES_HZ",
    "MonteCarloResult",
    "gross_error_rate",
    "run_monte_carlo",
    "simulate_frame",
]

NOMINAL_PITCHES_HZ = (176.0, 197.0, 240.0, 272.0)


def simulate_frame(
    rng: np.random.Generator,
    *,
    sample_rate: int = 8000,
    n_samples: int = 250,
    pitches_hz: tuple[float, ...] = NOMINAL_PITCHES_HZ,
    snr_db: float = 5.0,
    harmonics: tuple[int, int] = (3, 10),
    perturb_hz: float = 1.0,
    inharmonicity: float = 0.0,
) -> tuple[NDArray, NDArray]:
    """One Monte-Carlo draw. Returns ``(analytic_signal, true_pitches_hz)``."""
    true = np.asarray(pitches_hz, dtype=np.float64) + rng.uniform(
        -perturb_hz, perturb_hz, size=len(pitches_hz)
    )
    t = np.arange(n_samples, dtype=np.float64)
    sig = np.zeros(n_samples, dtype=np.complex128)
    n_partials = 0
    nyq = sample_rate / 2.0
    for f0 in true:
        n_harm = int(rng.integers(harmonics[0], harmonics[1] + 1))
        for k in range(1, n_harm + 1):
            freq = k * f0
            if inharmonicity:
                freq += rng.uniform(-1.0, 1.0) * inharmonicity * freq
            if freq >= nyq:
                continue
            phase = rng.uniform(0.0, 2.0 * np.pi)
            sig += np.exp(1j * (2.0 * np.pi * freq / sample_rate * t + phase))
            n_partials += 1
    sigma_sq = n_partials / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(sigma_sq / 2.0) * (
        rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    )
    obs = sig + noise
    return obs / np.sqrt(np.mean(np.abs(obs) ** 2)), true


def gross_error_rate(
    estimated_hz: NDArray, true_hz: NDArray, cents: float = 50.0
) -> tuple[float, NDArray]:
    """Fraction of estimates more than ``cents`` off, after optimal pairing.

    Estimates and references are paired by the Hungarian algorithm on the
    absolute cents deviation; the paper's eq. leaves the pairing implicit but
    scores one estimate per reference pitch.
    """
    from scipy.optimize import linear_sum_assignment

    est = np.asarray(estimated_hz, dtype=np.float64).ravel()
    ref = np.asarray(true_hz, dtype=np.float64).ravel()
    dev = np.abs(1200.0 * np.log2(est[:, None] / ref[None, :]))
    rows, cols = linear_sum_assignment(dev)
    paired = np.full(ref.size, np.inf)
    paired[cols] = dev[rows, cols]
    return float(np.mean(paired > cents)), paired


@dataclass
class MonteCarloResult:
    """Aggregate over draws."""

    ger: float
    per_draw_ger: NDArray
    deviations_cents: NDArray  # (draws, K)
    seconds_per_draw: float
    n_iter_mean: float
    converged_frac: float


def run_monte_carlo(
    draws: int = 50,
    cfg: OTMPConfig | None = None,
    *,
    snr_db: float = 5.0,
    seed: int = 0,
    verbose: bool = False,
) -> MonteCarloResult:
    """Run the Sec. VIII-A study and report the gross error rate."""
    cfg = cfg or simulated_config()
    rng = np.random.default_rng(seed)
    gers, devs, iters, conv = [], [], [], []
    start = time.perf_counter()
    for draw in range(draws):
        obs, true = simulate_frame(
            rng, sample_rate=cfg.sample_rate, n_samples=cfg.frame_len, snr_db=snr_db
        )
        est = estimate_frame(obs, cfg.sample_rate, cfg)
        ger, dev = gross_error_rate(est.pitches_hz, true)
        gers.append(ger)
        devs.append(dev)
        iters.append(est.n_iter)
        conv.append(est.converged)
        if verbose:
            print(
                f"draw {draw:3d}  GER {ger:4.2f}  "
                f"true {np.round(np.sort(true), 1)}  "
                f"est {np.round(np.sort(est.pitches_hz), 1)}  "
                f"({est.n_iter} it)",
                flush=True,
            )
    elapsed = time.perf_counter() - start
    return MonteCarloResult(
        ger=float(np.mean(gers)),
        per_draw_ger=np.asarray(gers),
        deviations_cents=np.asarray(devs),
        seconds_per_draw=elapsed / max(draws, 1),
        n_iter_mean=float(np.mean(iters)),
        converged_frac=float(np.mean(conv)),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=50)
    parser.add_argument("--snr-db", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--inner-iters", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    over = {}
    if args.max_iter is not None:
        over["max_iter"] = args.max_iter
    if args.inner_iters is not None:
        over["inner_iters"] = args.inner_iters
    cfg = simulated_config(**over)
    res = run_monte_carlo(
        args.draws, cfg, snr_db=args.snr_db, seed=args.seed, verbose=not args.quiet
    )
    print(
        f"\nGER = {100 * res.ger:.1f} %  over {args.draws} draws at {args.snr_db} dB\n"
        f"median |deviation| = {np.median(res.deviations_cents):.1f} cents\n"
        f"{res.seconds_per_draw:.2f} s/draw, {res.n_iter_mean:.0f} iterations mean, "
        f"{100 * res.converged_frac:.0f} % converged"
    )


if __name__ == "__main__":
    main()
