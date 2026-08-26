

def test_level_per_flight_holds_one_gain_for_a_whole_flight():
    """A real recording has ONE gain, so loudness tracks speed inside it.

    Off (the default), every window redraws the reference level and the
    across-flight spread lands INSIDE each flight. On, the flight's windows
    share one gain and only the speed law moves the level.
    """
    import numpy as np

    from data_processing.stochastic_rotor_noise import StochasticNoisePool

    def levels(per_flight: bool) -> np.ndarray:
        pool = StochasticNoisePool(
            sample_rate=16000,
            duration_s=0.5,
            n_mics=1,
            n_harmonics=40,
            rps_kind="full_flight",
            level_mode="window",  # isolate the gain draw from the speed law
            normalize_rms=(0.02, 0.25),
            flight_reuse=8,
            level_per_flight=per_flight,
        )
        rng = np.random.default_rng(0)
        return np.array(
            [
                20 * np.log10(float(np.sqrt(np.mean(np.square(pool.render(rng, 0.5)[0])))))
                for _ in range(8)
            ]
        )

    # level_mode="window" normalizes each window to the drawn level exactly, so
    # with one gain per flight the eight windows of a single flight are equal.
    assert float(np.std(levels(True))) < 0.1
    assert float(np.std(levels(False))) > 1.0
