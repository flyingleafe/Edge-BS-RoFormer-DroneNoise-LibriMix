

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


def test_mode_scales_make_near_degenerate_rotor_pairs():
    """Real quadrotors cruise with two rotors at nearly the same speed.

    On the frozen split's cruise frames two rotors sit within 1 rev/s in 71.6%
    of DREGON frames and 42.9% of Michael's, against 17 to 25% in every
    synthetic stream measured. Yaw drives the diagonal pairs apart and leaves
    each pair together, so scaling roll and pitch down against yaw reproduces a
    wide spread and a near-degenerate pair at the same time.
    """
    import numpy as np

    from data_processing import rps_synthesis as rs

    def degenerate_fraction(mode_scales):
        rng = np.random.default_rng(0)
        gaps = []
        for _ in range(30):
            w = rs.generate_full_flight(
                None, 200.0, drone_profile=0.0, aggressiveness=1.4,
                mode_scales=mode_scales, rng=rng,
            )
            cruise = w[:, w.mean(axis=0) >= 45.0]
            if cruise.shape[1] < 5:
                continue
            gaps.append(np.min(np.diff(np.sort(cruise, axis=0), axis=0), axis=0))
        return float((np.concatenate(gaps) < 1.0).mean())

    assert degenerate_fraction(None) < 0.40
    assert degenerate_fraction({"roll": 0.45, "pitch": 0.45, "yaw": 1.7}) > 0.60
