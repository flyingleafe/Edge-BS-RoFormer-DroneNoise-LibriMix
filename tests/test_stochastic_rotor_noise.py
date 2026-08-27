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
                None,
                200.0,
                drone_profile=0.0,
                aggressiveness=1.4,
                mode_scales=mode_scales,
                rng=rng,
            )
            cruise = w[:, w.mean(axis=0) >= 45.0]
            if cruise.shape[1] < 5:
                continue
            gaps.append(np.min(np.diff(np.sort(cruise, axis=0), axis=0), axis=0))
        return float((np.concatenate(gaps) < 1.0).mean())

    assert degenerate_fraction(None) < 0.40
    assert degenerate_fraction({"roll": 0.45, "pitch": 0.45, "yaw": 1.7}) > 0.60


def test_band_taper_fades_the_comb_out_at_the_band_edge():
    """The comb used to stop dead at ``n_harmonics * rps`` — a cutoff linear in speed.

    ``n_harmonics`` was sized from the flight's HOVER speed, so every frame slower
    than hover ended its comb at a frequency exactly proportional to the rotor
    speed being predicted. That is a readout no real recording offers. On the
    built stream it left a +1.84 dB step at the cutoff once the spectrum's own
    tilt was differenced out, against +0.50 dB at the same frequency in real
    DREGON audio, and 100% of ramp frames carried one.

    The taper fades the line power to zero across the top of the band instead, so
    the top of the comb sits in the floor and carries no speed information.
    """
    import numpy as np

    from data_processing.stochastic_rotor_noise import build_psd, sample_params

    sr, n_harm = 16000, 80
    freqs = np.linspace(0.0, sr / 2.0, 513)
    rps = np.tile(np.linspace(20.0, 90.0, 64), (4, 1))  # a ramp, all four rotors

    def top_band_share(taper: float) -> float:
        params = sample_params(
            np.random.default_rng(3), n_harmonics=n_harm, band_taper_frac=taper, sample_rate=sr
        )
        psd = build_psd(params, rps, freqs, dt=0.02, rng=np.random.default_rng(3))
        lines = psd["lines"] if "lines" in psd else psd["psd"] - psd["floor"]
        top = freqs > 0.85 * sr / 2.0
        mid = (freqs > 0.4 * sr / 2.0) & (freqs < 0.55 * sr / 2.0)
        return float(lines[..., top].mean() / max(lines[..., mid].mean(), 1e-30))

    faded, hard = top_band_share(0.30), top_band_share(0.0)
    assert faded < 0.25 * hard, f"taper barely faded the band edge: {faded:.4f} vs {hard:.4f}"
