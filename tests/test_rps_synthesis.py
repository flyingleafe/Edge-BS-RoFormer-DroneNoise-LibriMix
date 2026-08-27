def test_rotor_trim_breaks_the_unison_idle():
    """The differential modes are gated to cruise, so the four rotors idle in
    EXACT unison — the built stream's ramp frames had a spread of 0.00 rev/s
    with 100% of them inside 2 rev/s.

    One of the two rigs disagrees. On the frozen validation split Michael's ramp
    frames have a spread of 9.67 rev/s median and only 3.7% sit inside 2, while
    DREGON agrees with the stream (0.03 median, 83.0% inside 2). A per-rotor
    speed ratio, constant over a clip and drawn per clip, covers both.
    """
    import numpy as np

    from data_processing.rps_synthesis import generate_full_flight

    def ramp_spread(trim):
        out = []
        for seed in range(12):
            w = generate_full_flight(
                None, 200.0, rotor_trim_rel=trim, rng=np.random.default_rng(seed)
            )
            ramp = (w.max(axis=0) >= 1) & (w.mean(axis=0) < 45)
            if ramp.any():
                out.append(w[:, ramp].max(axis=0) - w[:, ramp].min(axis=0))
        return np.concatenate(out)

    assert float(np.median(ramp_spread(None))) < 0.5  # unison, as before
    spread = ramp_spread((0.0, 0.15))
    assert 2.0 < float(np.median(spread)) < 12.0  # between the two rigs
