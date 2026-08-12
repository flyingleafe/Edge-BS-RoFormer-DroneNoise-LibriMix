"""Unit tests for the importable core of ``scripts/eval_gen_comb_real.py``.

Everything here is torch-free and model-free: the spectral reading, the cell
admission, the floor exclusion and the tidy-output helpers. The two claims that
matter are the instrument's null (a comb read against itself must read flat) and
its power (a displaced comb must read differently), because a statistic that
fails either cannot decide the label A/B this script exists for.
"""

from __future__ import annotations

from types import SimpleNamespace

import eval_gen_comb_real as E
import numpy as np
import pytest

SR = 16000


def _comb(f0: float, k_max: int, seconds: float, *, noise: float, seed: int = 0) -> np.ndarray:
    """One microphone of a static comb plus white noise — ``(mic, time)``."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    x = np.zeros_like(t)
    for k in range(1, k_max + 1):
        if k * f0 < 0.45 * SR:
            x += np.sin(2.0 * np.pi * k * f0 * t + rng.uniform(0, 2 * np.pi))
    return (x + noise * rng.standard_normal(t.size))[None, :]


def _rates(f0: float, n_frames: int, n_rotors: int = 1) -> np.ndarray:
    return np.full((n_rotors, n_frames), f0, dtype=np.float64)


# --- spectra ---------------------------------------------------------------


def test_stft_power_shapes_and_frame_centres() -> None:
    x = np.zeros((3, 8000))
    power, freqs = E.stft_power(x, n_fft=2048, hop=512, sr=SR)
    n_frames = 1 + (8000 - 2048) // 512
    assert power.shape == (3, n_frames, 1025)
    assert freqs.shape == (1025,)
    assert freqs[1] - freqs[0] == pytest.approx(SR / 2048)
    centres = E.frame_centres(n_frames, n_fft=2048, hop=512)
    assert centres[0] == 1024 and centres[1] == 1024 + 512


def test_stft_power_rejects_a_signal_shorter_than_the_window() -> None:
    with pytest.raises(ValueError, match="shorter than n_fft"):
        E.stft_power(np.zeros(100), n_fft=2048, hop=512, sr=SR)


def test_rates_at_frames_samples_the_window_centre() -> None:
    rps = np.arange(4096, dtype=np.float64)[None, :]
    got = E.rates_at_frames(rps, 3, n_fft=2048, hop=512)
    assert got[0].tolist() == [1024.0, 1536.0, 2048.0]


def test_frequency_scaled_moves_a_tone_by_the_scale() -> None:
    t = np.arange(SR) / SR
    tone = np.sin(2.0 * np.pi * 1000.0 * t)[None, :]
    moved = E.frequency_scaled(tone, 0.9)
    power, freqs = E.stft_power(moved[:, :8192], n_fft=8192, hop=8192, sr=SR)
    peak = float(freqs[int(np.argmax(power[0, 0]))])
    assert peak == pytest.approx(900.0, abs=5.0)


# --- the line reading ------------------------------------------------------


def test_line_table_finds_a_comb_and_noise_alone_reads_at_the_null() -> None:
    """The power and the null of readout 2/3, on signals built to have each."""
    ks = np.arange(1, 41)
    kw = {"ks": ks, "sr": SR, "stride": 4, "gate_frac": 0.0}

    def _ptf(audio: np.ndarray) -> tuple[float, float]:
        power, freqs = E.stft_power(audio, n_fft=E.N_FFT, hop=E.HOP, sr=SR)
        tab = E.line_table(power, freqs, _rates(75.0, power.shape[1]), **kw)  # type: ignore[arg-type]
        got = [np.mean(tab.ptf_db(k)) for k in range(2, 40) if k in tab.band]
        null = [
            E.estimator_null_db(tab.n_band_bins[k], tab.n_floor_bins[k])
            for k in range(2, 40)
            if k in tab.band
        ]
        return float(np.mean(got)), float(np.mean(null))

    on_db, _ = _ptf(_comb(75.0, 40, 2.0, noise=0.5))
    assert on_db > 10.0  # a real tooth stands well clear of its valley

    rng = np.random.default_rng(1)
    noise_db, null_db = _ptf(rng.standard_normal((1, 2 * SR)))
    assert abs(noise_db - null_db) < 1.0  # and line-free audio lands on the null


def test_estimator_null_is_negative_and_shrinks_with_more_floor_bins() -> None:
    """The null is NOT at 0 dB, and it is below it.

    The sample median of a few exponential bins sits ABOVE the distribution's
    median, so dividing by ln 2 over-estimates the floor and the ratio reads low.
    More floor bins, less bias — which is why the two valley slots are unioned.
    """
    coarse = E.estimator_null_db(5.0, 5.0)
    fine = E.estimator_null_db(5.0, 40.0)
    assert -1.5 < coarse < -0.1
    assert abs(fine) < abs(coarse)


def test_line_table_is_deterministic() -> None:
    audio = _comb(75.0, 20, 1.0, noise=0.5)
    power, freqs = E.stft_power(audio, n_fft=E.N_FFT, hop=E.HOP, sr=SR)
    rates = _rates(75.0, power.shape[1])
    ks = np.arange(1, 21)
    kw = {"ks": ks, "sr": SR, "stride": 4, "gate_frac": 0.0}
    a = E.line_table(power, freqs, rates, **kw)  # type: ignore[arg-type]
    b = E.line_table(power, freqs, rates, **kw)  # type: ignore[arg-type]
    for k in a.line:
        assert np.array_equal(a.line[k], b.line[k])
        assert a.count[k] == b.count[k]


def test_line_table_geometry_follows_the_reference_not_the_candidate() -> None:
    """The fixed-degrees-of-freedom rule: a displaced candidate gets the SAME cells.

    This is the bug that made a 0.542 % label scale read as a spurious line-power
    gain — the candidate was being gated on its own carriers.
    """
    audio = _comb(75.0, 40, 2.0, noise=0.5)
    power, freqs = E.stft_power(audio, n_fft=E.N_FFT, hop=E.HOP, sr=SR)
    rates = _rates(75.0, power.shape[1], n_rotors=2)
    rates[1] *= 1.09  # a second, resolvable rotor, so the gate has work to do
    ks = np.arange(1, 41)
    kw = {"ks": ks, "sr": SR, "stride": 4, "gate_frac": 0.3}
    base = E.line_table(power, freqs, rates, **kw)  # type: ignore[arg-type]
    moved = E.line_table(power, freqs, rates * 0.99, ref_rates=rates, **kw)  # type: ignore[arg-type]
    assert base.count == moved.count


def test_line_table_drops_cells_past_nyquist() -> None:
    audio = _comb(75.0, 10, 1.0, noise=0.5)
    power, freqs = E.stft_power(audio, n_fft=E.N_FFT, hop=E.HOP, sr=SR)
    rates = _rates(75.0, power.shape[1])
    tab = E.line_table(power, freqs, rates, ks=np.arange(1, 121), sr=SR, stride=8, gate_frac=0.0)
    assert 110 not in tab.line  # 110 * 75 Hz = 8250 Hz, past Nyquist
    assert 20 in tab.line


def test_admit_cells_gate_is_off_by_default_and_bites_when_asked() -> None:
    rates = np.array([80.0, 80.4])  # a near-twin pair, as DREGON flies
    ks = np.arange(1, 41)
    ungated = E.admit_cells(rates, ks, sr=SR, df=7.8125, gate_frac=0.0)
    gated = E.admit_cells(rates, ks, sr=SR, df=7.8125, gate_frac=0.25)
    assert ungated.all()
    assert gated.sum() < ungated.sum()


def test_clean_floor_mask_excises_a_foreign_line() -> None:
    freqs = np.arange(0.0, 200.0, 5.0)
    plain = E.clean_floor_mask(freqs, 100.0, 20.0, np.array([]), 10.0)
    assert plain.sum() == 9  # 80..120 inclusive on a 5 Hz grid
    cut = E.clean_floor_mask(freqs, 100.0, 20.0, np.array([110.0]), 10.0)
    assert cut.sum() < plain.sum()
    assert not cut[np.argmin(np.abs(freqs - 110.0))]
    # A foreign line far outside the slot changes nothing.
    assert np.array_equal(E.clean_floor_mask(freqs, 100.0, 20.0, np.array([5.0]), 10.0), plain)


# --- readout 1 -------------------------------------------------------------


def test_track_delta_db_is_zero_against_itself_and_positive_when_displaced() -> None:
    audio = _comb(75.0, 40, 2.0, noise=0.5)
    power, _ = E.stft_power(audio, n_fft=E.N_FFT, hop=E.HOP, sr=SR)
    rates = _rates(75.0, power.shape[1])
    ks = np.arange(1, 41)
    null = E.track_delta_db(power, power, rates, ks=ks, sr=SR)
    assert max(float(np.max(np.abs(v))) for v in null.values()) == 0.0
    moved, _ = E.stft_power(E.frequency_scaled(audio, 0.97), n_fft=E.N_FFT, hop=E.HOP, sr=SR)
    sens = E.track_delta_db(
        moved, power[:, : moved.shape[1]], rates[:, : moved.shape[1]], ks=ks, sr=SR
    )
    assert float(np.mean(sens[40])) > float(np.mean(sens[2]))  # worse the higher the harmonic


# --- config / output plumbing ---------------------------------------------


def _cfg(**params: object) -> object:
    return SimpleNamespace(data=SimpleNamespace(train=SimpleNamespace(params=params)))


def test_label_variant_reads_the_arm_s_own_config() -> None:
    assert E.label_variant_of(_cfg()) == ("orig", 1.0)
    assert E.label_variant_of(_cfg(dregon_rps_scale=0.99458)) == ("scaled", 0.99458)
    assert E.label_variant_of(_cfg(dregon_rps_override_dir="a/b")) == ("refined", 1.0)
    # A config with no data block at all (a non-noise-gen experiment) is "orig".
    assert E.label_variant_of(SimpleNamespace()) == ("orig", 1.0)


def test_resolve_checkpoint_prefers_an_explicit_path_then_falls_back_to_r2() -> None:
    assert E.resolve_checkpoint("whatever", "/tmp/x.ckpt") == "/tmp/x.ckpt"
    got = E.resolve_checkpoint("a_run_that_does_not_exist_locally")
    assert got.startswith("r2://") and "a_run_that_does_not_exist_locally" in got


def test_chunk_label_scales_the_original_track() -> None:
    orig = np.ones((4, 10), dtype=np.float32) * 80.0
    chunk = E.Chunk(
        index=0,
        recording_id="r",
        t_rel=0.0,
        split="train",
        mean_rps=80.0,
        audio=np.zeros((1, 10)),
        labels={"orig": orig},
        mic_pos=np.zeros((1, 3)),
        rotor_pos=np.zeros((4, 3)),
        sample_rate=SR,
    )
    assert chunk.label("orig", 1.0)[0, 0] == pytest.approx(80.0)
    assert chunk.label("scaled", 0.99458)[0, 0] == pytest.approx(80.0 * 0.99458)
    with pytest.raises(KeyError):
        chunk.label("refined", 1.0)


def test_band_of_covers_1_to_80_and_nothing_else() -> None:
    assert [E.band_of(k) for k in (1, 9, 10, 24, 25, 49, 50, 80)] == [
        "k1-9",
        "k1-9",
        "k10-24",
        "k10-24",
        "k25-49",
        "k25-49",
        "k50-80",
        "k50-80",
    ]
    assert E.band_of(81) is None


def _row(arm: str, k: int, value: float) -> dict[str, object]:
    row: dict[str, object] = dict.fromkeys(E._ROW_FIELDS, 0.0)
    row.update(arm=arm, label_variant="orig", k=k, band=E.band_of(k), delta_logmag_db=value)
    return row


def test_summarize_averages_per_arm_and_band_and_skips_missing_values() -> None:
    rows = [_row("a", 1, 2.0), _row("a", 2, 4.0), _row("a", 50, 9.0), _row("b", 1, 1.0)]
    rows[1]["line_delta_db"] = None  # a cell the gate never admitted
    summary = E.summarize(rows)
    by_key = {(r["arm"], r["band"]): r for r in summary}
    assert by_key[("a", "k1-9")]["delta_logmag_db"] == pytest.approx(3.0)
    assert by_key[("a", "k50-80")]["delta_logmag_db"] == pytest.approx(9.0)
    assert by_key[("b", "k1-9")]["n_rows"] == 1
    assert by_key[("a", "k1-9")]["line_delta_db"] == pytest.approx(0.0)
    # bands stay in their declared order per arm
    assert [r["band"] for r in summary if r["arm"] == "a"] == ["k1-9", "k50-80"]


def test_write_rows_round_trips_through_csv(tmp_path) -> None:
    import csv

    path = tmp_path / "per_k.csv"
    E.write_rows(path, [_row("a", 1, 2.0)])
    with path.open() as fh:
        got = list(csv.DictReader(fh))
    assert len(got) == 1
    assert got[0]["arm"] == "a"
    assert set(got[0]) == set(E._ROW_FIELDS)
