"""Why does the comb family read a stopped rotor and the stochastic family not?

The observation the campaign cannot explain: `m3abl_comb_*_s1` sees 1.0% zero
frames and carries no silence source, and reads a real stopped-rotor clip at
4.73 rev/s. The stochastic arms see about 19% zeros INCLUDING a dedicated
silence pool, and read the same clips at 20 to 28.

The hypothesis this script tests is that the silence pool backfired. It is a
different generative family — room tone, colored noise, low-frequency rumble —
so "rotors off" became recognizable by TEXTURE rather than by the absence of a
comb. A real stopped-rotor clip is the same room and the same microphones as the
cruise clips around it, so it looks like the drone family with its comb removed,
which is what a ramp window also looks like.

Six inputs, each fed to every model, all at the same level:

    digital_silence   exactly zero samples
    stoch_floor       the stochastic family's own floor, rotors at zero
    stoch_lowspeed    the same family with rotors at 12 rev/s
    silence_pool      a chunk from the silence source (room tone and friends)
    comb_floor        the comb family's floor with rotors at zero
    real_zero         a real stopped-rotor clip from the frozen split

A model that reads "no comb" answers near zero on the first five. A model that
learned the silence pool's texture answers near zero on `silence_pool` and
`digital_silence` only, and calls the two combless drone floors a ramp.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if str(Path(__file__).resolve().parents[1] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SR = 16000
DURATION = 8.0
LEVEL = 0.0075  # the measured level of a real stopped-rotor clip


def build_inputs(seed: int = 0) -> dict[str, np.ndarray]:
    from data_processing import stochastic_rotor_noise as srn
    from data_processing.rotor_spectral_model import StaticCombNoisePool
    from data_processing.silence_noise import SilenceNoisePool

    n = int(DURATION * SR)
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {"digital_silence": np.zeros(n, dtype=np.float64)}

    # The stochastic family's own floor, with and without a turning rotor.
    for name, speed in (("stoch_floor", 0.0), ("stoch_lowspeed", 12.0)):
        params = srn.sample_params(np.random.default_rng(seed), n_rotors=4, n_harmonics=80)
        params = params.with_(floor_static_rel=0.06)
        rps = np.full((4, n), speed)
        audio, _ = srn.synthesize(
            params, rps, rng=np.random.default_rng(seed + 1), n_mics=1, normalize_rms=None
        )
        out[name] = audio[0].astype(np.float64)

    pool = SilenceNoisePool(sample_rate=SR, duration_s=DURATION, n_channels=1)
    out["silence_pool"] = np.asarray(
        pool.sample_timeframe(rng, DURATION)["audio"].data, dtype=np.float64
    ).reshape(-1)

    comb = StaticCombNoisePool(sample_rate=SR, duration_s=DURATION, n_mics=1, n_harmonics=100)
    audio, _, _ = comb.render(np.random.default_rng(seed + 2), DURATION)
    out["comb_floor"] = np.zeros(n, dtype=np.float64)  # its zero window IS silence
    del audio
    return out


def real_zero_clip() -> np.ndarray | None:
    from data_processing.frame_datasets import DregonLMFrameDataset

    ds = DregonLMFrameDataset(
        data_dir="dload:DREGON-LM-V4-michaels-valid-full",
        n_fft=2048,
        hop_length=512,
        sample_rate=SR,
        channel=0,
    )
    for i in range(len(ds)):
        frame = ds[i]
        target = np.asarray(frame["rps"].data, dtype=np.float64)
        if (target.max(axis=0) < 1.0).mean() > 0.9:
            return np.asarray(frame["mixture"].data, dtype=np.float64).reshape(-1)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import tdseries as td
    import torch

    import zoo
    from data_processing.frames import audio_series

    inputs = build_inputs()
    real = real_zero_clip()
    if real is not None:
        inputs["real_zero"] = real

    # One level for all of them, so level cannot be the explanation.
    for name, x in inputs.items():
        rms = float(np.sqrt(np.mean(np.square(x))))
        if rms > 0:
            inputs[name] = x / rms * LEVEL

    names = list(inputs)
    print(f"{'model':26s} " + " ".join(f"{n[:13]:>14s}" for n in names), flush=True)
    rows = []
    for experiment in args.exp:
        try:
            model = zoo.load(experiment, ckpt="best", device="cpu")
        except Exception as exc:  # noqa: BLE001
            print(f"{experiment}: FAILED ({exc!r})", flush=True)
            continue
        vals = []
        for name in names:
            frame = td.Frame({"mixture": audio_series(inputs[name].astype(np.float32)[None], SR)})
            with torch.no_grad():
                pred = np.asarray(model(frame)["rps_pred"].data, dtype=np.float64)
            vals.append(float(pred.mean()))
        print(f"{experiment:26s} " + " ".join(f"{v:14.2f}" for v in vals), flush=True)
        rows.append({"experiment": experiment, **dict(zip(names, vals, strict=True))})
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
