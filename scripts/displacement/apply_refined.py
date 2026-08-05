"""Add the pi_kalman-refined comb to the explorer: overlay + strips demodulated against it."""

import os

import hk_core as H
import numpy as np
from scipy.signal import resample_poly

R = np.load("refined_labels.npz")
ft = R["ft"]
r_init = R["r_init"]
r_ref = R["r_ref"]
RID = "free-flight_nosource_room1"
T0 = 22.56481
DUR = 16.0
audio, sr, g, rates = H.load_raw(RID, T0, DUR)
tfull = np.arange(audio.shape[1]) / sr
# refined trajectories back onto the audio grid
g_ref = np.stack([np.interp(tfull, ft, r_ref[r]) for r in range(4)])

DEC = 32
fs_env = sr / DEC
SEGS = [0.10, 0.50, 2.00]
NR = 76
YL = 6.0
NC_S = [110, 55, 22]
KS = np.arange(1, 101)
grid = np.linspace(-YL, YL, NR)
qs = {}
for r in range(4):
    phi = 2 * np.pi * np.cumsum(g_ref[r]) / sr  # REFINED carrier
    imgs = {}
    for si, seg in enumerate(SEGS):
        n = int(round(seg * fs_env))
        hp = max(n // 4, 1)
        ns = len(np.arange(0, int(DUR * fs_env) - n + 1, hp))
        imgs[si] = np.full((len(KS), NR, min(NC_S[si], ns)), np.nan, np.float32)
    for kj, k in enumerate(KS):
        z = np.stack(
            [resample_poly(audio[c] * np.exp(-1j * k * phi), 1, DEC) for c in range(audio.shape[0])]
        )
        for si, seg in enumerate(SEGS):
            n = int(round(seg * fs_env))
            hp = max(n // 4, 1)
            st = np.arange(0, z.shape[1] - n + 1, hp)
            NC = imgs[si].shape[2]
            pick = st[np.linspace(0, len(st) - 1, NC).astype(int)]
            win = np.hanning(n)
            fr = np.fft.fftshift(np.fft.fftfreq(n, 1 / fs_env))
            rev = fr / k
            sel = np.abs(rev) <= YL
            if sel.sum() < 3:
                continue
            sa = np.stack([z[:, s : s + n] * win for s in pick])
            P = (np.abs(np.fft.fftshift(np.fft.fft(sa, axis=-1), axes=-1)) ** 2).mean(1)[:, sel]
            db = 10 * np.log10(P + 1e-30)
            db -= np.median(db, axis=1, keepdims=True)
            imgs[si][kj] = np.stack(
                [np.interp(grid, rev[sel], db[j]) for j in range(db.shape[0])], axis=1
            )
    for si in range(3):
        img = imgs[si]
        lo = np.nanpercentile(img, 55, axis=(1, 2), keepdims=True)
        hi = np.nanpercentile(img, 99.5, axis=(1, 2), keepdims=True)
        qs[f"r{r}R{si}"] = np.nan_to_num(
            np.clip((img - lo) / np.maximum(hi - lo, 1e-6) * 255, 0, 255)
        ).astype(np.uint8)
    print("refined strips rotor", r, "ok", flush=True)
np.savez_compressed("f0v2_strips_ref.npz", **qs)
# refined trajectory on the spectrogram time grid
D = np.load("f0v2_data.npz")
t = D["t"]
gref_t = np.stack([np.interp(t, ft, r_ref[r]) for r in range(4)])
np.savez_compressed("f0v2_refined.npz", gref=gref_t.astype(np.float32))
print("MB", round(os.path.getsize("f0v2_strips_ref.npz") / 1e6, 2))
