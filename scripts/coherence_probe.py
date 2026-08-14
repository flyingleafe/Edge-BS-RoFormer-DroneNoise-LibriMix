"""Coherence probe v3: class-mean correlations of LOW-PASSED k-normalized phase increments.

Raw 100 Hz increments are dominated by flat measurement noise (differentiation boosts
high frequencies); the true wander lives below ~2 Hz. Low-pass with a boxcar, then
compute pair-class mean correlations:
  shaft share    ~= mean corr(cross-mic, same rotor)
  arrival share  ~= corr(within-mic, same rotor) - shaft share
  rig common     ~= corr(cross-rotor)  (battery/ESC common-mode; subtracted context)
Ratio estimates via class-mean corr are robust to small track counts.
SNR cut is per rotor (weight > 0.05 * rotor max).
"""

import sys

import numpy as np

RID = sys.argv[1]
K_USE = 9
LP_S = [0.5, 2.0]  # boxcar lengths, seconds

d = np.load(f"results/vk_decompose_v2/{RID}/envelopes.npz")
amp, ph = d["amp"], d["phase_err"]
rotor, kk, valid = d["rotor"], d["k"], d["valid"]
fs = float(d["fs_env"])
C = amp.shape[0]

selall = np.where(kk <= K_USE)[0]
v = valid[selall].all(axis=0)

rows, meta, wts = [], [], []
for j in selall:
    for c in range(C):
        p = np.unwrap(ph[c, j])[v]
        dD = np.diff(p) * fs / (2 * np.pi * kk[j])
        rows.append(dD)
        meta.append((int(rotor[j]), int(kk[j]), c))
        wts.append(float((amp[c, j][v].astype(np.float64) ** 2).mean()))
D0 = np.array(rows)
meta = np.array(meta)
wts = np.array(wts)
R, K, MC = meta[:, 0], meta[:, 1], meta[:, 2]

keep = np.zeros(len(wts), bool)
for r in np.unique(R):
    m = r == R
    keep[m] = wts[m] > 0.05 * wts[m].max()
D0, meta, wts = D0[keep], meta[keep], wts[keep]
R, K, MC = meta[:, 0], meta[:, 1], meta[:, 2]
kept_desc = {int(r): sorted(set(K[r == R])) for r in np.unique(R)}
print(f"{RID}: kept {keep.sum()}/{keep.size} tracks; k per rotor: {kept_desc}")

for lp in LP_S:
    n = int(lp * fs)
    ker = np.ones(n) / n
    D = np.array([np.convolve(x, ker, mode="valid") for x in D0])[:, ::n]  # decimate too
    Dz = D - D.mean(axis=1, keepdims=True)
    sd = Dz.std(axis=1) + 1e-15
    Cm = (Dz / sd[:, None]) @ (Dz / sd[:, None]).T / Dz.shape[1]
    pw = np.sqrt(np.outer(wts, wts))
    iu = np.triu_indices(len(D), 1)

    def avg(mask, Cm=Cm, pw=pw, iu=iu):
        m = mask[iu]
        if m.sum() == 0:
            return float("nan"), 0
        return float((Cm[iu][m] * pw[iu][m]).sum() / pw[iu][m].sum()), int(m.sum())

    same_r = R[:, None] == R[None, :]
    same_c = MC[:, None] == MC[None, :]
    same_k = K[:, None] == K[None, :]
    xmic, n1 = avg(same_r & ~same_c & ~same_k)
    wmic, n2 = avg(same_r & same_c & ~same_k)
    xrot, n3 = avg(~same_r & ~same_c)
    xrot_sm, n4 = avg(~same_r & same_c)
    tot_rms = float(np.sqrt(np.mean(D.var(axis=1))))
    print(f"\nLP={lp:.1f}s  (total lowpassed wander rms {tot_rms:.3f} Hz per unit k)")
    print(
        f"  shaft (crossmic sr)  {xmic:+.3f} [n={n1}]   sh+arr (withinmic sr) {wmic:+.3f} [n={n2}]"
    )
    print(
        f"  rig common (crossrot) {xrot:+.3f} [n={n3}]   crossrot samemic {xrot_sm:+.3f} [n={n4}]"
    )
    print(
        f"  => shares of lowpassed variance: rig-common~{max(xrot, 0):.2f} "
        f"rotor-shaft~{max(xmic - max(xrot, 0), 0):.2f} arrival~{max(wmic - xmic, 0):.2f} "
        f"incoherent~{1 - max(wmic, 0):.2f}"
    )
