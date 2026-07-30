"""Complex-KLA Triton op: fused scan + readout for :func:`models.ckla.complex_kla_scan`.

Structure mirrors the flat-KLA Triton op (kla-loglinear ``fkla/ops/flat_triton.py``)
wholesale: one program per (batch, D-block), the (N, BD) fp32 state — here the
complex pair (eta_re, eta_im) plus the real precision lam — lives in registers
across the whole T loop, with chunk-boundary state checkpoints every BT steps.
Backward recomputes each chunk forward into a per-program HBM stash, then walks
it in reverse accumulating adjoints.

Semantics == ``models.ckla.complex_kla_scan`` (the sequential fp32 reference):

    phi[t]   = k[t]^2 (N,1) * lam_v[t] (1,D)
    kappa[t] = k[t] * (lam_v[t] * v[t])          # kappa_im = 0
    den      = abar_mag^2 + pbar * lam           (per (N, D) cell)
    rot      = (cos w[t] + i sin w[t]) * eta     (per-slot rotation)
    eta     <- abar_mag * rot / den + kappa      (complex)
    lam     <- lam / den + phi                   (real — rotation never enters)
    y[t, d]  = sum_n q[t, n] * eta[n, d] / max(lam[n, d], eps)   (complex)

Gradient plumbing (identical to the base op): gk/gq — and the new per-step
rotation grads gc/gs — are accumulated per D-block into (B, nd, T, N) buffers
(each program only sees its d-slice of the readout/state sums) and reduced on
the host; gv/glv are per-cell and race-free; ga/gp come back as per-batch
(B, N, D) partials, summed on the host. No atomics anywhere.

fp32 state discipline throughout (inputs may be fp16/bf16; upcast on load).

The module is importable without triton: ``HAS_TRITON`` reports availability
and :func:`complex_kla_scan_triton` raises only at call time.
"""

from __future__ import annotations

from typing import cast

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised only on triton-less installs
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _fwd_kernel(
        K, Q, V, LV, COS, SIN, ABAR, PBAR, Y_RE, Y_IM,
        CKPT_ER, CKPT_EI, CKPT_L,
        T, D,
        N: tl.constexpr, BD: tl.constexpr, BT: tl.constexpr,
        EPS: tl.constexpr, NCHUNK,
    ):  # fmt: skip
        b = tl.program_id(0)
        pd = tl.program_id(1)
        offs_n = tl.arange(0, N)
        offs_d = pd * BD + tl.arange(0, BD)
        dmask = offs_d < D

        a = tl.load(ABAR + offs_n[:, None] * D + offs_d[None, :],
                    mask=dmask[None, :], other=1.0).to(tl.float32)  # fmt: skip
        p = tl.load(PBAR + offs_n[:, None] * D + offs_d[None, :],
                    mask=dmask[None, :], other=0.0).to(tl.float32)  # fmt: skip
        a2 = a * a

        eta_re = tl.zeros((N, BD), dtype=tl.float32)
        eta_im = tl.zeros((N, BD), dtype=tl.float32)
        lam = tl.zeros((N, BD), dtype=tl.float32)

        for t in range(0, T):
            if t % BT == 0:
                c = t // BT
                base = ((b * NCHUNK + c) * N + offs_n[:, None]) * D + offs_d[None, :]
                tl.store(CKPT_ER + base, eta_re, mask=dmask[None, :])
                tl.store(CKPT_EI + base, eta_im, mask=dmask[None, :])
                tl.store(CKPT_L + base, lam, mask=dmask[None, :])
            kv = tl.load(K + (b * T + t) * N + offs_n).to(tl.float32)  # (N,)
            qv = tl.load(Q + (b * T + t) * N + offs_n).to(tl.float32)
            cv = tl.load(COS + (b * T + t) * N + offs_n).to(tl.float32)
            sv = tl.load(SIN + (b * T + t) * N + offs_n).to(tl.float32)
            vv = tl.load(V + (b * T + t) * D + offs_d, mask=dmask, other=0.0).to(tl.float32)
            lv = tl.load(LV + (b * T + t) * D + offs_d, mask=dmask, other=0.0).to(tl.float32)

            phi = (kv * kv)[:, None] * lv[None, :]
            kap = kv[:, None] * (lv * vv)[None, :]
            den = a2 + p * lam
            r = 1.0 / den
            ar = a * r
            rot_re = cv[:, None] * eta_re - sv[:, None] * eta_im
            rot_im = cv[:, None] * eta_im + sv[:, None] * eta_re
            eta_re = ar * rot_re + kap
            eta_im = ar * rot_im
            lam = lam * r + phi

            lamc = tl.maximum(lam, EPS)
            y_re = tl.sum(qv[:, None] * (eta_re / lamc), axis=0)
            y_im = tl.sum(qv[:, None] * (eta_im / lamc), axis=0)
            tl.store(Y_RE + (b * T + t) * D + offs_d, y_re, mask=dmask)
            tl.store(Y_IM + (b * T + t) * D + offs_d, y_im, mask=dmask)

    @triton.jit
    def _bwd_kernel(
        K, Q, V, LV, COS, SIN, ABAR, PBAR,
        CKPT_ER, CKPT_EI, CKPT_L, GY_RE, GY_IM,
        GK, GQ, GC, GS, GV, GLV, GA_PART, GP_PART,
        STASH_ER, STASH_EI, STASH_L,
        T, D,
        N: tl.constexpr, BD: tl.constexpr, BT: tl.constexpr,
        EPS: tl.constexpr, NCHUNK,
    ):  # fmt: skip
        b = tl.program_id(0)
        pd = tl.program_id(1)
        offs_n = tl.arange(0, N)
        offs_d = pd * BD + tl.arange(0, BD)
        dmask = offs_d < D

        a = tl.load(ABAR + offs_n[:, None] * D + offs_d[None, :],
                    mask=dmask[None, :], other=1.0).to(tl.float32)  # fmt: skip
        p = tl.load(PBAR + offs_n[:, None] * D + offs_d[None, :],
                    mask=dmask[None, :], other=0.0).to(tl.float32)  # fmt: skip
        a2 = a * a

        g_eta_re = tl.zeros((N, BD), dtype=tl.float32)  # dL/d eta_t (through state)
        g_eta_im = tl.zeros((N, BD), dtype=tl.float32)
        g_lam = tl.zeros((N, BD), dtype=tl.float32)
        g_a = tl.zeros((N, BD), dtype=tl.float32)
        g_p = tl.zeros((N, BD), dtype=tl.float32)

        for ci in range(0, NCHUNK):
            c = NCHUNK - 1 - ci
            t0 = c * BT
            base = ((b * NCHUNK + c) * N + offs_n[:, None]) * D + offs_d[None, :]
            eta_re0 = tl.load(CKPT_ER + base, mask=dmask[None, :], other=0.0)
            eta_im0 = tl.load(CKPT_EI + base, mask=dmask[None, :], other=0.0)
            lam0 = tl.load(CKPT_L + base, mask=dmask[None, :], other=0.0)

            # ---- forward recompute of this chunk; per-step pre-states go to a
            # per-program HBM scratch (programs own disjoint (b, :, :, d) slices)
            eta_re = eta_re0
            eta_im = eta_im0
            lam = lam0
            for i in range(0, BT):
                t = t0 + i
                sbase = ((b * BT + i) * N + offs_n[:, None]) * D + offs_d[None, :]
                tl.store(STASH_ER + sbase, eta_re, mask=dmask[None, :])
                tl.store(STASH_EI + sbase, eta_im, mask=dmask[None, :])
                tl.store(STASH_L + sbase, lam, mask=dmask[None, :])
                inb = t < T
                kv = tl.load(K + (b * T + t) * N + offs_n, mask=inb & (offs_n >= 0),
                             other=0.0).to(tl.float32)  # fmt: skip
                cv = tl.load(COS + (b * T + t) * N + offs_n, mask=inb & (offs_n >= 0),
                             other=1.0).to(tl.float32)  # fmt: skip
                sv = tl.load(SIN + (b * T + t) * N + offs_n, mask=inb & (offs_n >= 0),
                             other=0.0).to(tl.float32)  # fmt: skip
                vv = tl.load(V + (b * T + t) * D + offs_d, mask=inb & dmask,
                             other=0.0).to(tl.float32)  # fmt: skip
                lv = tl.load(LV + (b * T + t) * D + offs_d, mask=inb & dmask,
                             other=0.0).to(tl.float32)  # fmt: skip
                phi = (kv * kv)[:, None] * lv[None, :]
                kap = kv[:, None] * (lv * vv)[None, :]
                den = a2 + p * lam
                r = 1.0 / den
                ar = a * r
                rot_re = cv[:, None] * eta_re - sv[:, None] * eta_im
                rot_im = cv[:, None] * eta_im + sv[:, None] * eta_re
                eta_re = ar * rot_re + kap
                eta_im = ar * rot_im
                lam = lam * r + phi

            # ---- reverse pass over the chunk ----
            for i in range(0, BT):
                j = BT - 1 - i
                t = t0 + j
                inb = t < T
                sbase = ((b * BT + j) * N + offs_n[:, None]) * D + offs_d[None, :]
                e_re = tl.load(STASH_ER + sbase, mask=dmask[None, :], other=0.0)
                e_im = tl.load(STASH_EI + sbase, mask=dmask[None, :], other=0.0)
                l_prev = tl.load(STASH_L + sbase, mask=dmask[None, :], other=0.0)

                kv = tl.load(K + (b * T + t) * N + offs_n, mask=inb & (offs_n >= 0),
                             other=0.0).to(tl.float32)  # fmt: skip
                qv = tl.load(Q + (b * T + t) * N + offs_n, mask=inb & (offs_n >= 0),
                             other=0.0).to(tl.float32)  # fmt: skip
                cv = tl.load(COS + (b * T + t) * N + offs_n, mask=inb & (offs_n >= 0),
                             other=1.0).to(tl.float32)  # fmt: skip
                sv = tl.load(SIN + (b * T + t) * N + offs_n, mask=inb & (offs_n >= 0),
                             other=0.0).to(tl.float32)  # fmt: skip
                vv = tl.load(V + (b * T + t) * D + offs_d, mask=inb & dmask,
                             other=0.0).to(tl.float32)  # fmt: skip
                lv = tl.load(LV + (b * T + t) * D + offs_d, mask=inb & dmask,
                             other=0.0).to(tl.float32)  # fmt: skip
                gy_re = tl.load(GY_RE + (b * T + t) * D + offs_d, mask=inb & dmask,
                                other=0.0).to(tl.float32)  # fmt: skip
                gy_im = tl.load(GY_IM + (b * T + t) * D + offs_d, mask=inb & dmask,
                                other=0.0).to(tl.float32)  # fmt: skip

                # recompute the step from the stashed PREV state
                phi = (kv * kv)[:, None] * lv[None, :]
                kap = kv[:, None] * (lv * vv)[None, :]
                den = a2 + p * l_prev
                r = 1.0 / den
                r2 = r * r
                ar = a * r
                rot_re = cv[:, None] * e_re - sv[:, None] * e_im
                rot_im = cv[:, None] * e_im + sv[:, None] * e_re
                eta_re_t = ar * rot_re + kap
                eta_im_t = ar * rot_im
                lam_t = l_prev * r + phi

                # readout adjoints: y = sum_n q_n * eta / max(lam, eps)
                lamc = tl.maximum(lam_t, EPS)
                mu_re = eta_re_t / lamc
                mu_im = eta_im_t / lamc
                gq = tl.sum(mu_re * gy_re[None, :] + mu_im * gy_im[None, :], axis=1)  # (N,)
                tl.store(GQ + ((b * tl.num_programs(1) + pd) * T + t) * N + offs_n,
                         gq, mask=inb & (offs_n >= 0))  # fmt: skip
                live = (lam_t > EPS).to(tl.float32)
                ge_re = g_eta_re + qv[:, None] * gy_re[None, :] / lamc
                ge_im = g_eta_im + qv[:, None] * gy_im[None, :] / lamc
                gl = g_lam - qv[:, None] * (gy_re[None, :] * eta_re_t
                                            + gy_im[None, :] * eta_im_t) \
                    / (lamc * lamc) * live  # fmt: skip

                # step adjoints
                g_kap = ge_re  # kappa_im = 0: the imaginary evidence path drops
                g_phi = gl
                # adjoint of the (a*r) factor: eta = (a*r)*rot + kap
                grot = rot_re * ge_re + rot_im * ge_im
                # d eta / d e_prev = (a*r) * R  ->  pull back through R^T
                g_eta_re = ar * (cv[:, None] * ge_re + sv[:, None] * ge_im)
                g_eta_im = ar * (-sv[:, None] * ge_re + cv[:, None] * ge_im)
                # d lam_t / d l_prev = r - l_prev*r^2*p ; eta path via den
                g_lam = (r - l_prev * r2 * p) * gl - a * r2 * p * grot
                # params
                g_a += (r - 2.0 * a2 * r2) * grot - 2.0 * a * l_prev * r2 * gl
                g_p += -a * r2 * l_prev * grot - l_prev * l_prev * r2 * gl
                # rotation phase: rot = R(c, s) e_prev, scaled by (a*r)
                gc = tl.sum(ar * (e_re * ge_re + e_im * ge_im), axis=1)  # (N,)
                gs = tl.sum(ar * (e_re * ge_im - e_im * ge_re), axis=1)
                tl.store(GC + ((b * tl.num_programs(1) + pd) * T + t) * N + offs_n,
                         gc, mask=inb & (offs_n >= 0))  # fmt: skip
                tl.store(GS + ((b * tl.num_programs(1) + pd) * T + t) * N + offs_n,
                         gs, mask=inb & (offs_n >= 0))  # fmt: skip

                # fan out to inputs: phi = k^2 lv ; kap = k lv v
                gk = tl.sum(2.0 * kv[:, None] * lv[None, :] * g_phi
                            + (lv * vv)[None, :] * g_kap, axis=1)  # fmt: skip
                glv = tl.sum((kv * kv)[:, None] * g_phi
                             + kv[:, None] * vv[None, :] * g_kap, axis=0)  # fmt: skip
                gv = tl.sum(kv[:, None] * lv[None, :] * g_kap, axis=0)
                tl.store(GK + ((b * tl.num_programs(1) + pd) * T + t) * N + offs_n,
                         gk, mask=inb & (offs_n >= 0))  # fmt: skip
                tl.store(GLV + (b * T + t) * D + offs_d, glv, mask=inb & dmask)
                tl.store(GV + (b * T + t) * D + offs_d, gv, mask=inb & dmask)

        base = (b * N + offs_n[:, None]) * D + offs_d[None, :]
        tl.store(GA_PART + base, g_a, mask=dmask[None, :])
        tl.store(GP_PART + base, g_p, mask=dmask[None, :])

    class _ComplexKLAFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, abar_mag, cos_w, sin_w, pbar, k, v, lam_v, q, eps, BD, BT):
            B, T, N = k.shape
            D = v.shape[-1]
            assert (N & (N - 1)) == 0, "N must be a power of two"
            nchunk = triton.cdiv(T, BT)
            dev = k.device
            f32 = torch.float32
            y_re = torch.empty(B, T, D, device=dev, dtype=f32)
            y_im = torch.empty(B, T, D, device=dev, dtype=f32)
            ckpt_er = torch.empty(B, nchunk, N, D, device=dev, dtype=f32)
            ckpt_ei = torch.empty(B, nchunk, N, D, device=dev, dtype=f32)
            ckpt_l = torch.empty(B, nchunk, N, D, device=dev, dtype=f32)
            abar_c = abar_mag.contiguous()
            pbar_c = pbar.contiguous()
            grid = (B, triton.cdiv(D, BD))
            _fwd_kernel[grid](
                k, q, v, lam_v, cos_w, sin_w, abar_c, pbar_c,
                y_re, y_im, ckpt_er, ckpt_ei, ckpt_l,
                T, D, N=N, BD=BD, BT=BT, EPS=eps, NCHUNK=nchunk,
            )  # fmt: skip
            ctx.save_for_backward(
                abar_c, cos_w, sin_w, pbar_c, k, v, lam_v, q, ckpt_er, ckpt_ei, ckpt_l
            )
            ctx.meta = (eps, BD, BT, nchunk)
            return y_re, y_im

        @staticmethod
        def backward(ctx, gy_re, gy_im):
            (abar_mag, cos_w, sin_w, pbar, k, v, lam_v, q,
             ckpt_er, ckpt_ei, ckpt_l) = ctx.saved_tensors  # fmt: skip
            eps, BD, BT, nchunk = ctx.meta
            B, T, N = k.shape
            D = v.shape[-1]
            dev = k.device
            f32 = torch.float32
            nd = triton.cdiv(D, BD)
            gk = torch.zeros(B, nd, T, N, device=dev, dtype=f32)
            gq = torch.zeros(B, nd, T, N, device=dev, dtype=f32)
            gc = torch.zeros(B, nd, T, N, device=dev, dtype=f32)
            gs = torch.zeros(B, nd, T, N, device=dev, dtype=f32)
            stash_er = torch.empty(B, BT, N, D, device=dev, dtype=f32)
            stash_ei = torch.empty(B, BT, N, D, device=dev, dtype=f32)
            stash_l = torch.empty(B, BT, N, D, device=dev, dtype=f32)
            gv = torch.zeros(B, T, D, device=dev, dtype=f32)
            glv = torch.zeros(B, T, D, device=dev, dtype=f32)
            ga = torch.zeros(B, N, D, device=dev, dtype=f32)
            gp = torch.zeros(B, N, D, device=dev, dtype=f32)
            _bwd_kernel[(B, nd)](
                k, q, v, lam_v, cos_w, sin_w, abar_mag, pbar,
                ckpt_er, ckpt_ei, ckpt_l, gy_re.contiguous(), gy_im.contiguous(),
                gk, gq, gc, gs, gv, glv, ga, gp,
                stash_er, stash_ei, stash_l,
                T, D, N=N, BD=BD, BT=BT, EPS=eps, NCHUNK=nchunk,
            )  # fmt: skip
            return (
                ga.sum(0).to(abar_mag.dtype),
                gc.sum(1).to(cos_w.dtype), gs.sum(1).to(sin_w.dtype),
                gp.sum(0).to(pbar.dtype),
                gk.sum(1).to(k.dtype), gv.to(v.dtype), glv.to(lam_v.dtype),
                gq.sum(1).to(q.dtype), None, None, None,
            )  # fmt: skip


def complex_kla_scan_triton(
    abar_mag: torch.Tensor,
    cos_w: torch.Tensor,
    sin_w: torch.Tensor,
    pbar: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lam_v: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-6,
    BD: int = 32,
    BT: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused complex-KLA scan + readout. Returns (y_re, y_im), each (B, T, D) fp32.

    Same argument order and semantics as ``models.ckla.complex_kla_scan``
    (default ``eps`` matches its 1e-6, not the base real-KLA op's 1e-8).
    Requires triton (``HAS_TRITON``); raises at call time when unavailable.
    """
    if not HAS_TRITON:  # pragma: no cover - exercised only on triton-less installs
        raise RuntimeError(
            "complex_kla_scan_triton requires triton; it is not installed (HAS_TRITON is False)"
        )
    cos_w, sin_w, k, v, lam_v, q = (x.contiguous() for x in (cos_w, sin_w, k, v, lam_v, q))
    out = _ComplexKLAFn.apply(abar_mag, cos_w, sin_w, pbar, k, v, lam_v, q, eps, BD, BT)
    return cast(tuple[torch.Tensor, torch.Tensor], out)
