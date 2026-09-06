"""Joint Harmonic Trajectory Refinement (offline, mono, 16 kHz).

The magnitude locator and the differentiable waveform reader are separate evidence
paths. A conditional call replaces only the locator's slot initialization; it does
not replace the trainable global magnitude memory. Six untied blocks re-read local
complex observations and jointly update continuous trajectories. No loss, activity
classifier, hard track assignment, or seed-relative correction tube lives here.
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from models.comb_salience import CombGather, local_floor_torch
from tracking.dsp import analytic_signal_tensor, demodulate_trajectories

_BANDWIDTHS = (8.0, 32.0, 128.0)
_LAGS = (1, 4, 16, 64)
_MAX_RATE = 150.0


def _sample_spectrum(
    spectrum: Tensor, frequencies: Tensor, sample_rate: int, n_fft: int
) -> tuple[Tensor, Tensor]:
    """Linear reads at (B, ..., T); invalid positions never alias a real bin."""
    b, nf, t = spectrum.shape
    valid = (frequencies > 0) & (frequencies < sample_rate / 2)
    position = frequencies * (n_fft / sample_rate)
    lo = position.floor().long().clamp(0, nf - 2)
    fraction = (position - lo).clamp(0, 1)
    flat = lo.reshape(b, -1, t)
    left = spectrum.gather(1, flat).reshape_as(frequencies)
    right = spectrum.gather(1, flat + 1).reshape_as(frequencies)
    return (left + fraction * (right - left)) * valid, valid


def _modal_rates(logits: Tensor, offset_logits: Tensor, candidates: Tensor) -> Tensor:
    """Hard categorical forward, documented softmax straight-through backward.

    Offset logits broadcast over candidates; the OFF candidate always has zero
    offset. This is deliberately not a probability-weighted forward centroid.
    """
    probabilities = logits.float().softmax(-1)
    mode = F.one_hot(probabilities.argmax(-1), candidates.numel()).to(probabilities)
    weights = probabilities + (mode - probabilities).detach()
    values = candidates + 0.5 * offset_logits.float().tanh() * (candidates > 0)
    return (weights * values).sum(-1).clamp(0, _MAX_RATE)


def _interloper_geometry(
    rates: Tensor, orders: Tensor, *, joint_slots: bool, sample_rate: int = 16000
) -> Tensor:
    """Signed tooth geometry (B, R, K, T, source-R, bandwidth, 9).

    Arithmetic floor/ceil finds nearest teeth and counts without enumerating an
    interloper bank: foreign orders may exceed the reader's k_max. Own central
    teeth are excluded, but own adjacent teeth remain. Sources <=0.5 rev/s are
    absent. Identity-free per-source encoding/pooling preserves row equivariance.
    Nearest offsets are normalized by bandwidth and clipped to [-1, 1]; separate
    existence flags distinguish a missing side from a distant tooth.
    """
    b, r, t = rates.shape
    k = orders.numel()
    centre = rates[:, :, None, :, None, None] * orders[None, None, :, None, None, None]
    source = rates[:, None, None, :, :, None].transpose(3, 4)
    own = torch.eye(r, dtype=torch.bool, device=rates.device)[None, :, None, None, :, None]
    active = (source > 0.5) & (rates[:, :, None, :, None, None] > 0.5)
    if not joint_slots:
        active = active & own
    safe = torch.where(source > 0.5, source, torch.ones_like(source))
    nyquist = sample_rate / 2 - 1e-3
    highest = torch.floor(nyquist / safe)
    lower_order = torch.floor(centre / safe)
    upper_order = torch.ceil(centre / safe).clamp_min(1)
    lower_order = torch.where(own, orders[None, None, :, None, None, None] - 1, lower_order)
    upper_order = torch.where(own, orders[None, None, :, None, None, None] + 1, upper_order)
    lower_ok = (lower_order >= 1) & (lower_order <= highest) & active
    upper_ok = (upper_order >= 1) & (upper_order <= highest) & active
    widths = rates.new_tensor(_BANDWIDTHS).view(1, 1, 1, 1, 1, -1)
    first = torch.ceil((centre - widths) / safe).clamp_min(1)
    last = torch.minimum(torch.floor((centre + widths) / safe), highest)
    count = (last - first + 1).clamp_min(0)
    count = (count - own.to(count.dtype)).clamp_min(0) * active
    shape = (b, r, k, t, r, len(_BANDWIDTHS))
    features = (
        ((lower_order * safe - centre) / widths).clamp(-1, 1) * lower_ok,
        ((upper_order * safe - centre) / widths).clamp(-1, 1) * upper_ok,
        count.log1p(),
        lower_ok,
        upper_ok,
        active,
        own,
        widths / 128,
        (source - rates[:, :, None, :, None, None]) / _MAX_RATE,
    )
    return torch.stack([value.expand(shape).to(rates.dtype) for value in features], -1)


class _AttentionFF(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x: Tensor) -> Tensor:
        normalized = self.norm(x)
        x = x + self.attention(normalized, normalized, normalized, need_weights=False)[0]
        return x + self.ff(self.ff_norm(x))


class _CandidateRead(nn.Module):
    """Cross-attention with explicit signed candidate-minus-current coordinates."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(width)
        self.memory = nn.Linear(64, width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.relative = nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, heads))
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, state: Tensor, memory: Tensor, rates: Tensor, candidates: Tensor) -> Tensor:
        b, r, t, d = state.shape
        query = self.norm(state).transpose(1, 2).reshape(b * t, r, d)
        keys = self.memory(memory).reshape(b * t, candidates.numel(), d)
        delta = (candidates[None, None, None, :] - rates.transpose(1, 2)[..., None]) / _MAX_RATE
        bias = self.relative(torch.stack((delta, delta.abs()), -1))
        bias = bias.reshape(b * t, r, candidates.numel(), self.heads).permute(0, 3, 1, 2)
        bias = bias.reshape(b * t * self.heads, r, candidates.numel()).to(query.dtype)
        read = self.attention(query, keys, keys, attn_mask=bias, need_weights=False)[0]
        state = state + read.reshape(b, t, r, d).transpose(1, 2)
        return state + self.ff(self.ff_norm(state))


class _CoarseMemory(nn.Module):
    """Order-axis encoder over centre, near-tooth and half-order gap reads."""

    def __init__(self, n_fft: int, sample_rate: int, k_max: int):
        super().__init__()
        self.n_fft, self.sample_rate = n_fft, sample_rate
        # Candidate chunks keep K x G x T x feature intermediates off the full
        # batch allocation path. All chunks share the same learned encoder.
        self.gathers = nn.ModuleList(
            [
                CombGather(
                    grid=torch.arange(start, min(start + 15, 151)),
                    k_max=k_max,
                    sr=sample_rate,
                    n_fft=n_fft,
                    f_max=sample_rate / 2,
                )
                for start in range(1, 151, 15)
            ]
        )
        self.order_encoder = nn.Sequential(
            nn.Conv2d(18, 32, (3, 1), padding=(1, 0)),
            nn.GELU(),
            nn.Conv2d(32, 64, (3, 1), padding=(1, 0)),
            nn.GELU(),
        )
        self.coordinates = nn.Linear(3, 64)
        self.off_memory = nn.Parameter(torch.zeros(64))

    def _chunk(self, power: Tensor, floor: Tensor, gather: CombGather, times: Tensor) -> Tensor:
        b, _, t = power.shape
        rates = cast(Tensor, gather.grid).to(power)
        orders = torch.arange(1, gather.k_max + 1, device=power.device, dtype=power.dtype)
        centre = orders[:, None] * rates[None, :]
        df = self.sample_rate / self.n_fft
        positions = (
            centre - df,
            centre,
            centre + df,
            (orders[:, None] - 0.5) * rates,
            (orders[:, None] + 0.5) * rates,
        )
        centre_power, centre_floor = gather(power), gather(floor)
        channels = []
        for index, position in enumerate(positions):
            frequency = position[None, ..., None].expand(b, -1, -1, t)
            background, valid = _sample_spectrum(floor, frequency, self.sample_rate, self.n_fft)
            if index == 1:
                value, background = centre_power, centre_floor
            else:
                value, _ = _sample_spectrum(power, frequency, self.sample_rate, self.n_fft)
            channels.extend(
                (
                    (value / background.clamp_min(1e-12)).log1p() * valid,
                    background.log1p() * valid,
                    valid.to(power.dtype),
                )
            )
        shape = (b, orders.numel(), rates.numel(), t)
        channels.extend(
            (
                (rates[None, None, :, None] / _MAX_RATE).expand(shape),
                (orders[None, :, None, None] / 32).expand(shape),
                times[None, None, None, :].expand(shape),
            )
        )
        features = torch.stack(channels, 1).permute(0, 3, 1, 2, 4)
        encoded = self.order_encoder(features.reshape(b * rates.numel(), 18, orders.numel(), t))
        return encoded.mean(2).reshape(b, rates.numel(), 64, t).permute(0, 3, 1, 2)

    def forward(
        self, power: Tensor, floor: Tensor, times: Tensor, energy: Tensor, use_checkpoint: bool
    ) -> Tensor:
        chunks = []
        for module in self.gathers:
            gather = cast(CombGather, module)
            if use_checkpoint:
                # Bind the chunk object now: backward recomputation must not
                # capture the final loop variable.
                def read_candidates(
                    chunk_power: Tensor, chunk_floor: Tensor, chunk_gather: CombGather = gather
                ) -> Tensor:
                    return self._chunk(chunk_power, chunk_floor, chunk_gather, times)

                chunks.append(
                    cast(Tensor, checkpoint(read_candidates, power, floor, use_reentrant=False))
                )
            else:
                chunks.append(self._chunk(power, floor, gather, times))
        active_memory = torch.cat(chunks, 2)
        b, t, _, _ = active_memory.shape
        off = self.off_memory.view(1, 1, 1, 64).expand(b, t, 1, 64)
        memory = torch.cat((off, active_memory), 2)
        rate = torch.arange(151, device=power.device, dtype=power.dtype) / _MAX_RATE
        coords = torch.stack(
            (
                rate[None, None, :].expand(b, t, -1),
                times[None, :, None].expand(b, -1, 151),
                energy.log1p()[:, None, None].expand(-1, t, 151),
            ),
            -1,
        )
        return memory + self.coordinates(coords)


class _LocalPatchEncoder(nn.Module):
    """129-sample learned local patch, evaluated without an unfold expansion.

    Kernels 9/7/7, strides 4/4/1, symmetric pads 4/3/3 give centres m=16j
    and receptive field 9 + 6*4 + 6*16 = 129 at 500 Hz. Convolution taps
    preserve signed patch-relative positions, rather than pooling then striding
    the lag products. Fixed channel identities supply lag/bandwidth descriptors;
    their numerical values, absolute time and energy validity are separate inputs.
    """

    def __init__(self, width: int):
        super().__init__()
        # Each bandwidth: power, eight re/im products, five validity masks.
        # Plus three bandwidths, four lags, absolute time, energy-valid bit.
        self.network = nn.Sequential(
            nn.Conv1d(51, 16, 9, stride=4, padding=4),
            nn.GELU(),
            nn.Conv1d(16, 32, 7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv1d(32, width, 7, padding=3),
        )

    def forward(
        self, envelope: Tensor, valid: Tensor, reference: Tensor, phase_products: bool
    ) -> Tensor:
        b, r, k, w, m = envelope.shape
        energy_valid = reference > 1e-12
        mask = valid & energy_valid[:, None, None, None, None]
        scale = reference.clamp_min(1e-12)[:, None, None, None, None]
        channels = [envelope.abs().square() / scale * mask]
        lag_masks = []
        for lag in _LAGS:
            if lag < m:
                available = mask & F.pad(mask[..., :-lag], (lag, 0), value=False)
            else:
                available = torch.zeros_like(mask)
            if phase_products:
                previous = (
                    F.pad(envelope[..., :-lag], (lag, 0)) if lag < m else torch.zeros_like(envelope)
                )
                product = envelope * previous.conj() / scale
                channels.extend((product.real * available, product.imag * available))
            else:
                channels.extend((torch.zeros_like(envelope.real), torch.zeros_like(envelope.real)))
            lag_masks.append(available)
        channels.extend([mask.to(reference.dtype)] + [v.to(reference.dtype) for v in lag_masks])
        features = torch.stack(channels, -2).reshape(b * r * k, w * 14, m)
        descriptors = reference.new_tensor(
            [value / 128 for value in _BANDWIDTHS] + [value / 500 for value in _LAGS]
        )
        descriptors = descriptors[None, :, None].expand(b * r * k, -1, m)
        times = torch.arange(m, device=envelope.device, dtype=reference.dtype) / 500
        times = times[None, None, :].expand(b * r * k, 1, -1)
        energy_bit = energy_valid[:, None, None, None, None].expand(b, r, k, 1, m)
        features = torch.cat(
            (features, descriptors, times, energy_bit.reshape(b * r * k, 1, m).to(reference.dtype)),
            1,
        )
        encoded = self.network(features)
        return encoded.transpose(1, 2).reshape(b, r, k, encoded.shape[-1], -1)


class _RefinementBlock(nn.Module):
    def __init__(self, width: int, heads: int, joint_slots: bool, harmonic_chunk: int):
        super().__init__()
        self.joint_slots, self.harmonic_chunk = joint_slots, harmonic_chunk
        self.patch = _LocalPatchEncoder(width)
        self.local_descriptors = nn.Linear(21, width)
        self.geometry = nn.Sequential(nn.Linear(9, 32), nn.GELU(), nn.Linear(32, width))
        self.state_features = nn.Sequential(nn.Linear(6, width), nn.GELU(), nn.Linear(width, width))
        self.time_attention = _AttentionFF(width, heads)
        self.order_attention = _AttentionFF(width, heads)
        self.slot_attention = _AttentionFF(width, heads) if joint_slots else None
        self.order_score = nn.Linear(width, 1)
        self.global_read = _CandidateRead(width, heads)
        self.out_norm = nn.LayerNorm(width)
        self.update = nn.Linear(width, 1)
        nn.init.normal_(self.update.weight, std=1e-3)
        nn.init.zeros_(self.update.bias)

    def forward(
        self,
        local: Tensor,
        descriptors: Tensor,
        state: Tensor,
        rates: Tensor,
        previous_update: Tensor,
        memory: Tensor,
        observation_rates: Tensor,
        orders: Tensor,
        candidates: Tensor,
        times: Tensor,
    ) -> tuple[Tensor, Tensor]:
        b, r, k, t, d = local.shape
        before = F.pad(rates[..., 1:] - rates[..., :-1], (1, 0))
        after = F.pad(rates[..., 1:] - rates[..., :-1], (0, 1))
        scalars = torch.stack(
            (
                rates / _MAX_RATE,
                previous_update / _MAX_RATE,
                before / _MAX_RATE,
                after / _MAX_RATE,
                times[None, None, :].expand_as(rates),
                (rates > 0.5).to(rates.dtype),
            ),
            -1,
        )
        state = state + self.state_features(scalars)
        pieces = []
        for start in range(0, k, self.harmonic_chunk):
            geom = _interloper_geometry(
                observation_rates,
                orders[start : start + self.harmonic_chunk],
                joint_slots=self.joint_slots,
            )
            source_valid = geom[..., 5:6]
            pieces.append((self.geometry(geom) * source_valid).sum((4, 5)) / len(_BANDWIDTHS))
        tokens = local + self.local_descriptors(descriptors) + torch.cat(pieces, 2)
        tokens = tokens + state[:, :, None]
        tokens = self.time_attention(tokens.reshape(b * r * k, t, d)).reshape(b, r, k, t, d)
        tokens = tokens.transpose(2, 3)
        tokens = self.order_attention(tokens.reshape(b * r * t, k, d)).reshape(b, r, t, k, d)
        if self.slot_attention is not None:
            tokens = tokens.permute(0, 2, 3, 1, 4)
            tokens = self.slot_attention(tokens.reshape(b * t * k, r, d)).reshape(b, t, k, r, d)
            tokens = tokens.permute(0, 3, 1, 2, 4)
        weights = self.order_score(tokens).softmax(3)
        state = state + (weights * tokens).sum(3)
        state = self.global_read(state, memory, rates, candidates)
        correction = self.update(self.out_norm(state)).squeeze(-1).float()
        return (rates.float() + correction).clamp(0, _MAX_RATE), state


class JHTR(nn.Module):
    """Joint six-block RPS predictor/refiner with native Hydra construction.

    ``audio`` is (B,N) or (B,1,N). Optional ``cond`` is already on the physical
    hop-512 frame grid, (B,num_rotors,N//512+1). Conditional rows are neither
    sorted nor matched; shared row-equivariant updates preserve their order.
    ``forward`` returns a tensor; ``forward_with_diagnostics`` additionally
    exposes initialization plus every block in ``trajectories`` (B,S+1,R,T).

    Ablations do not change the objective: ``phase_products=False`` zeroes only
    product inputs, ``reread=False`` freezes the complete initial local read,
    and ``joint_slots=False`` removes slot attention and foreign descriptors
    in both initialization and refinement. Same-slot temporal/order reasoning
    and the common audio-only magnitude memory remain available.
    """

    window: Tensor
    orders: Tensor
    candidates: Tensor

    def __init__(
        self,
        n_fft: int = 4096,
        hop_length: int = 512,
        num_rotors: int = 4,
        sample_rate: int = 16000,
        n_blocks: int = 6,
        d_model: int = 128,
        n_heads: int = 4,
        k_max: int = 32,
        harmonic_chunk: int = 4,
        checkpoint_blocks: bool = True,
        phase_products: bool = True,
        reread: bool = True,
        joint_slots: bool = True,
    ):
        super().__init__()
        if sample_rate != 16000 or hop_length != 512:
            raise ValueError(
                "JHTR's 500 Hz/129-sample patch contract requires 16000 Hz and hop 512"
            )
        if n_fft < 4 or n_fft % 2 or n_blocks < 1 or k_max < 1 or harmonic_chunk < 1:
            raise ValueError("Use an even n_fft >=4 and positive block/order/chunk counts")
        if num_rotors < 1 or d_model < 1 or n_heads < 1 or d_model % n_heads:
            raise ValueError(
                "Positive rotor/width/head counts and head-divisible width are required"
            )
        self.n_fft, self.hop_length, self.sample_rate = n_fft, hop_length, sample_rate
        self.num_rotors, self.n_blocks, self.d_model = num_rotors, n_blocks, d_model
        self.k_max, self.harmonic_chunk = k_max, harmonic_chunk
        self.checkpoint_blocks = checkpoint_blocks
        self.phase_products, self.reread, self.joint_slots = phase_products, reread, joint_slots
        self.pad_samples = 8000
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer(
            "orders", torch.arange(1, k_max + 1, dtype=torch.float32), persistent=False
        )
        self.register_buffer("candidates", torch.arange(151, dtype=torch.float32), persistent=False)
        self.coarse = _CoarseMemory(n_fft, sample_rate, k_max)
        self.slot_queries = nn.Parameter(torch.randn(num_rotors, d_model) / math.sqrt(d_model))
        self.initial_coordinates = nn.Linear(1, d_model)
        self.initial_time = _AttentionFF(d_model, n_heads)
        self.initial_slots = _AttentionFF(d_model, n_heads) if joint_slots else None
        self.initial_read = _CandidateRead(d_model, n_heads)
        self.initial_query = nn.Linear(d_model, 64)
        self.initial_keys = nn.Linear(64, 64)
        self.initial_offset = nn.Linear(d_model, 1)
        nn.init.zeros_(self.initial_offset.bias)
        self.conditional_state = nn.Linear(2, d_model)
        self.blocks = nn.ModuleList(
            [
                _RefinementBlock(d_model, n_heads, joint_slots, harmonic_chunk)
                for _ in range(n_blocks)
            ]
        )

    def _initialize(self, memory: Tensor, times: Tensor) -> tuple[Tensor, Tensor]:
        b, t, _, _ = memory.shape
        r, d = self.num_rotors, self.d_model
        state = (
            self.slot_queries[None, :, None, :]
            + self.initial_coordinates(times[:, None])[None, None]
        )
        state = state.expand(b, -1, -1, -1)
        state = self.initial_read(state, memory, memory.new_zeros(b, r, t), self.candidates)
        state = self.initial_time(state.reshape(b * r, t, d)).reshape(b, r, t, d)
        if self.initial_slots is not None:
            state = state.transpose(1, 2)
            state = (
                self.initial_slots(state.reshape(b * t, r, d)).reshape(b, t, r, d).transpose(1, 2)
            )
        query, keys = self.initial_query(state), self.initial_keys(memory)
        logits = torch.einsum("brtd,btgd->brtg", query, keys) / 8
        return _modal_rates(logits, self.initial_offset(state), self.candidates), state

    def _local_descriptors(
        self, floor: Tensor, rates: Tensor, orders: Tensor, times: Tensor, energy_valid: Tensor
    ) -> Tensor:
        centre = rates[:, :, None, :] * orders[None, None, :, None]
        values = []
        for bandwidth in _BANDWIDTHS:
            for offset in (-bandwidth, 0.0, bandwidth):
                value, valid = _sample_spectrum(
                    floor, centre + offset, self.sample_rate, self.n_fft
                )
                values.extend((value.log1p(), valid.to(value.dtype)))
        shape = centre.shape
        values.extend(
            (
                orders[None, None, :, None].expand(shape) / 32,
                times[None, None, None, :].expand(shape),
                energy_valid[:, None, None, None].expand(shape).to(floor.dtype),
            )
        )
        return torch.stack(values, -1)

    def _read_chunk(
        self,
        analytic: Tensor,
        rates: Tensor,
        reference: Tensor,
        orders: Tensor,
        n_samples: int,
        patch: _LocalPatchEncoder,
    ) -> Tensor:
        # Autocast must never quantize carriers, lag products or the Q division.
        with torch.autocast(device_type=rates.device.type, enabled=False):
            envelopes, valid = demodulate_trajectories(
                analytic,
                rates.float(),
                orders,
                n_samples=n_samples,
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
                envelope_rate=500,
                half_bandwidths=_BANDWIDTHS,
                pad_samples=self.pad_samples,
                harmonic_chunk=self.harmonic_chunk,
            )
        # The encoder prepares features in fp32, then uses the caller's neural
        # autocast context for its convolutions.
        return patch(envelopes, valid, reference, self.phase_products)

    def _run(
        self, audio: Tensor, cond: Tensor | None, diagnostics: bool
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if audio.ndim == 3 and audio.shape[1] == 1:
            audio = audio[:, 0]
        if audio.ndim != 2 or audio.shape[-1] < 1:
            raise ValueError("audio must be nonempty mono (B,N) or (B,1,N)")
        b, n = audio.shape
        t = n // self.hop_length + 1
        use_checkpoint = self.checkpoint_blocks and self.training and torch.is_grad_enabled()
        with torch.autocast(device_type=audio.device.type, enabled=False):
            audio = audio.float()
            analytic = analytic_signal_tensor(audio, pad_samples=self.pad_samples)
            reference = analytic[:, self.pad_samples : self.pad_samples + n].abs().square().mean(-1)
            spectrum = torch.stft(
                audio,
                self.n_fft,
                self.hop_length,
                window=self.window.float(),
                center=True,
                pad_mode="constant",
                return_complex=True,
            )
            normalization = reference.clamp_min(1e-12)[:, None, None] * self.window.square().sum()
            power = spectrum.abs().square() / normalization
            power = power * (reference > 1e-12)[:, None, None]
            # The same established median-floor normalizer; it is not a learned
            # spectral-denoising path. Carrier/waveform feature paths stay live.
            floor = local_floor_torch(power, width_bins=31).detach()
            times = torch.arange(t, device=audio.device, dtype=torch.float32) * (
                self.hop_length / self.sample_rate
            )
        memory = self.coarse(power, floor, times, reference, use_checkpoint)
        if cond is None:
            rates, state = self._initialize(memory, times)
        else:
            if cond.shape != (b, self.num_rotors, t):
                raise ValueError(
                    f"cond must be on hop-512 timestamps with shape {(b, self.num_rotors, t)}"
                )
            rates = cond.to(device=audio.device, dtype=torch.float32).clamp(0, _MAX_RATE)
            state = self.conditional_state(
                torch.stack((rates / _MAX_RATE, times[None, None, :].expand_as(rates)), -1)
            )
        trajectories = [rates] if diagnostics else []
        previous_update = torch.zeros_like(rates)
        initial_rates = rates
        frozen_envelopes: list[tuple[Tensor, Tensor]] | None = None
        if not self.reread:
            # Cache raw observations, not one block's learned embedding: all six
            # untied patch encoders must still learn in the frozen-read control.
            frozen_envelopes = []
            for orders in self.orders.split(self.harmonic_chunk):

                def read_envelopes(
                    waveform: Tensor, trajectories: Tensor, chunk_orders: Tensor = orders
                ) -> tuple[Tensor, Tensor]:
                    return demodulate_trajectories(
                        waveform,
                        trajectories,
                        chunk_orders,
                        n_samples=n,
                        sample_rate=self.sample_rate,
                        hop_length=self.hop_length,
                        envelope_rate=500,
                        half_bandwidths=_BANDWIDTHS,
                        pad_samples=self.pad_samples,
                        harmonic_chunk=self.harmonic_chunk,
                    )

                with torch.autocast(device_type=audio.device.type, enabled=False):
                    result = (
                        cast(
                            tuple[Tensor, Tensor],
                            checkpoint(
                                read_envelopes, analytic, initial_rates, use_reentrant=False
                            ),
                        )
                        if use_checkpoint
                        else read_envelopes(analytic, initial_rates)
                    )
                frozen_envelopes.append(result)
        for module in self.blocks:
            block = cast(_RefinementBlock, module)
            local_chunks = []
            read_rates = rates if self.reread else initial_rates
            for index, orders in enumerate(self.orders.split(self.harmonic_chunk)):
                if frozen_envelopes is None:

                    def read_local(
                        waveform: Tensor,
                        trajectories: Tensor,
                        power_reference: Tensor,
                        chunk_orders: Tensor = orders,
                        patch_encoder: _LocalPatchEncoder = block.patch,
                    ) -> Tensor:
                        return self._read_chunk(
                            waveform, trajectories, power_reference, chunk_orders, n, patch_encoder
                        )

                    local = (
                        cast(
                            Tensor,
                            checkpoint(
                                read_local, analytic, read_rates, reference, use_reentrant=False
                            ),
                        )
                        if use_checkpoint
                        else read_local(analytic, read_rates, reference)
                    )
                else:
                    envelopes, valid = frozen_envelopes[index]

                    def encode_frozen(
                        observations: Tensor,
                        observation_valid: Tensor,
                        power_reference: Tensor,
                        patch_encoder: _LocalPatchEncoder = block.patch,
                    ) -> Tensor:
                        return patch_encoder(
                            observations, observation_valid, power_reference, self.phase_products
                        )

                    local = (
                        cast(
                            Tensor,
                            checkpoint(
                                encode_frozen, envelopes, valid, reference, use_reentrant=False
                            ),
                        )
                        if use_checkpoint
                        else encode_frozen(envelopes, valid, reference)
                    )
                local_chunks.append(local)
            local = torch.cat(local_chunks, 2)
            descriptors = self._local_descriptors(
                floor, read_rates, self.orders, times, reference > 1e-12
            )

            def refine(
                local_features: Tensor,
                local_descriptors: Tensor,
                hidden_state: Tensor,
                current_rates: Tensor,
                last_update: Tensor,
                coarse_memory: Tensor,
                observation_rates: Tensor,
                refinement_block: _RefinementBlock = block,
            ) -> tuple[Tensor, Tensor]:
                return refinement_block(
                    local_features,
                    local_descriptors,
                    hidden_state,
                    current_rates,
                    last_update,
                    coarse_memory,
                    observation_rates,
                    self.orders,
                    self.candidates,
                    times,
                )

            arguments = (local, descriptors, state, rates, previous_update, memory, read_rates)
            updated, state = (
                cast(tuple[Tensor, Tensor], checkpoint(refine, *arguments, use_reentrant=False))
                if use_checkpoint
                else refine(*arguments)
            )
            previous_update, rates = updated - rates, updated
            if diagnostics:
                trajectories.append(rates)
        info = {"trajectories": torch.stack(trajectories, 1)} if diagnostics else {}
        return rates, info

    def forward(self, audio: Tensor, cond: Tensor | None = None) -> Tensor:
        return self._run(audio, cond, diagnostics=False)[0]

    def forward_with_diagnostics(
        self, audio: Tensor, cond: Tensor | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        return self._run(audio, cond, diagnostics=True)
