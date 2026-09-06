"""
Multi-F0 CNN models — precise PyTorch reimplementation of:

    H. Cuesta, B. McFee, E. Gómez,
    "Multiple F0 Estimation in Vocal Ensembles using Convolutional Neural Networks",
    ISMIR 2020.

Model variants:
    - EarlyDeep       (build_model1 / exp1multif0)   — Fig 3a with extra 3×3 convs
    - EarlyShallow    (build_model2 / exp2multif0)   — Fig 3a without extra convs
    - LateDeep        (build_model3 / exp3multif0)   — Fig 3b (BEST model)
    - LateDeepNoPhase (build_model3_mag)             — Late/Deep, magnitude only

Architecture semantics (from paper §4.3 and original Keras code):
    - BN *before* every conv; activation *after* (ReLU everywhere except sigmoid at output).
    - (5,5) filters ≈ 1 semitone in freq × 50 ms in time.
    - (70,3) filters ≈ 14 semitones — captures harmonic relations within an octave.
    - (360,1) filter spans all frequency bins with padding='same' → each of the 360
      output positions sees a slightly shifted 360-bin window of the input.
    - Final (1,1) + sigmoid → per-bin salience in [0,1].

Input:  (batch, harmonics=5, freq=360, time=T)
Output: (batch, 1, freq=360, time=T)  — pitch salience map
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Asymmetric 'same' padding helper ───────────────────────────────────────


def _keras_same_padding(kernel_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Compute padding that replicates Keras/TF 'same' padding for stride=1.

    For odd kernels this is symmetric.  For even kernels, Keras puts the
    extra pad on the bottom/right (e.g. 34 top, 35 bottom for size 70).

    Returns (pad_left, pad_right, pad_top, pad_bottom) for F.pad.
    """
    kh, kw = kernel_size
    pad_h = kh - 1
    pad_w = kw - 1
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return (pad_left, pad_right, pad_top, pad_bottom)


def _same_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: tuple[int, int],
    bias: bool = False,
) -> nn.Conv2d:
    """Conv2d whose padding is set so that we handle it manually via F.pad.

    We always use padding=0 on the conv itself and apply F.pad before calling it.
    This lets us replicate Keras asymmetric padding exactly.
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=bias,
    )


# ── Building blocks ────────────────────────────────────────────────────────


class ConvBlock(nn.Module):
    """BN → (pad) → Conv2d → activation.

    Mirrors Keras:
        x = BatchNormalization()(x)
        x = Conv2D(filters, kernel, padding='same', activation=act)(x)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        activation: str = "relu",
    ):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.padding = _keras_same_padding(kernel_size)
        self.conv = _same_conv2d(in_channels, out_channels, kernel_size)
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):
        x = self.bn(x)
        x = F.pad(x, self.padding, mode="constant", value=0)
        x = self.conv(x)
        x = self.act(x)
        return x


class HarmBlock(ConvBlock):
    """'Harmonic' conv with (70, 3) kernel — octave-spanning receptive field."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(in_channels, out_channels, kernel_size=(70, 3), activation="relu")


# ── Base branch (shared by all two-input models) ───────────────────────────

# Per-block (out_per_group, kernel) spec of _base_branch — single source of truth
# so the fused (grouped) twin stays in lock-step with it (the first block's input
# is n_harmonics, set per-model).
_BASE_BRANCH_SPEC: list[tuple[int, tuple[int, int]]] = [
    (16, (5, 5)),
    (32, (5, 5)),
    (32, (5, 5)),
    (32, (5, 5)),
    (32, (70, 3)),
    (32, (70, 3)),
]


def _base_branch(in_channels: int) -> nn.Sequential:
    """Per-input branch used in all two-input models.

    Structure (from original build_model1/build_model2/base_model):
        BN → Conv(16, 5×5) + ReLU → BN
        → Conv(32, 5×5) + ReLU → BN
        → Conv(32, 5×5) + ReLU → BN
        → Conv(32, 5×5) + ReLU → BN
        → HarmBlock(32, 70×3) + ReLU → BN
        → HarmBlock(32, 70×3) + ReLU → BN

    Returns 32-channel feature map at original freq × time resolution.
    """
    return nn.Sequential(
        ConvBlock(in_channels, 16, (5, 5), activation="relu"),
        ConvBlock(16, 32, (5, 5), activation="relu"),
        ConvBlock(32, 32, (5, 5), activation="relu"),
        ConvBlock(32, 32, (5, 5), activation="relu"),
        HarmBlock(32, 32),
        HarmBlock(32, 32),
    )


# ── Fused (grouped) twin of the two-branch stack ────────────────────────────


class GroupedConvBlock(nn.Module):
    """BN(2C) → pad → grouped Conv2d(groups=2) → ReLU.

    Fuses the two structurally identical ``ConvBlock``s (one per input branch)
    into a single block operating on the channel-stacked ``[mag, dphase]`` input.
    ``groups=2`` keeps the two branches' channels from mixing, so the result is
    mathematically identical to the separate branches (BN stats are per-channel,
    so a single BN over the stacked channels == two independent BNs, in train and
    eval alike). The grouped-conv output ordering ``[group0, group1]`` already
    equals ``torch.cat([mag_feats, phase_feats])`` — the old concat is free.
    """

    def __init__(self, in_per_group: int, out_per_group: int, kernel_size: tuple[int, int]):
        super().__init__()
        self.bn = nn.BatchNorm2d(2 * in_per_group)
        self.padding = _keras_same_padding(kernel_size)
        self.conv = nn.Conv2d(
            2 * in_per_group, 2 * out_per_group, kernel_size, stride=1, padding=0, bias=False, groups=2
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x = F.pad(x, self.padding, mode="constant", value=0)
        x = self.conv(x)
        x = self.act(x)
        return x


def _fused_base_branch(in_channels: int) -> nn.Sequential:
    """Grouped (groups=2) twin of two stacked ``_base_branch`` stacks.

    Input  : ``(B, 2·in_channels, F, T)`` = ``cat([mag, dphase], dim=1)``.
    Output : ``(B, 64, F, T)`` = ``cat([mag_feats(32), phase_feats(32)], dim=1)``.
    """
    blocks: list[nn.Module] = []
    in_pg = in_channels
    for out_pg, k in _BASE_BRANCH_SPEC:
        blocks.append(GroupedConvBlock(in_pg, out_pg, k))
        in_pg = out_pg
    return nn.Sequential(*blocks)


def _merge_branches_in_state_dict(sd: dict, prefix: str) -> None:
    """In place: ``branch_mag.* + branch_phase.* → fused_branch.*`` (concat dim 0)."""
    mp, pp, fp = prefix + "branch_mag.", prefix + "branch_phase.", prefix + "fused_branch."
    idxs = sorted({int(k[len(mp):].split(".")[0]) for k in sd if k.startswith(mp)})
    for i in idxs:
        for sub in ("bn.weight", "bn.bias", "bn.running_mean", "bn.running_var", "conv.weight"):
            sd[f"{fp}{i}.{sub}"] = torch.cat([sd[f"{mp}{i}.{sub}"], sd[f"{pp}{i}.{sub}"]], dim=0)
        nbt = f"{mp}{i}.bn.num_batches_tracked"
        if nbt in sd:
            sd[f"{fp}{i}.bn.num_batches_tracked"] = sd[nbt]
    for k in [k for k in sd if k.startswith(mp) or k.startswith(pp)]:
        del sd[k]


def _split_branch_in_state_dict(sd: dict, prefix: str) -> None:
    """In place: ``fused_branch.* → branch_mag.* + branch_phase.*`` (split dim 0)."""
    fp, mp, pp = prefix + "fused_branch.", prefix + "branch_mag.", prefix + "branch_phase."
    idxs = sorted({int(k[len(fp):].split(".")[0]) for k in sd if k.startswith(fp)})
    for i in idxs:
        for sub in ("bn.weight", "bn.bias", "bn.running_mean", "bn.running_var", "conv.weight"):
            v = sd[f"{fp}{i}.{sub}"]
            c = v.shape[0] // 2
            sd[f"{mp}{i}.{sub}"], sd[f"{pp}{i}.{sub}"] = v[:c], v[c:]
        nbt = f"{fp}{i}.bn.num_batches_tracked"
        if nbt in sd:
            sd[f"{mp}{i}.bn.num_batches_tracked"] = sd[nbt]
            sd[f"{pp}{i}.bn.num_batches_tracked"] = sd[nbt]
    for k in [k for k in sd if k.startswith(fp)]:
        del sd[k]


# ── Model classes ──────────────────────────────────────────────────────────


class MultiF0Estimator(nn.Module):
    """Base class for all multi-F0 models.

    Input:  (batch, harmonics=5, freq=360, time=T)  — for mag and optionally dphase
    Output: (batch, 1, freq=360, time=T)             — pitch salience in [0, 1]
    """

    def __init__(self):
        super().__init__()

    def predict_salience(
        self, mag: torch.Tensor, dphase: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Convenience: run forward in eval mode, return squeezed salience."""
        self.eval()
        with torch.no_grad():
            out = self.forward(mag, dphase)
        return out


class EarlyShallow(MultiF0Estimator):
    """Early/Shallow model (Fig 3a, without the two 3×3 convs).

    Both inputs get one conv each, then concatenated and processed jointly.
    """

    def __init__(self, n_harmonics: int = 5):
        super().__init__()

        # Per-input first conv
        self.conv1a = ConvBlock(n_harmonics, 16, (5, 5), activation="relu")
        self.conv1b = ConvBlock(n_harmonics, 16, (5, 5), activation="relu")

        # Shared trunk (input: 32 ch from concat)
        self.shared = nn.Sequential(
            ConvBlock(32, 32, (5, 5), activation="relu"),
            ConvBlock(32, 32, (5, 5), activation="relu"),
            ConvBlock(32, 32, (5, 5), activation="relu"),
            HarmBlock(32, 32),
            HarmBlock(32, 32),
        )

        # distribution: (360, 1) conv, 8 filters, preserves 360 freq bins
        self.dist_conv = _same_conv2d(32, 8, (360, 1))
        self.dist_pad = _keras_same_padding((360, 1))
        self.dist_bn = nn.BatchNorm2d(8)
        self.dist_act = nn.ReLU(inplace=True)

        # squishy: (1, 1) → 1 channel, sigmoid
        self.squishy = nn.Sequential(
            nn.BatchNorm2d(8),
            _same_conv2d(8, 1, (1, 1)),
            nn.Sigmoid(),
        )

    def forward(self, mag: torch.Tensor, dphase: torch.Tensor | None = None) -> torch.Tensor:
        if dphase is None:
            raise ValueError("EarlyShallow requires dphase input")

        x1 = self.conv1a(mag)
        x2 = self.conv1b(dphase)
        x = torch.cat([x1, x2], dim=1)  # (B, 32, F, T)
        x = self.shared(x)  # (B, 32, F, T)
        x = F.pad(x, self.dist_pad, mode="constant", value=0)
        x = self.dist_conv(x)  # (B, 8, F, T)
        x = self.dist_bn(x)
        x = self.dist_act(x)
        x = self.squishy(x)  # (B, 1, F, T)
        return x


class EarlyDeep(MultiF0Estimator):
    """Early/Deep model (Fig 3a, with two extra 3×3 conv layers)."""

    def __init__(self, n_harmonics: int = 5):
        super().__init__()

        self.conv1a = ConvBlock(n_harmonics, 16, (5, 5), activation="relu")
        self.conv1b = ConvBlock(n_harmonics, 16, (5, 5), activation="relu")

        self.shared = nn.Sequential(
            ConvBlock(32, 32, (5, 5), activation="relu"),
            ConvBlock(32, 32, (5, 5), activation="relu"),
            ConvBlock(32, 32, (5, 5), activation="relu"),
            HarmBlock(32, 32),
            HarmBlock(32, 32),
        )

        # Extra 3×3 convs (only in Early/Deep)
        self.extra = nn.Sequential(
            ConvBlock(32, 64, (3, 3), activation="relu"),
            ConvBlock(64, 64, (3, 3), activation="relu"),
        )

        self.dist_conv = _same_conv2d(64, 8, (360, 1))
        self.dist_pad = _keras_same_padding((360, 1))
        self.dist_bn = nn.BatchNorm2d(8)
        self.dist_act = nn.ReLU(inplace=True)

        self.squishy = nn.Sequential(
            nn.BatchNorm2d(8),
            _same_conv2d(8, 1, (1, 1)),
            nn.Sigmoid(),
        )

    def forward(self, mag: torch.Tensor, dphase: torch.Tensor | None = None) -> torch.Tensor:
        if dphase is None:
            raise ValueError("EarlyDeep requires dphase input")

        x1 = self.conv1a(mag)
        x2 = self.conv1b(dphase)
        x = torch.cat([x1, x2], dim=1)
        x = self.shared(x)
        x = self.extra(x)  # (B, 64, F, T)
        x = F.pad(x, self.dist_pad, mode="constant", value=0)
        x = self.dist_conv(x)
        x = self.dist_bn(x)
        x = self.dist_act(x)
        x = self.squishy(x)
        return x


class LateDeep(MultiF0Estimator):
    """Late/Deep model (Fig 3b) — BEST performing model in the paper.

    Two independent branches (6 conv layers each) process mag and dphase
    separately.  Concatenated at channel 64, then 3×3 convs, then output.

    ``fused_branches=True`` replaces the two branches with a single grouped
    (``groups=2``) stack over the channel-stacked ``[mag, dphase]`` input —
    mathematically identical (verified to float32 precision) but one kernel
    launch per layer instead of two, and the concat becomes free. Checkpoints
    convert transparently between the two layouts via a load-state-dict pre-hook,
    so a model trained one way loads either way.

    ``n_maps`` is the number of output maps. 1 is the paper model — one shared
    multi-hot salience map. ``n_maps > 1`` widens the final 1x1 convolution
    only, so the trunk emits ONE MAP PER SOURCE; `models.salience_rps` uses it
    for the per-rotor salience layers (the ``_l4`` rows). The parameter name
    stays ``squishy.1``, thus a ``n_maps=1`` model is weight-identical to every
    checkpoint made before this option existed.
    """

    def __init__(self, n_harmonics: int = 5, fused_branches: bool = False, n_maps: int = 1):
        super().__init__()
        self.fused_branches = fused_branches
        self.n_maps = int(n_maps)

        if fused_branches:
            self.fused_branch = _fused_base_branch(n_harmonics)
        else:
            # Independent branches (32 channels each)
            self.branch_mag = _base_branch(n_harmonics)
            self.branch_phase = _base_branch(n_harmonics)

        # Remap branch_mag/branch_phase <-> fused_branch when the checkpoint's
        # layout differs from this instance's. Pre-hook (not a load_state_dict
        # override) so it fires even when loaded via a parent module.
        self._register_load_state_dict_pre_hook(self._branch_remap_hook)

        # Post-concat (64 ch = 32 + 32)
        self.post = nn.Sequential(
            ConvBlock(64, 64, (3, 3), activation="relu"),
            ConvBlock(64, 64, (3, 3), activation="relu"),
        )

        self.dist_conv = _same_conv2d(64, 8, (360, 1))
        self.dist_pad = _keras_same_padding((360, 1))
        self.dist_bn = nn.BatchNorm2d(8)
        self.dist_act = nn.ReLU(inplace=True)

        self.squishy = nn.Sequential(
            nn.BatchNorm2d(8),
            _same_conv2d(8, self.n_maps, (1, 1)),
            nn.Sigmoid(),
        )

    def _branch_remap_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Convert checkpoint branch layout to match this instance's, in place."""
        has_fused = any(k.startswith(prefix + "fused_branch.") for k in state_dict)
        has_split = any(k.startswith(prefix + "branch_mag.") for k in state_dict)
        if self.fused_branches and has_split and not has_fused:
            _merge_branches_in_state_dict(state_dict, prefix)
        elif not self.fused_branches and has_fused and not has_split:
            _split_branch_in_state_dict(state_dict, prefix)

    def forward(
        self,
        mag: torch.Tensor,
        dphase: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        if dphase is None:
            raise ValueError("LateDeep requires dphase input")

        if self.fused_branches:
            # Grouped stack: output is already [mag_feats, phase_feats] -> (B, 64, F, T)
            x = self.fused_branch(torch.cat([mag, dphase], dim=1))
        else:
            x_mag = self.branch_mag(mag)  # (B, 32, F, T)
            x_phase = self.branch_phase(dphase)  # (B, 32, F, T)
            x = torch.cat([x_mag, x_phase], dim=1)  # (B, 64, F, T)
        x = self.post(x)  # (B, 64, F, T)
        x = F.pad(x, self.dist_pad, mode="constant", value=0)
        x = self.dist_conv(x)  # (B, 8, F, T)
        x = self.dist_bn(x)
        x = self.dist_act(x)
        if return_logits:
            # Pre-sigmoid logits for BCEWithLogitsLoss. squishy = [BN, Conv(1×1),
            # Sigmoid]; replay BN + Conv and skip the final activation. Indexing the
            # Sequential (rather than restructuring it) keeps checkpoints loadable.
            return self.squishy[1](self.squishy[0](x))  # (B, n_maps, F, T) logits
        x = self.squishy(x)  # (B, n_maps, F, T)
        return x


class LateDeepNoPhase(MultiF0Estimator):
    """Late/Deep without phase — magnitude-only baseline (Experiment 1).

    Same structure as Late/Deep but omits the phase branch entirely.
    The post-concat stage takes 32 channels instead of 64.
    """

    def __init__(self, n_harmonics: int = 5):
        super().__init__()

        self.branch = _base_branch(n_harmonics)

        self.post = nn.Sequential(
            ConvBlock(32, 64, (3, 3), activation="relu"),
            ConvBlock(64, 64, (3, 3), activation="relu"),
        )

        self.dist_conv = _same_conv2d(64, 8, (360, 1))
        self.dist_pad = _keras_same_padding((360, 1))
        self.dist_bn = nn.BatchNorm2d(8)
        self.dist_act = nn.ReLU(inplace=True)

        self.squishy = nn.Sequential(
            nn.BatchNorm2d(8),
            _same_conv2d(8, 1, (1, 1)),
            nn.Sigmoid(),
        )

    def forward(self, mag: torch.Tensor, dphase: torch.Tensor | None = None) -> torch.Tensor:
        x = self.branch(mag)  # (B, 32, F, T)
        x = self.post(x)  # (B, 64, F, T)
        x = F.pad(x, self.dist_pad, mode="constant", value=0)
        x = self.dist_conv(x)
        x = self.dist_bn(x)
        x = self.dist_act(x)
        x = self.squishy(x)
        return x


# ── Factory ────────────────────────────────────────────────────────────────

_MODEL_REGISTRY = {
    "early_shallow": EarlyShallow,
    "early_deep": EarlyDeep,
    "late_deep": LateDeep,
    "late_deep_nophase": LateDeepNoPhase,
}


def build_model(name: str, **kwargs) -> MultiF0Estimator:
    """Build a multi-F0 model by name.

    Args:
        name: one of 'early_shallow', 'early_deep', 'late_deep', 'late_deep_nophase'
        **kwargs: passed to the model constructor (n_harmonics)

    Returns:
        MultiF0Estimator instance
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Options: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name](**kwargs)
