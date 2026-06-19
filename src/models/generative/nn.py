"""
Shared neural-network building blocks for the generative models.

Ported verbatim from `drone_audition.models.nn` (the dead TensorFlow-era
comment blocks were dropped). No dependency on `env.settings`.

Classes
-------
- `Identity`                  : no-op module for skip connections
- `ChannelLayerNorm`          : LayerNorm over the channel axis of a 4-D tensor
- `NormReluConv`              : Norm -> ReLU -> Conv2d
- `ResidualLayer` / `ResidualStack` / `ResNet` : ResNet-18-style encoder
- `Fc` / `FcStack`            : Dense -> LayerNorm -> LeakyReLU stacks
- `RnnSandwich`               : FcStack -> GRU -> FcStack
- `CausalConv1d` / `CausalConv1dBlock` : left-padded causal 1-D convolution
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------


def ensure_4d(x):
    """Add extra dimensions to make sure tensor has height and width."""
    if len(x.shape) == 2:
        return x.unsqueeze(-1).unsqueeze(-1)
    elif len(x.shape) == 3:
        return x.unsqueeze(-1)
    else:
        return x


def inv_ensure_4d(x, n_dims):
    """Remove excess dims, inverse of ensure_4d() function."""
    if n_dims == 2:
        return x.squeeze(-1).squeeze(-1)
    if n_dims == 3:
        return x.squeeze(-1)
    else:
        return x


# ---------------------------------------------------------------------------
# Normalization / conv blocks
# ---------------------------------------------------------------------------


class Identity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class ChannelLayerNorm(nn.Module):
    def __init__(self, in_ch, **kwargs):
        super().__init__(**kwargs)
        self.norm = nn.LayerNorm(in_ch)

    def forward(self, x):
        return self.norm(x.transpose(1, 3)).transpose(1, 3)


class NormReluConv(nn.Sequential):
    """Norm -> ReLU -> Conv layer."""

    def __init__(self, in_ch, out_ch, k, s, **kwargs):
        """Downsample frequency by stride."""
        layers = [
            ChannelLayerNorm(in_ch),
            nn.ReLU(),
            nn.Conv2d(
                in_ch,
                out_ch,
                (k, k),
                (1, s),
                padding="same" if s == 1 else (k // 2, k // 2),
            ),
        ]
        super().__init__(*layers, **kwargs)


class ResidualLayer(nn.Module):
    """A single layer for ResNet, with a bottleneck."""

    def __init__(self, in_ch, ch_out, stride, shortcut, **kwargs):
        """Downsample frequency by stride, upsample channels by 4."""
        super().__init__(**kwargs)
        ch = ch_out // 4

        assert shortcut or (in_ch == ch_out)
        self.shortcut = shortcut

        # Layers.
        self.norm_input = ChannelLayerNorm(in_ch)

        if self.shortcut:
            self.conv_proj = nn.Conv2d(in_ch, ch_out, (1, 1), (1, stride))

        layers = [
            nn.Conv2d(in_ch, ch, (1, 1), (1, 1), padding="same"),
            NormReluConv(ch, ch, 3, stride),
            NormReluConv(ch, ch_out, 1, 1),
        ]
        self.bottleneck = nn.Sequential(*layers)

    def forward(self, inputs):
        x = inputs
        r = x

        x = ensure_4d(x)
        x = F.relu(self.norm_input(x))

        # The projection shortcut should come after the first norm and ReLU
        # since it performs a 1x1 convolution.
        r = self.conv_proj(x) if self.shortcut else r
        x = self.bottleneck(x)
        return x + r


class ResidualStack(nn.Module):
    """LayerNorm -> ReLU -> Conv layer."""

    def __init__(self, in_channels, filters, block_sizes, strides, **kwargs):
        """ResNet layers."""
        super().__init__(**kwargs)
        layers = []
        prev_ch = in_channels
        for ch, n_layers, stride in zip(filters, block_sizes, strides):
            # Only the first block per residual_stack uses shortcut and strides.
            layers.append(ResidualLayer(prev_ch, ch, stride, True))

            # Add the additional (n_layers - 1) layers to the stack.
            for _ in range(1, n_layers):
                layers.append(ResidualLayer(ch, ch, 1, False))

            prev_ch = ch

        layers.append(ChannelLayerNorm(prev_ch))
        layers.append(nn.ReLU())
        self.layers = nn.ModuleList(layers)

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class ResNet(nn.Module):
    """Residual network."""

    def __init__(self, size="large", **kwargs):
        super().__init__(**kwargs)
        size_dict = {
            "small": (32 * 4, [2, 3, 4]),
            "medium": (32 * 4, [3, 4, 6]),
            "large": (64 * 4, [3, 4, 6]),
        }
        ch, blocks = size_dict[size]
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(1, 64, (7, 7), (1, 2), padding=(3, 3)),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
                ResidualStack(64, [ch, 2 * ch, 4 * ch], blocks, [1, 2, 2]),
                ResidualStack(4 * ch, [8 * ch], [3], [2]),
            ]
        )

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Fully-connected / RNN utils
# ---------------------------------------------------------------------------


class Fc(nn.Sequential):
    """Makes a Dense -> LayerNorm -> Leaky ReLU layer."""

    def __init__(self, in_ch, ch=128, **kwargs):
        layers = [
            nn.Linear(in_ch, ch),
            nn.LayerNorm(ch),
            nn.LeakyReLU(),
        ]
        super().__init__(*layers, **kwargs)


class FcStack(nn.Sequential):
    """Stack Dense -> LayerNorm -> Leaky ReLU layers."""

    def __init__(self, in_ch, ch=256, layers=2, **kwargs):
        layers = [Fc(in_ch, ch)] + [Fc(ch, ch) for _ in range(1, layers)]
        super().__init__(*layers, **kwargs)


class RnnSandwich(nn.Module):
    """RNN Sandwiched by two FC Stacks."""

    def __init__(self, in_ch, fc_stack_ch=256, fc_stack_layers=2, rnn_ch=512, **kwargs):
        super().__init__(**kwargs)

        self.entry = FcStack(in_ch, fc_stack_ch, fc_stack_layers)
        self.rnn = nn.GRU(fc_stack_ch, rnn_ch, batch_first=True)
        self.exit = FcStack(rnn_ch, fc_stack_ch, fc_stack_layers)

    def forward(self, x):
        x = self.entry(x)
        x, _ = self.rnn(x)
        return self.exit(x)


# ---------------------------------------------------------------------------
# Convolutions
# ---------------------------------------------------------------------------


class CausalConv1d(nn.Conv1d):
    """1-D convolution that only sees the past (left-padded)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        padding_mode="constant",
        bias=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.__padding = (kernel_size - 1) * dilation
        self.__padding_mode = padding_mode

    def forward(self, input):
        return super().forward(F.pad(input, (self.__padding, 0), mode=self.__padding_mode))


class CausalConv1dBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1, **kwargs):
        super().__init__(
            CausalConv1d(in_ch, out_ch, kernel_size, stride, dilation, **kwargs),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
