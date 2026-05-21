"""
Generate two variants of the SimpleConv architecture diagram:
  - figures/fig_arch_detailed.tex  (every layer: Conv2D / BN / LReLU separately)
  - figures/fig_arch_compact.tex   (each encoder block as one banded box)

Run from the paper directory:
    python make_fig_arch.py

The .tex files are standalone documents compiled to PDF by `make pdf`.
PlotNeuralNet must be cloned at ./PlotNeuralNet (done already).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "PlotNeuralNet"))
from pycore.tikzeng import to_head, to_cor, to_begin, to_end, to_connection, to_generate

LAYERS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "PlotNeuralNet")
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom TikZ primitives (PlotNeuralNet only ships Conv/Pool/SoftMax/etc.)
# ──────────────────────────────────────────────────────────────────────────────

def to_extra_colors():
    """Extra named colors and packages on top of PlotNeuralNet defaults."""
    return r"""
\usepackage{graphicx}
\def\InputColor{rgb:white,2;black,1}
\def\BnColor{rgb:green,3;white,5}
\def\ActColor{rgb:orange,4;red,1;white,4}
\def\DropColor{rgb:blue,1;green,1;white,6}
\def\PoolColor{rgb:red,3;white,5}
\def\OutColor{rgb:yellow,3;white,5}
"""


def _box(name, fill, width, height, depth,
         xlabel=None, zlabel="", caption="", offset="(0,0,0)", to="(0,0,0)"):
    """Generic coloured Box (extends to_Conv with arbitrary fill)."""
    xlabel_line = (r"        xlabel={" + xlabel + r"}," + "\n") if xlabel else ""
    return r"""
\pic[shift={""" + offset + r"""}] at """ + to + r"""
    {Box={
        name=""" + name + r""",
        caption=""" + caption + r""",
""" + xlabel_line + r"""        zlabel=""" + str(zlabel) + r""",
        fill=""" + fill + r""",
        height=""" + str(height) + r""",
        width=""" + str(width) + r""",
        depth=""" + str(depth) + r"""
        }
    };
"""


def _banded(name, fill, bandFill, widths, height, depth,
            xlabel='{""}', zlabel="", caption="", offset="(0,0,0)", to="(0,0,0)"):
    """RightBandedBox: main slab + right band (like Conv+Activation)."""
    w0, w1 = widths
    return r"""
\pic[shift={""" + offset + r"""}] at """ + to + r"""
    {RightBandedBox={
        name=""" + name + r""",
        caption=""" + caption + r""",
        xlabel={""" + xlabel + r"""},
        zlabel=""" + str(zlabel) + r""",
        fill=""" + fill + r""",
        bandfill=""" + bandFill + r""",
        height=""" + str(height) + r""",
        width={ """ + str(w0) + r""" , """ + str(w1) + r""" },
        depth=""" + str(depth) + r"""
        }
    };
"""


def rot(text):
    """Wrap caption in rotatebox and flatten any newlines to spaces."""
    flat = text.replace(r"\\", " ")
    return r"\rotatebox{90}{" + flat + "}"


def _arrow(src, dst):
    return (r"\draw [connection]  (" + src + r"-east) -- node {\midarrow} ("
            + dst + r"-west);" + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Architecture parameters
# ──────────────────────────────────────────────────────────────────────────────

# Frequency heights per block output (log-scaled for visual proportion)
# 1025 → 513 → 257 → 129 → 65 → 33
F_HEIGHTS = [32, 28, 24, 20, 16, 12]  # input, after B1…B5

T_DEPTH   = 12   # time axis depth (constant)
BLOCK_GAP = "(1.2,0,0)"
SMALL_GAP = "(0.6,0,0)"
NO_GAP    = "(0,0,0)"

# ──────────────────────────────────────────────────────────────────────────────
# VARIANT A — detailed: Conv2D | BN | LReLU shown individually
# ──────────────────────────────────────────────────────────────────────────────

def make_detailed(outpath):
    # Encoder block specs: (ch_in, ch_out, kernel_label, freq_out_label)
    encoder_blocks = [
        (1,  45, "7×5", "513"),
        (45, 90, "7×5", "257"),
        (90, 90, "5×3", "129"),
        (90, 90, "5×3", "65"),
        (90, 90, "5×3", "33"),
    ]
    # channel→visual width
    ch_w = {1: 3, 45: 5, 90: 8}

    arch = [
        to_head(LAYERS_DIR),
        to_cor(),
        to_extra_colors(),
        to_begin(),
    ]

    # ── Input ──
    arch += [
        _box("input", r"\InputColor", width=3,
             height=F_HEIGHTS[0], depth=T_DEPTH,
             xlabel='{"1"}', zlabel="1025",
             caption=rot("Input log|X|")),
    ]
    prev = "input"

    # ── Encoder blocks ──
    for i, (ch_in, ch_out, kern, f_out) in enumerate(encoder_blocks):
        h  = F_HEIGHTS[i + 1]
        cw = ch_w[ch_out]

        conv_name = f"conv{i+1}"
        bn_name   = f"bn{i+1}"
        act_name  = f"act{i+1}"

        arch += [
            _box(conv_name, r"\ConvColor",
                 width=cw, height=h, depth=T_DEPTH,
                 xlabel='{"' + str(ch_out) + '"}',
                 zlabel=f_out,
                 caption=rot(f"Conv2D {kern}"),
                 offset=BLOCK_GAP, to=f"({prev}-east)"),
            _box(bn_name, r"\BnColor",
                 width=2, height=h, depth=T_DEPTH,
                 caption=rot("BN"),
                 offset=NO_GAP, to=f"({conv_name}-east)"),
            _box(act_name, r"\ActColor",
                 width=2, height=h, depth=T_DEPTH,
                 caption=rot("LReLU"),
                 offset=NO_GAP, to=f"({bn_name}-east)"),
            _arrow(prev, conv_name),
        ]
        prev = act_name

    # ── Freq AvgPool ──
    h_pool = F_HEIGHTS[-1]
    arch += [
        _box("pool", r"\PoolColor",
             width=2, height=h_pool // 2, depth=T_DEPTH,
             caption=rot("AvgPool freq"),
             offset=BLOCK_GAP, to=f"({prev}-east)"),
        _arrow(prev, "pool"),
    ]
    prev = "pool"
    h_flat = h_pool // 2

    # ── Head: Conv1D(90→64, k=5) + ReLU + Dropout ──
    arch += [
        _box("hconv1", r"\ConvColor",
             width=5, height=h_flat, depth=T_DEPTH,
             xlabel='{"64"}', caption=rot("Conv1D k=5"),
             offset=BLOCK_GAP, to=f"({prev}-east)"),
        _box("hrelu", r"\ActColor",
             width=2, height=h_flat, depth=T_DEPTH,
             caption=rot("ReLU"),
             offset=NO_GAP, to="(hconv1-east)"),
        _box("hdrop", r"\DropColor",
             width=2, height=h_flat, depth=T_DEPTH,
             caption=rot("Drop"),
             offset=NO_GAP, to="(hrelu-east)"),
        _arrow(prev, "hconv1"),
    ]
    prev = "hdrop"

    # ── Head: Conv1D(64→4, k=1) ──
    arch += [
        _box("hconv2", r"\ConvColor",
             width=3, height=h_flat, depth=T_DEPTH,
             xlabel='{"4"}', caption=rot("Conv1D k=1"),
             offset=BLOCK_GAP, to=f"({prev}-east)"),
        _arrow(prev, "hconv2"),
    ]
    prev = "hconv2"

    # ── Output ──
    arch += [
        _box("output", r"\OutColor",
             width=2, height=h_flat, depth=T_DEPTH,
             xlabel='{"4"}', caption=rot(r"$\hat{r}(t)$"),
             offset=SMALL_GAP, to=f"({prev}-east)"),
        _arrow(prev, "output"),
    ]

    arch += [to_end()]
    to_generate(arch, outpath)
    print(f"Written: {outpath}")


# ──────────────────────────────────────────────────────────────────────────────
# VARIANT B — compact: each encoder block as a single banded box
# ──────────────────────────────────────────────────────────────────────────────

def make_compact(outpath):
    encoder_blocks = [
        (1,  45, "7×5", "513"),
        (45, 90, "7×5", "257"),
        (90, 90, "5×3", "129"),
        (90, 90, "5×3", "65"),
        (90, 90, "5×3", "33"),
    ]
    ch_w = {1: 3, 45: 5, 90: 8}

    arch = [
        to_head(LAYERS_DIR),
        to_cor(),
        to_extra_colors(),
        to_begin(),
    ]

    # ── Input ──
    arch += [
        _box("input", r"\InputColor", width=3,
             height=F_HEIGHTS[0], depth=T_DEPTH,
             xlabel='{"1"}', zlabel="1025",
             caption="Input\\\\log|X|"),
    ]
    prev = "input"

    # ── Encoder blocks (banded: Conv2D|BN+LReLU) ──
    for i, (ch_in, ch_out, kern, f_out) in enumerate(encoder_blocks):
        h  = F_HEIGHTS[i + 1]
        cw = ch_w[ch_out]
        blk = f"blk{i+1}"
        arch += [
            _banded(blk,
                    fill=r"\ConvColor", bandFill=r"\ActColor",
                    widths=(cw, 2),
                    height=h, depth=T_DEPTH,
                    xlabel='{"' + str(ch_out) + '"}',
                    zlabel=f_out,
                    caption=f"B{i+1}: Conv2D {kern}\\\\BN + LReLU",
                    offset=BLOCK_GAP, to=f"({prev}-east)"),
            _arrow(prev, blk),
        ]
        prev = blk

    # ── Freq AvgPool ──
    h_pool = F_HEIGHTS[-1]
    arch += [
        _box("pool", r"\PoolColor",
             width=2, height=h_pool // 2, depth=T_DEPTH,
             caption="AvgPool\\\\freq",
             offset=BLOCK_GAP, to=f"({prev}-east)"),
        _arrow(prev, "pool"),
    ]
    prev = "pool"
    h_flat = h_pool // 2

    # ── Head: banded Conv1D+ReLU then Conv1D ──
    arch += [
        _banded("head1",
                fill=r"\ConvColor", bandFill=r"\ActColor",
                widths=(5, 2),
                height=h_flat, depth=T_DEPTH,
                xlabel='{"64"}',
                caption="Conv1D k=5\\\\ReLU + Drop",
                offset=BLOCK_GAP, to=f"({prev}-east)"),
        _arrow(prev, "head1"),
    ]
    prev = "head1"

    arch += [
        _box("head2", r"\ConvColor",
             width=3, height=h_flat, depth=T_DEPTH,
             xlabel='{"4"}',
             caption="Conv1D k=1",
             offset=BLOCK_GAP, to=f"({prev}-east)"),
        _arrow(prev, "head2"),
    ]
    prev = "head2"

    # ── Output ──
    arch += [
        _box("output", r"\OutColor",
             width=2, height=h_flat, depth=T_DEPTH,
             xlabel='{"4"}', caption="\\^r(t)",
             offset=SMALL_GAP, to=f"({prev}-east)"),
        _arrow(prev, "output"),
    ]

    arch += [to_end()]
    to_generate(arch, outpath)
    print(f"Written: {outpath}")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    figures = os.path.join(os.path.dirname(__file__), "figures")
    make_detailed(os.path.join(figures, "fig_arch_detailed.tex"))
    make_compact (os.path.join(figures, "fig_arch_compact.tex"))
