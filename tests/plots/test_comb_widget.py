"""The comb explorer as a notebook widget: payload shape, page layout, and the
four properties a reader can only otherwise check by eye.

The page is HTML + JavaScript, not matplotlib, so it is verified in two steps:

* here, in Python, over the payload and the rendered markup — how many
  microphone channels are selectable, which controls sit in the top row, and
  that the frame carries a height cap
* in ``scripts/displacement/verify_page.js``, which EXECUTES the page's script
  against a stubbed DOM and drives it: the last test in this file runs that
  harness over a page built from a synthetic Frame, so an overlay that survives
  reset, a frame that grows on every redraw, or a channel selector that does
  not change the pixels is a failing test rather than a complaint.

Everything is built from a 2 s, 8-microphone synthetic Frame with two
rotor-speed tracks, which is the shape the defects were reported on.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tdseries as td

from plots.comb_page import CSS, render_html
from plots.comb_widget import build_widget_payload, discover, widget_html

ROOT = Path(__file__).resolve().parents[2]
SR = 8000
DUR_S = 2.0
N_MICS = 8
N_ROTORS = 2
RATES = (50.0, 53.0)

#: Small enough that the whole page builds in a few seconds, big enough that
#: every axis of the payload (channels x carriers x rotors x harmonics x
#: segment lengths) has more than one entry.
BUILD: dict[str, Any] = dict(
    ks="1-8",
    k_max=8,
    segs=(0.25, 0.5),
    nfft=512,
    spec_cols=120,
    strip_rows=32,
    strip_cols=48,
    decim=16,
    jobs=1,
    verbose=False,
)


def _frame() -> td.Frame:
    """8 mics of rotor-like harmonic noise plus two rotor-speed tracks.

    Each microphone gets its own gain, so the mean of its spectrogram image
    differs from every other one: that is what makes "the channel selector
    changed the pixels" checkable.
    """
    rng = np.random.default_rng(0)
    n = int(SR * DUR_S)
    t = np.arange(n) / SR
    g = np.stack([r + 0.5 * np.sin(2 * np.pi * 0.3 * t + i) for i, r in enumerate(RATES)])
    phi = 2.0 * np.pi * np.cumsum(g, axis=1) / SR
    mono = 0.02 * rng.standard_normal(n)
    for r in range(N_ROTORS):
        for k in range(1, 9):
            mono += (0.25 / k) * np.sin(k * phi[r])
    gains = np.linspace(0.5, 1.5, N_MICS)[:, None]
    audio = (mono[None, :] * gains + 0.01 * rng.standard_normal((N_MICS, n))).astype(np.float32)

    n_tel = int(DUR_S * 100)
    t_tel = np.arange(n_tel) / 100.0
    g_tel = np.stack([np.interp(t_tel, t, g[r]) for r in range(N_ROTORS)])
    return td.Frame(
        {
            "audio": td.uniform(audio, SR, dims=("mic", "time"), t_start=0.0),
            "motors_measured": td.uniform(g_tel, 100, dims=("rotor", "time"), t_start=0.0),
            "motors_command": td.events(
                t_tel, g_tel * 1.004, dims=("rotor", "time"), t_start=0.0, t_end=DUR_S
            ),
            "meta": td.Frame({"recording_id": "SYNTH"}),
        }
    )


@pytest.fixture(scope="module")
def payload() -> dict:
    p, _found = build_widget_payload(_frame(), **BUILD)
    return p


# ─── discovery ────────────────────────────────────────────────────────────────


def test_discover_finds_the_audio_and_both_rotor_tracks():
    found = discover(_frame())
    assert found.audio_key == "audio"
    assert found.n_mics == N_MICS
    assert found.n_rotors == N_ROTORS
    assert found.rps_keys == ["motors_measured", "motors_command"]


# ─── defect 3: every microphone is selectable ─────────────────────────────────


def test_every_microphone_is_a_selectable_channel(payload):
    ids = [c["id"] for c in payload["chans"]]
    assert ids == ["avg"] + [f"mic{i:02d}" for i in range(N_MICS)]
    assert [c["file"] for c in payload["chans"]] == [None] * (N_MICS + 1), (
        "in-page channels must not claim a sibling file"
    )
    # each one carries its own spectrogram, and they are DIFFERENT images
    assert set(payload["spec"]) == set(ids)
    means = [payload["spec"][i]["mean"] for i in ids]
    assert len(set(means)) == len(means)


def test_strips_are_built_for_a_subset_only(payload):
    """Selectability is cheap (a spectrogram), strips are not — the default
    must not pay for eight strip stacks to make eight mics selectable."""
    assert set(payload["strips"]) < set(payload["spec"])
    assert "avg" in payload["strips"]
    for ch, stacks in payload["strips"].items():
        assert stacks, f"{ch} is listed with no strip stacks"


def test_strip_channels_is_honoured():
    p, _ = build_widget_payload(_frame(), channels="avg,0,1", strip_channels="1", **BUILD)
    assert [c["id"] for c in p["chans"]] == ["avg", "mic00", "mic01"]
    assert set(p["strips"]) == {"mic01"}


# ─── defect 4: no header text, rotor-series selector in the top row ───────────


def test_page_starts_with_the_controls_not_a_wall_of_text(payload):
    html = render_html(payload)
    body = html.split("</style>", 1)[1]
    assert "<h1>" not in body
    head, _ = body.split('<canvas id="spec"', 1)
    assert 'class="sub"' not in head, "the description block is still above the figure"
    assert '<div class="prov"' not in head, "provenance is still above the figure"
    # ... but it is still ON the page, one click away: this figure is evidence
    assert 'id="prov"' in body and "<details" in body


def test_the_rotor_series_selector_is_in_the_top_row(payload):
    body = render_html(payload).split("</style>", 1)[1]
    first_row = body.split('<div class="row">', 2)[1]
    for control in ('id="chan"', 'id="car"', 'id="tf"', 'id="alt"'):
        assert control in first_row, f"{control} is not in the top selector row"
    strip_row = body.split('<div class="row">')[-1]
    assert 'id="car"' not in strip_row


# ─── defect 1: the cell has a bounded height ──────────────────────────────────


def test_iframe_declares_a_starting_and_a_maximum_height(payload):
    html = widget_html(payload, height=900, max_height=1800)
    assert 'data-max-height="1800"' in html
    assert "height:900px" in html and "max-height:1800px" in html


def test_the_page_never_reads_documentelement_scrollheight():
    """The growth bug was ``documentElement.scrollHeight`` (>= the viewport,
    which inside an iframe is the frame itself) written back into the frame."""
    from plots.comb_page import SCRIPT

    assert "documentElement.scrollHeight" not in SCRIPT


def test_details_summary_is_styled_so_the_panel_is_collapsible():
    assert "details.panel>summary" in CSS


# ─── the page itself, executed ────────────────────────────────────────────────


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_page_script_runs_and_holds_every_invariant(payload, tmp_path):
    """Run ``verify_page.js`` over the widget page.

    That harness executes the page's JavaScript against a stubbed DOM and
    drives every render path, then asserts the four properties this module's
    defects were about: one bounded, non-growing frame height over 25 redraws;
    no dashed or dotted overlay stroke left after reset; a channel selector
    whose choice changes the decoded image in use; and no page warning.
    """
    page = tmp_path / "widget.html"
    page.write_text(widget_html(payload))
    proc = subprocess.run(
        ["node", str(ROOT / "scripts/displacement/verify_page.js"), str(page)],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ALL PAGES OK" in proc.stdout


def test_split_payload_survives_a_channel_without_strips(payload):
    """The file-per-channel CLI splits the same payload: a channel that carries
    a spectrogram and no strips must give a page, not a KeyError."""
    from plots.comb_page import split_payload

    pages = split_payload(payload, lambda cid: f"page_{cid}.html")
    assert set(pages) == set(payload["spec"])
    assert pages["mic01"]["strips"] == {}
    assert set(pages["avg"]["strips"]) == {"avg"}
    for page in pages.values():
        assert [c["file"] for c in page["chans"]] == [f"page_{c['id']}.html" for c in page["chans"]]


def test_payload_is_json_serialisable_and_small(payload):
    """A notebook saves this into the ``.ipynb``; eight selectable microphones
    must not turn a page into a hundred megabytes."""
    n = len(json.dumps(payload))
    assert n < 12e6, f"{n / 1e6:.1f} MB for a 2 s synthetic frame"
    assert re.match(r"comb_page\.py v\d", payload["meta"]["code_version"])
