"""KAIST rotating-machine acoustic source."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import tdseries as td

from data_processing.sources._common import audio_frame, meta_frame, safe_key


def build(raw_dir: Path) -> Iterator[tuple[str, td.Frame]]:
    """KAIST rotating machine (acoustic.zip): mic sound-pressure in ``.mat``,
    ~51.2 kHz, 0 Nm. Each file has one ``Signal`` struct with ``y_values.values``
    (the pressure vector) and ``x_values.increment`` (sample period). Filename
    ``<load>Nm_<fault>[_<severity>]``."""
    from scipy.io import loadmat

    for mat in sorted(raw_dir.rglob("*.mat")):
        d = loadmat(str(mat), squeeze_me=True, struct_as_record=False)
        sig = d.get("Signal")
        if sig is None or not hasattr(sig, "y_values"):
            continue
        values = np.asarray(sig.y_values.values, dtype=np.float32).reshape(-1)
        if values.size == 0:
            continue
        sr = 51200
        inc = getattr(getattr(sig, "x_values", None), "increment", None)
        if inc is not None and float(inc) > 0:
            sr = int(round(1.0 / float(inc)))
        rid = mat.stem
        toks = rid.split("_")
        load_tok = next((t for t in toks if t.lower().endswith("nm")), None)
        meta = meta_frame(
            rid,
            "KAIST-rotating-acoustic",
            system={"category": "industrial_machine", "machine": "rotating_machine_testbed"},
            observation={
                "type": "fixed_mic_bench",
                "source_motion": "static",
                "relative_trajectory": "none",
            },
            operating={"load": load_tok, "rpm": 3010, "sensor": "PCB378B02"},
            label={
                "fault": toks[1] if len(toks) > 1 else None,
                "severity": toks[2] if len(toks) > 2 else None,
            },
            extra={"raw_relpath": str(mat.relative_to(raw_dir))},
        )
        yield safe_key(rid), audio_frame(values[None, :], sr, meta)


PROVENANCE = {
    "source_url": "https://data.mendeley.com/datasets/ztmf3m7h5x/5",
    "doi": "10.17632/ztmf3m7h5x.5",
    "license": "CC BY 4.0",
    "citation": "Jung et al., Vibration/Acoustic/Temp/Current of Rotating Machine (Data in Brief 48:109049, 2023).",
    "collection_method": "rotating-machine testbed; acoustic mic only at 0 Nm load",
    "equipment": "PCB378B02 microphone, 51.2 kHz",
    "observation_type": "fixed_mic_bench",
    "sample_rate": 51200,
    "channels": 1,
    "description": "Acoustic subset only (acoustic.zip): sound-pressure .mat, 3010 RPM, fault/severity in filename. Vibration/current zips excluded (not audio).",
}
