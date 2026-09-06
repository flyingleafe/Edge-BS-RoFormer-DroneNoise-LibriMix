"""Launch the canonical evaluator; extra arguments are passed through unchanged."""

import subprocess
import sys

from utils.checkpoints import resolve_checkpoint_uri

for stage in ("s1", "s2"):
    experiment = f"jhtr_cond_{stage}_nomix"
    checkpoint = resolve_checkpoint_uri(
        f"r2://ml-data/artifacts/{experiment}/checkpoints/best.ckpt"
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/jhtr_campaign.py",
            "evaluate",
            "--experiment",
            experiment,
            "--checkpoint",
            checkpoint,
            "--device",
            "cuda",
            "--batch",
            "32",
            "--out",
            f"results/jhtr-eval-{stage}",
            *sys.argv[1:],
        ],
        check=True,
    )
