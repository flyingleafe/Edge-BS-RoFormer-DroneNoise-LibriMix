#!/usr/bin/env python3
"""Validate the per-experiment documentation contract (pre-commit hook).

Two-tier experiment docs:

* **Per-experiment doc** — one ``conf/experiment/<name>.md`` beside every
  ``conf/experiment/<name>.yaml``. Carries YAML frontmatter linking the config
  to its *experiment-batch* write-up, plus a Motivation/Conclusion body.
* **Experiment-batch doc** — a ``docs/experiments/<batch>.md`` narrative
  (motivation / results / conclusion) shared by a group of related
  experiments.

This script enforces, for every ``conf/experiment/*.yaml``:

1. a sibling ``.md`` exists;
2. it opens with a ``---``-delimited YAML frontmatter block;
3. the frontmatter has ``experiment`` == the basename, ``training_config`` ==
   the sibling ``.yaml`` (which must exist), and ``batch`` pointing at an
   existing batch doc under ``docs/experiments/`` (not an ``AGENTS.md`` and not
   under ``bets/``);
4. an optional ``eval_config`` (if present) points at an existing file;
5. the body contains a ``## Motivation`` and a ``## Conclusion`` heading.

It also flags orphan ``conf/experiment/*.md`` with no matching ``.yaml``.

Pure stdlib (no PyYAML) so it runs under any interpreter the hook provides.
Exit code 0 = all good, 1 = violations (printed).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXP_DIR = REPO / "conf" / "experiment"
BATCH_DIR = REPO / "docs" / "experiments"

REQUIRED_KEYS = ("experiment", "training_config", "batch")
REQUIRED_HEADINGS = ("## Motivation", "## Conclusion")


def parse_frontmatter(text: str) -> tuple[dict[str, str] | None, str]:
    """Return (frontmatter_dict, body). ``frontmatter_dict`` is None if the
    file does not open with a ``---`` block. Flat ``key: value`` only; lines
    starting with ``#`` inside the block are treated as comments (so an
    optional key can be left commented out)."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, text
    fm: dict[str, str] = {}
    body_start = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body_start = i + 1
            break
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        fm[key.strip()] = value.strip().strip("'\"")
    if body_start is None:
        return None, text  # unterminated frontmatter
    return fm, "\n".join(lines[body_start:])


def validate() -> list[str]:
    errors: list[str] = []
    if not EXP_DIR.is_dir():
        return [f"experiment config dir not found: {EXP_DIR.relative_to(REPO)}"]

    yamls = sorted(EXP_DIR.glob("*.yaml"))
    md_stems = {p.stem for p in EXP_DIR.glob("*.md")}
    yaml_stems = {p.stem for p in yamls}

    # Orphan docs (a .md with no matching .yaml).
    for stem in sorted(md_stems - yaml_stems):
        errors.append(f"orphan doc conf/experiment/{stem}.md has no matching {stem}.yaml")

    for yaml in yamls:
        name = yaml.stem
        doc = EXP_DIR / f"{name}.md"
        rel = f"conf/experiment/{name}.md"
        if not doc.exists():
            errors.append(f"missing per-experiment doc: {rel} (for {yaml.name})")
            continue

        fm, body = parse_frontmatter(doc.read_text())
        if fm is None:
            errors.append(f"{rel}: missing/!unterminated `---` YAML frontmatter block")
            continue

        for key in REQUIRED_KEYS:
            if key not in fm or not fm[key]:
                errors.append(f"{rel}: frontmatter missing required key `{key}`")

        if fm.get("experiment") not in (None, name):
            errors.append(
                f"{rel}: frontmatter `experiment: {fm.get('experiment')}` != basename `{name}`"
            )

        tc = fm.get("training_config")
        if tc:
            if tc != f"conf/experiment/{name}.yaml":
                errors.append(
                    f"{rel}: `training_config: {tc}` should be conf/experiment/{name}.yaml"
                )
            elif not (REPO / tc).exists():
                errors.append(f"{rel}: `training_config` path does not exist: {tc}")

        ec = fm.get("eval_config")
        if ec and not (REPO / ec).exists():
            errors.append(f"{rel}: `eval_config` path does not exist: {ec}")

        batch = fm.get("batch")
        if batch:
            bpath = REPO / batch
            if not batch.startswith("docs/experiments/"):
                errors.append(f"{rel}: `batch` must live under docs/experiments/: {batch}")
            elif not bpath.exists():
                errors.append(f"{rel}: `batch` doc does not exist: {batch}")
            elif bpath.name == "AGENTS.md" or "/bets/" in batch:
                errors.append(f"{rel}: `batch` must be a real batch write-up, not {batch}")

        missing_h = [h for h in REQUIRED_HEADINGS if h not in body]
        if missing_h:
            errors.append(f"{rel}: body missing required heading(s): {', '.join(missing_h)}")

    return errors


def main() -> int:
    errors = validate()
    if errors:
        print(f"experiment-doc validation FAILED ({len(errors)} issue(s)):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        n = len(list(EXP_DIR.glob("*.yaml"))) if EXP_DIR.is_dir() else 0
        print(
            f"\nEvery conf/experiment/*.yaml ({n} configs) needs a sibling *.md with valid "
            "frontmatter (experiment/training_config/batch) and a ## Motivation + ## Conclusion "
            "body. See scripts/validate_experiment_docs.py for the full contract.",
            file=sys.stderr,
        )
        return 1
    n = len(list(EXP_DIR.glob("*.yaml")))
    print(
        f"experiment-doc validation OK: {n} experiment configs each have a valid per-experiment doc."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
