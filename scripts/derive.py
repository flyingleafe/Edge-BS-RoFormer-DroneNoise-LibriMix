#!/usr/bin/env python
"""Materialize or adopt dload *derived datasets* declared in
``data_processing.derivations.SPECS``.

Three subcommands:

- ``list``                — show every spec, its offline fingerprint, whether
                            it is adopt-only, and whether its derivation ref is
                            already published.
- ``derive <name>``       — run the pipeline once and commit it as a memoized
                            derived version (``repo.derive``), forwarding the
                            ``sample-dir-v1`` manifest meta + the spec as the
                            recipe, then pin it in ``dload.lock``. Refuses
                            ``adopt_only`` specs unless ``--force``.
- ``adopt <name>``        — *adopt in place*: point the derivation ref at the
                            dataset's existing ``dload.lock`` pin instead of
                            re-materializing (the historical ``DREGON-LM-V4-*``
                            uploads). Offline except the ref write; **dry-run by
                            default**, writes only with ``--commit``.

Run from the repo root inside the project venv (so ``data_processing`` and the
R2 credentials in ``.env`` are importable/available). After any change to
``dload.lock``: ``git add dload.lock`` and commit+push. See
``docs/derived-datasets-plan.md`` and ``docs/data-and-artifacts.md``.
"""

from __future__ import annotations

import argparse
import json
import sys

import dload

from data_processing.derivations import (
    SPECS,
    build_pipeline,
    dataset_meta,
    fingerprint,
    spec_layout,
)
from data_processing.streams import open_repository


def _derived_ref(name: str, fp: str) -> str:
    # Mirror dload.repo._derived_key (kept inline to avoid a private import).
    return f"datasets/{name}/derived/{fp}"


def _ref_target(repo, ref: str) -> str | None:
    try:
        return repo.remote.get_bytes(ref).decode().strip()
    except dload.NotFoundError:
        return None


def cmd_list(args: argparse.Namespace) -> int:
    repo = None
    if args.check_remote:
        repo = open_repository()
    for name, entry in SPECS.items():
        fp = fingerprint(name)
        flags = "adopt-only" if entry["adopt_only"] else "derivable"
        status = ""
        if repo is not None:
            target = _ref_target(repo, _derived_ref(name, fp))
            status = "  ref=" + (target[:12] if target else "UNPUBLISHED")
        print(f"{name:32s} {flags:11s} fp={fp[:16]}{status}")
        if args.verbose:
            print(f"    {entry['note']}")
    return 0


def cmd_derive(args: argparse.Namespace) -> int:
    name = args.name
    entry = SPECS[name]
    if entry["adopt_only"] and not args.force:
        print(
            f"{name} is adopt_only (see its note): use `adopt`, or `derive "
            "--force` to materialize a genuinely new version anyway.",
            file=sys.stderr,
        )
        return 2
    repo = open_repository()
    pipe = build_pipeline(name)
    recipe = json.dumps({"name": name, **entry}, indent=2, sort_keys=True)
    dataset = repo.derive(
        name,
        pipe,
        meta=dataset_meta(name),
        recipe=recipe,
        progress=print,
    )
    version = dataset.manifest.version
    print(f"materialized {name} -> {version}")
    if not args.no_pin:
        repo.pin(name, version)
        print(f"pinned {name} = {version} in dload.lock (remember: git add dload.lock)")
    return 0


def cmd_adopt(args: argparse.Namespace) -> int:
    name = args.name
    fp = fingerprint(name)
    ref = _derived_ref(name, fp)
    repo = open_repository()

    version = repo._read_lock().get(name)
    if not version:
        print(
            f"no historical pin for {name} in dload.lock — nothing to adopt "
            "(materialize it with `derive` instead).",
            file=sys.stderr,
        )
        return 2

    # Sanity: the pinned version must exist and already carry the layout the
    # spec declares — otherwise the adopted ref would resolve to a semantically
    # wrong manifest.
    manifest = repo.manifest(name, version)
    layout = (manifest.meta or {}).get("layout")
    want_layout = spec_layout(name)
    print(f"{name}")
    print(f"  fingerprint : {fp}")
    print(f"  ref key     : {ref}")
    print(f"  pinned ver  : {version}")
    print(
        f"  manifest    : layout={layout!r} fields={sorted((manifest.meta or {}).get('fields', {}))}"
    )
    if layout != want_layout:
        print(
            f"  REFUSING: pinned manifest layout is {layout!r}, expected "
            f"{want_layout!r}; the spec would not match.",
            file=sys.stderr,
        )
        return 2

    existing = _ref_target(repo, ref)
    if existing == version:
        print("  already adopted (ref points at the pinned version).")
        return 0
    if existing and existing != version:
        print(f"  NOTE: ref currently points at {existing[:12]}, will repoint to {version[:12]}.")

    if not args.commit:
        print("  DRY-RUN: pass --commit to write this derivation ref to R2.")
        return 0

    repo.remote.put_bytes(ref, version.encode())
    print(f"  wrote ref {ref} -> {version}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="show specs + fingerprints")
    p_list.add_argument(
        "--check-remote", action="store_true", help="also query which refs are published"
    )
    p_list.add_argument("-v", "--verbose", action="store_true", help="print each spec's note")
    p_list.set_defaults(func=cmd_list)

    p_derive = sub.add_parser("derive", help="materialize + pin a derived dataset")
    p_derive.add_argument("name", choices=list(SPECS))
    p_derive.add_argument("--force", action="store_true", help="materialize even if adopt_only")
    p_derive.add_argument("--no-pin", action="store_true", help="do not touch dload.lock")
    p_derive.set_defaults(func=cmd_derive)

    p_adopt = sub.add_parser("adopt", help="point the derivation ref at the existing pin")
    p_adopt.add_argument("name", choices=list(SPECS))
    p_adopt.add_argument(
        "--commit", action="store_true", help="actually write the ref (default: dry-run)"
    )
    p_adopt.set_defaults(func=cmd_adopt)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
