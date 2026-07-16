#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Published-record freeze guard
"""Fail when a published record's source is modified, added, or removed.

Published preprints and paper submissions are read-only (owner ruling,
2026-07-16): a record with a public DOI or a published page is a fixed
historical document. Corrections travel as dated amendments on live
surfaces (``docs/results.md``, the ledger, README) and as new Zenodo
versions — never as in-place edits of the published source.

How it works:
  1. ``data/published_record_freeze.json`` pins the SHA-256 of every file
     belonging to a published record: whole frozen trees (each submission
     directory) plus individual frozen pages.
  2. The guard re-walks the frozen trees and re-hashes every pinned file.
     A hash mismatch, a missing pinned file, or an unpinned file inside a
     frozen tree is a blocking finding.
  3. ``--update`` regenerates the manifest from the working tree. Run it
     only for an owner-approved change (e.g. registering a new published
     record), and say so in the commit message.

Usage:
  python tools/check_published_record_freeze.py            # verify
  python tools/check_published_record_freeze.py --update   # owner-approved refresh
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

DEFAULT_MANIFEST = Path("data/published_record_freeze.json")

#: Whole trees whose every file is part of a published record.
FROZEN_TREES = ("paper/submissions",)

#: Individual published pages outside the frozen trees.
FROZEN_FILES = (
    "docs/preprint.md",
    "docs/paper_sync_witnesses.md",
    "docs/paper_dla_parity.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_paths(root: Path) -> list[Path]:
    """Return every file that belongs to a published record, sorted."""
    paths: list[Path] = []
    for tree in FROZEN_TREES:
        base = root / tree
        if base.is_dir():
            paths.extend(p for p in base.rglob("*") if p.is_file())
    for name in FROZEN_FILES:
        candidate = root / name
        if candidate.is_file():
            paths.append(candidate)
    return sorted(paths)


def build_manifest(root: Path) -> dict[str, str]:
    """Hash every published-record file in the working tree."""
    return {path.relative_to(root).as_posix(): _sha256(path) for path in frozen_paths(root)}


def load_manifest(manifest_path: Path) -> dict[str, str]:
    """Load the pinned manifest."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, dict) or not records:
        raise ValueError(f"{manifest_path}: 'records' must be a non-empty object")
    return {str(path): str(digest) for path, digest in records.items()}


def compare(pinned: dict[str, str], current: dict[str, str]) -> list[str]:
    """Return human-readable blocking findings, empty when frozen state holds."""
    findings: list[str] = []
    for path, digest in sorted(pinned.items()):
        if path not in current:
            findings.append(f"missing published-record file: {path}")
        elif current[path] != digest:
            findings.append(f"published record modified: {path}")
    for path in sorted(set(current) - set(pinned)):
        findings.append(f"unpinned file inside a frozen record tree: {path}")
    return findings


def write_manifest(manifest_path: Path, records: dict[str, str]) -> None:
    """Write the manifest with its policy header."""
    payload = {
        "description": (
            "SHA-256 freeze of published-record sources (preprints, paper "
            "submissions). Read-only by owner ruling 2026-07-16; corrections "
            "go to live surfaces and new Zenodo versions, never in-place. "
            "Regenerate only with owner approval via --update."
        ),
        "records": dict(sorted(records.items())),
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: current directory).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Freeze manifest (default: data/published_record_freeze.json under root).",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Regenerate the manifest from the working tree (owner-approved only).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    root = args.root.resolve()
    manifest_path = args.manifest if args.manifest is not None else root / DEFAULT_MANIFEST
    current = build_manifest(root)
    if args.update:
        write_manifest(manifest_path, current)
        print(f"Published-record freeze manifest written: {len(current)} file(s).")
        return 0
    pinned = load_manifest(manifest_path)
    findings = compare(pinned, current)
    if findings:
        print(f"Published-record freeze guard: {len(findings)} blocking finding(s):")
        for finding in findings:
            print(f"  {finding}")
        print(
            "Published records are read-only (owner ruling 2026-07-16). "
            "Corrections belong on live surfaces or in a new Zenodo version."
        )
        return 1
    print(f"Published-record freeze guard: {len(pinned)} file(s) intact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
