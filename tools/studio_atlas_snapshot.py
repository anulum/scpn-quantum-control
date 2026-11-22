#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — committed Atlas snapshot for the Studio browser
"""Build, write and check the committed Atlas snapshot the Studio browser loads.

The browser shows the repository's Atlas without running the extractor. This
tool extracts the snapshot from the committed inputs and writes it, with its
canonical digest, to ``data/studio/atlas_snapshot_20261122.json``. The browser
validates the snapshot with the shared contract and refuses it when the digest
differs.

- ``--write --source-commit <full commit>`` records the commit the working tree
  is based on, normally ``git rev-parse HEAD``. The inputs are read from the
  working tree, so a pin written for a change records that change's parent.
- ``--check`` re-extracts with the commit recorded in the committed file and
  reports every difference, so a changed input or extractor is caught.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from scpn_quantum_control.atlas import atlas_snapshot_digest, snapshot_to_wire
from scpn_quantum_control.atlas.extractors import directory_reader, extract_atlas_snapshot
from scpn_quantum_control.studio.committed_data import run_committed_data_cli

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
PIN_SCHEMA: Final[str] = "scpn_qc_studio_atlas_snapshot_pin_v1"
DEFAULT_PATH: Final[Path] = ROOT / "data/studio/atlas_snapshot_20261122.json"
_COMMIT: Final = re.compile(r"[0-9a-f]{40}")


def build_atlas_snapshot_pin(source_commit: str, root: Path = ROOT) -> dict[str, object]:
    """Extract the committed inputs and wrap the snapshot with its digest.

    Parameters
    ----------
    source_commit
        Full commit the inputs belong to.
    root
        Repository root to read the inputs from.

    Returns
    -------
    dict[str, object]
        ``{"schema", "snapshot_digest", "snapshot"}`` with the canonical wire snapshot.

    """
    snapshot = extract_atlas_snapshot(directory_reader(root), source_commit).snapshot
    return {
        "schema": PIN_SCHEMA,
        "snapshot_digest": atlas_snapshot_digest(snapshot),
        "snapshot": snapshot_to_wire(snapshot),
    }


def _recorded_commit(path: Path) -> str | None:
    """Return the source commit recorded in a committed pin, or ``None`` when unreadable."""
    try:
        pin = json.loads(path.read_text(encoding="utf-8"))
        commit = pin["snapshot"]["source_commit"]
    except (OSError, UnicodeError, ValueError, KeyError, TypeError):
        return None
    return commit if isinstance(commit, str) and _COMMIT.fullmatch(commit) else None


def main(argv: Sequence[str] | None = None) -> int:
    """Print, write or check the committed Atlas snapshot.

    Parameters
    ----------
    argv
        CLI arguments, or the process arguments when omitted.

    Returns
    -------
    int
        ``0`` on success, ``1`` when the committed snapshot differs from a fresh
        extraction, ``2`` when the source commit is missing, malformed or unreadable.

    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    source = argparse.ArgumentParser(add_help=False)
    source.add_argument("--source-commit")
    known, rest = source.parse_known_args(arguments)
    location = argparse.ArgumentParser(add_help=False)
    location.add_argument("--json-path", type=Path, default=DEFAULT_PATH)
    located, _unused = location.parse_known_args(rest)
    if "--write" in rest:
        commit = known.source_commit
        if commit is None or not _COMMIT.fullmatch(commit):
            print(
                "atlas snapshot: --write needs --source-commit <full 40-hex commit>",
                file=sys.stderr,
            )
            return 2
    elif known.source_commit is not None:
        print("atlas snapshot: --source-commit applies to --write only", file=sys.stderr)
        return 2
    else:
        commit = _recorded_commit(located.json_path)
        if commit is None:
            print(
                f"atlas snapshot: no readable source commit in {located.json_path}",
                file=sys.stderr,
            )
            return 2
    return run_committed_data_cli(
        rest,
        build=lambda: build_atlas_snapshot_pin(commit),
        default_path=DEFAULT_PATH,
        label="Atlas snapshot",
        description=__doc__,
    )


if __name__ == "__main__":
    sys.exit(main())
