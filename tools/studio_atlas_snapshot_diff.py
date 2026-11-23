# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — compare two Atlas snapshot pins for the Studio
"""Compare two Atlas snapshot pins and write the diff record the Studio shows.

Each pin is admitted first: its schema, its snapshot contract and its recorded
digest must hold. The record names both snapshot digests and source commits,
holds the diff of :func:`scpn_quantum_control.atlas.diff.diff_atlas_snapshots`
and ends with a canonical record digest the browser recomputes.

Examples
--------
Compare the pin of an earlier commit with the current one::

    git show <commit>:data/studio/atlas_snapshot_20261122.json > before.json
    cp data/studio/atlas_snapshot_20261122.json after.json
    python tools/studio_atlas_snapshot_diff.py --before before.json --after after.json

"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from scpn_quantum_control.atlas import (
    AtlasContractError,
    AtlasSnapshot,
    atlas_snapshot_digest,
    parse_atlas_snapshot,
)
from scpn_quantum_control.atlas.diff import AtlasDiffError, diff_atlas_snapshots, diff_to_wire
from scpn_quantum_control.studio_workspace.canonical import canonical_digest

PIN_SCHEMA: Final[str] = "scpn_qc_studio_atlas_snapshot_pin_v1"
DIFF_RECORD_SCHEMA: Final[str] = "scpn_qc_studio_atlas_snapshot_diff_v1"
RECORD_DIGEST_SCHEMA: Final[str] = "scpn_qc_studio_atlas_snapshot_diff_record_v1"


class PinError(ValueError):
    """A snapshot pin cannot be admitted.

    Parameters
    ----------
    message
        Why the pin was refused.

    """


def read_pin(path: Path) -> tuple[AtlasSnapshot, str]:
    """Admit a snapshot pin: schema, contract and recorded digest.

    Parameters
    ----------
    path
        The pin's JSON file.

    Returns
    -------
    tuple[AtlasSnapshot, str]
        The validated snapshot and its digest.

    Raises
    ------
    PinError
        The file is not JSON, has another schema, breaks the contract, or does
        not match its recorded digest.

    """
    try:
        pin = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PinError(f"{path}: not a readable JSON pin ({error})") from error
    if not isinstance(pin, Mapping) or pin.get("schema") != PIN_SCHEMA:
        raise PinError(f"{path}: not a {PIN_SCHEMA} pin")
    try:
        snapshot = parse_atlas_snapshot(pin.get("snapshot"))
    except AtlasContractError as error:
        raise PinError(f"{path}: the snapshot breaks the contract ({error})") from error
    digest = atlas_snapshot_digest(snapshot)
    if digest != pin.get("snapshot_digest"):
        raise PinError(f"{path}: the snapshot does not match its recorded digest")
    return snapshot, digest


def build_diff_record(before: Path, after: Path) -> dict[str, object]:
    """Compare two pins and return the sealed diff record.

    Parameters
    ----------
    before
        The earlier pin.
    after
        The later pin.

    Returns
    -------
    dict[str, object]
        The record, ending with its canonical ``record_digest``.

    """
    old, old_digest = read_pin(before)
    new, new_digest = read_pin(after)
    record: dict[str, object] = {
        "schema": DIFF_RECORD_SCHEMA,
        "before": {"snapshot_digest": old_digest, "source_commit": old.source_commit},
        "after": {"snapshot_digest": new_digest, "source_commit": new.source_commit},
        "diff": diff_to_wire(diff_atlas_snapshots(old, new)),
    }
    record["record_digest"] = canonical_digest(RECORD_DIGEST_SCHEMA, record)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    """Print or write the diff record of two pins.

    Parameters
    ----------
    argv
        CLI arguments, or the process arguments when omitted.

    Returns
    -------
    int
        0 on success; 2 when a pin is refused, the snapshots cannot be compared or
        the output already exists.

    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--before", type=Path, required=True, help="the earlier snapshot pin")
    parser.add_argument("--after", type=Path, required=True, help="the later snapshot pin")
    parser.add_argument(
        "--output", type=Path, help="write the record here; an existing file is kept"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        record = build_diff_record(args.before, args.after)
    except (PinError, AtlasDiffError) as error:
        print(f"atlas snapshot diff: {error}", file=sys.stderr)
        return 2
    text = json.dumps(record, indent=2, ensure_ascii=False) + "\n"
    if args.output is None:
        sys.stdout.write(text)
        return 0
    try:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(text)
    except FileExistsError:
        print(f"atlas snapshot diff: {args.output} exists and is kept", file=sys.stderr)
        return 2
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
