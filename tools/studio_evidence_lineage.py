#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — committed evidence lineage for the Studio Atlas
"""Build, write and check the committed evidence lineage the Studio Atlas shows.

The lineage joins the claim ledger, the reference-validation registry, the
hardware result packs and the declared replication analyses, with every
source-owned status kept verbatim beside the others. The browser shows it
without running the emitters; the Atlas extractor reads it as bytes.

Run ``python tools/studio_evidence_lineage.py --write`` after one of those
sources changes, and ``--check`` to compare the committed file with a fresh
build.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from scpn_quantum_control.studio.committed_data import run_committed_data_cli
from scpn_quantum_control.studio.evidence_lineage import build_evidence_lineage

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_PATH: Final[Path] = ROOT / "data/studio/evidence_lineage_20261123.json"


def build_committed_evidence_lineage() -> dict[str, object]:
    """Build the evidence lineage of this repository.

    Returns
    -------
    dict[str, object]
        The lineage document.

    """
    return build_evidence_lineage(ROOT)


def main(argv: Sequence[str] | None = None) -> int:
    """Print, write or check the committed evidence lineage.

    Parameters
    ----------
    argv
        CLI arguments, or the process arguments when omitted.

    Returns
    -------
    int
        The shared committed-data writer's exit status.

    """
    return run_committed_data_cli(
        argv,
        build=build_committed_evidence_lineage,
        default_path=DEFAULT_PATH,
        label="evidence lineage data",
        description=__doc__,
    )


if __name__ == "__main__":
    sys.exit(main())
