# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — committed capability gap report for the Studio Atlas
"""Build, write and check the capability gap report the Studio Atlas shows.

The report joins each architecture-map capability with the test,
documentation and UI surfaces it declares and the CI jobs that name its tests
(:func:`scpn_quantum_control.atlas.coverage.capability_gap_report`). It lists
gaps with their exact owners and keeps what cannot be known apart; it scores
nothing and opens no ticket.

Run ``python tools/studio_atlas_capability_gaps.py --write`` after the
architecture map, the claim ledger, a declared surface or a CI workflow
changes, and ``--check`` to compare the committed file with a fresh build.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from scpn_quantum_control.atlas.coverage import capability_gap_report, gap_report_to_wire
from scpn_quantum_control.studio.committed_data import run_committed_data_cli

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_PATH: Final[Path] = ROOT / "data/studio/atlas_capability_gaps_20261123.json"
REPORT_SCHEMA: Final[str] = "scpn_qc_studio_atlas_capability_gaps_v1"


def build_committed_capability_gaps() -> dict[str, object]:
    """Build the capability gap report of this repository.

    Returns
    -------
    dict[str, object]
        The schema-tagged report.

    """
    return {"schema": REPORT_SCHEMA, **gap_report_to_wire(capability_gap_report(ROOT))}


def main(argv: Sequence[str] | None = None) -> int:
    """Print, write or check the committed capability gap report.

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
        build=build_committed_capability_gaps,
        default_path=DEFAULT_PATH,
        label="Atlas capability gap report",
        description=__doc__,
    )


if __name__ == "__main__":
    sys.exit(main())
