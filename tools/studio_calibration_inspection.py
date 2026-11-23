# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — committed calibration inspection for the Studio operators view
"""Build, write and check the calibration inspection record the Studio shows.

The declaration (``data/studio/calibration_inspection_declaration_20261123.json``)
lists committed device calibration snapshots, the committed plans that name
their calibration sets, and each experiment type's freshness policy with the
document that fixes it. The build verifies every file by sha256, maps each
snapshot's elements, compares consecutive snapshots and judges every plan
(:func:`scpn_quantum_control.hardware.calibration_inspection.build_inspection_record`).

Run ``python tools/studio_calibration_inspection.py --write`` after the
declaration or a cited file changes, and ``--check`` to compare the committed
record with a fresh build.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from scpn_quantum_control.hardware.calibration_inspection import build_inspection_record
from scpn_quantum_control.studio.committed_data import run_committed_data_cli

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DECLARATION_PATH: Final[Path] = (
    ROOT / "data/studio/calibration_inspection_declaration_20261123.json"
)
DEFAULT_PATH: Final[Path] = ROOT / "data/studio/calibration_inspection_20261123.json"


def build_committed_calibration_inspection() -> dict[str, object]:
    """Build the sealed inspection record of the declared snapshots and plans.

    Returns
    -------
    dict[str, object]
        The record.

    """
    return build_inspection_record(json.loads(DECLARATION_PATH.read_text(encoding="utf-8")), ROOT)


def main(argv: Sequence[str] | None = None) -> int:
    """Print, write or check the committed calibration inspection record.

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
        build=build_committed_calibration_inspection,
        default_path=DEFAULT_PATH,
        label="Calibration inspection",
        description=__doc__,
    )


if __name__ == "__main__":
    sys.exit(main())
