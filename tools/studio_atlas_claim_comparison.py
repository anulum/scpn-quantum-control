# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — committed claim comparisons for the Studio Atlas
"""Build, write and check the claim comparison record the Studio Atlas shows.

The declared values (``data/studio/atlas_claim_comparison_values_20261123.json``)
each cite the committed field they come from. The build verifies every value
against its source, compares the declared pairs only under compatible
assumptions (:func:`scpn_quantum_control.atlas.comparability.build_comparison_record`)
and seals the result.

Run ``python tools/studio_atlas_claim_comparison.py --write`` after the declared
values or a cited source change, and ``--check`` to compare the committed record
with a fresh build.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from scpn_quantum_control.atlas.comparability import build_comparison_record
from scpn_quantum_control.studio.committed_data import run_committed_data_cli

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
VALUES_PATH: Final[Path] = ROOT / "data/studio/atlas_claim_comparison_values_20261123.json"
DEFAULT_PATH: Final[Path] = ROOT / "data/studio/atlas_claim_comparisons_20261123.json"


def build_committed_claim_comparisons() -> dict[str, object]:
    """Build the sealed comparison record of the declared values.

    Returns
    -------
    dict[str, object]
        The record.

    """
    return build_comparison_record(json.loads(VALUES_PATH.read_text(encoding="utf-8")), ROOT)


def main(argv: Sequence[str] | None = None) -> int:
    """Print, write or check the committed claim comparison record.

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
        build=build_committed_claim_comparisons,
        default_path=DEFAULT_PATH,
        label="Atlas claim comparisons",
        description=__doc__,
    )


if __name__ == "__main__":
    sys.exit(main())
