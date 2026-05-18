#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- symmetry-sector mitigation fixture comparator
"""Compare regenerated symmetry-sector mitigation fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scpn_quantum_control.mitigation.symmetry_sector_fixtures import (
    fixture_payload,
    normalised_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPECTED = (
    REPO_ROOT / "data" / "symmetry_sector_mitigation" / "symmetry_sector_mitigation_fixtures.json"
)


def main() -> int:
    """Run fixture comparison."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    expected = args.expected.read_text(encoding="utf-8")
    actual = normalised_json(fixture_payload())
    valid = expected == actual
    result = {
        "schema": "symmetry_sector_mitigation_fixture_comparison_v1",
        "valid": valid,
        "blockers": []
        if valid
        else ["regenerated planner fixtures differ from committed artefact"],
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"symmetry-sector mitigation fixture comparison valid: {valid}")
        for blocker in result["blockers"]:
            print(f"  blocker: {blocker}")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
