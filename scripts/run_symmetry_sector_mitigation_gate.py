#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — run symmetry sector mitigation gate script
# scpn-quantum-control -- symmetry-sector mitigation fixture gate
"""Regenerate and compare symmetry-sector mitigation planner fixtures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    """Run one command and fail closed on non-zero status."""

    print(f"[symmetry-sector-gate] {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    """Run the full planner fixture gate."""

    run([sys.executable, "scripts/export_symmetry_sector_mitigation_fixtures.py"])
    run([sys.executable, "scripts/compare_symmetry_sector_mitigation_fixtures.py"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
