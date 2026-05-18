#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- stable-core release gate
"""Run the stable-core release gate scripts in deterministic order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CAPABILITY_GATE_SCRIPT = Path("scripts") / "run_stable_core_capability_gate.py"
CONTRACT_GATE_SCRIPT = Path("scripts") / "run_stable_core_contract_gate.py"
PREFLIGHT_GATE_SCRIPT = Path("scripts") / "run_stable_core_preflight_gate.py"


def build_stable_core_release_gate_commands() -> tuple[tuple[str, ...], ...]:
    """Return commands for the stable-core release gate in deterministic order."""

    return (
        (sys.executable, str(CAPABILITY_GATE_SCRIPT)),
        (sys.executable, str(CONTRACT_GATE_SCRIPT)),
        (sys.executable, str(PREFLIGHT_GATE_SCRIPT)),
    )


def run_command(command: tuple[str, ...]) -> None:
    """Run one gate command and fail closed if the command exits non-zero."""

    print(f"[stable-core-release-gate] {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    """Run stable-core release gate."""

    for command in build_stable_core_release_gate_commands():
        run_command(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
