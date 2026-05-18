#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- synchronisation benchmark regeneration gate
"""Regenerate all synchronisation benchmark artefacts and compare them."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scpn_quantum_control.benchmark_harness.synchronisation_runner import (
    CHAIN_N8_BENCHMARK_ID,
    RING_N4_BENCHMARK_ID,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_IDS = (RING_N4_BENCHMARK_ID, CHAIN_N8_BENCHMARK_ID)


def run_command(command: list[str]) -> None:
    """Run one gate command and fail closed on non-zero status."""

    print(f"[sync-benchmark-gate] {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    """Run the full no-QPU synchronisation benchmark gate."""

    run_command([sys.executable, "scripts/export_synchronisation_benchmark_registry.py"])
    for benchmark_id in BENCHMARK_IDS:
        run_command(
            [
                sys.executable,
                "scripts/run_synchronisation_benchmark.py",
                "--benchmark-id",
                benchmark_id,
            ]
        )
    run_command([sys.executable, "scripts/compare_synchronisation_benchmark.py"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
