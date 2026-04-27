#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts & Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Frontier Credible Test Runner

"""
Focused runner: T1, T4, T7 — tests whose observables are computed
entirely from real QPU bitstring counts (DLAParityWitness, SyncOrderParameter).
These three produce results with direct scientific credibility.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Resolve imports when run from this directory
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from test_multi_backend_distributed import run_multi_backend
from test_pt_symmetric_kuramoto import run_pt_symmetric
from test_quantum_advantage_scaling import run_advantage_scaling


async def run_credible_tests():
    token = os.environ.get("SCPN_IBM_TOKEN")
    if not token:
        print(
            "SCPN_IBM_TOKEN is not set. Export it first:\n    export SCPN_IBM_TOKEN='your_token'"
        )
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    summary = {
        "runner": "credible_observables_only (DLAParityWitness, SyncOrderParameter)",
        "start_time": timestamp,
        "tests": {},
        "status": "completed",
    }

    tests = [
        ("T1_quantum_advantage_scaling", run_advantage_scaling),
        ("T4_multi_backend_distributed", run_multi_backend),
        ("T7_pt_symmetric_kuramoto", run_pt_symmetric),
    ]

    print("Running 3 scientifically credible frontier tests")
    print("   Observables: DLAParityWitness + SyncOrderParameter (computed from real QPU counts)")
    print(f"   Timestamp: {timestamp}\n")

    for name, fn in tests:
        t0 = time.time()
        print(f"{name} ...")
        try:
            await fn()
            runtime = round(time.time() - t0, 2)
            summary["tests"][name] = {"status": "success", "runtime_s": runtime}
            print(f"{name} completed in {runtime}s\n")
        except Exception as e:
            runtime = round(time.time() - t0, 2)
            summary["tests"][name] = {"status": "failed", "runtime_s": runtime, "error": str(e)}
            print(f"{name} failed after {runtime}s - {e}\n")

    summary["total_runtime_s"] = round(sum(v["runtime_s"] for v in summary["tests"].values()), 2)

    out = results_dir / f"credible_run_summary_{timestamp}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Summary -> {out}")


if __name__ == "__main__":
    asyncio.run(run_credible_tests())
