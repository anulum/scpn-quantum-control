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
from typing import Any, Callable

# Resolve imports when run from this directory
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from campaign_io import campaign_path
from test_multi_backend_distributed import run_multi_backend
from test_pt_symmetric_kuramoto import run_pt_symmetric
from test_quantum_advantage_scaling import run_advantage_scaling

CredibleTest = tuple[str, Callable[[], Any]]


def _default_tests() -> list[CredibleTest]:
    return [
        ("T1_quantum_advantage_scaling", run_advantage_scaling),
        ("T4_multi_backend_distributed", run_multi_backend),
        ("T7_pt_symmetric_kuramoto", run_pt_symmetric),
    ]


def _summary_status(summary: dict[str, Any]) -> str:
    if summary["counts"]["failed"] > 0:
        return "completed_with_failures"
    return "completed"


async def run_credible_tests(
    tests: list[CredibleTest] | None = None,
    results_dir: str | Path | None = None,
) -> dict[str, Any]:
    token = os.environ.get("SCPN_IBM_TOKEN")
    if not token:
        print(
            "SCPN_IBM_TOKEN is not set. Export it first:\n    export SCPN_IBM_TOKEN='your_token'"
        )
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = Path(results_dir) if results_dir is not None else campaign_path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "runner": "credible_observables_only (DLAParityWitness, SyncOrderParameter)",
        "start_time": timestamp,
        "tests": {},
        "counts": {"success": 0, "failed": 0},
    }

    tests = _default_tests() if tests is None else tests

    print("Running 3 scientifically credible frontier tests")
    print("   Observables: DLAParityWitness + SyncOrderParameter (computed from real QPU counts)")
    print(f"   Timestamp: {timestamp}\n")

    for name, fn in tests:
        t0 = time.time()
        print(f"{name} ...")
        try:
            if asyncio.iscoroutinefunction(fn):
                await fn()
            else:
                fn()
            runtime = round(time.time() - t0, 2)
            summary["tests"][name] = {"status": "success", "runtime_s": runtime}
            summary["counts"]["success"] += 1
            print(f"{name} completed in {runtime}s\n")
        except Exception as e:
            runtime = round(time.time() - t0, 2)
            summary["tests"][name] = {"status": "failed", "runtime_s": runtime, "error": str(e)}
            summary["counts"]["failed"] += 1
            print(f"{name} failed after {runtime}s - {e}\n")

    summary["total_runtime_s"] = round(sum(v["runtime_s"] for v in summary["tests"].values()), 2)
    summary["status"] = _summary_status(summary)

    out = results_dir / f"credible_run_summary_{timestamp}.json"
    out.write_text(json.dumps(summary, indent=2))
    summary["summary_path"] = str(out)
    print(f"Summary -> {out}")
    return summary


if __name__ == "__main__":
    asyncio.run(run_credible_tests())
