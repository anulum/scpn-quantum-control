#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts & Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Frontier Campaign Orchestrator

"""
Master Orchestrator — Frontier Advanced Tests (Batch 4)
Runs all 8 frontier tests cleanly in sequence.
Uses only real analysis classes and StructuredAnsatz (no mocks).
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from test_dla_tensor_network import run_dla_tn_mapping
from test_live_scneurocore_loop import run_live_scneurocore
from test_logical_sync_protection import run_logical_protection
from test_multi_backend_distributed import run_multi_backend
from test_pt_symmetric_kuramoto import run_pt_symmetric

# Import run functions from each test
from test_quantum_advantage_scaling import run_advantage_scaling
from test_rl_pulse_optimization import run_rl_pulse_opt
from test_sync_distillation import run_distillation


async def run_frontier_campaign():
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    campaign_dir = Path("results/frontier_campaign")
    campaign_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "campaign": "Frontier Advanced Tests (Batch 4)",
        "start_time": timestamp,
        "tests": {},
        "total_runtime_seconds": 0.0,
        "status": "completed",
    }

    tests = [
        ("1_quantum_advantage_scaling", run_advantage_scaling),
        ("2_live_scneurocore_loop", run_live_scneurocore),
        ("3_sync_distillation", run_distillation),
        ("4_multi_backend_distributed", run_multi_backend),
        ("5_dla_tensor_network", run_dla_tn_mapping),
        ("6_rl_pulse_optimization", run_rl_pulse_opt),
        ("7_pt_symmetric_kuramoto", run_pt_symmetric),
        ("8_logical_sync_protection", run_logical_protection),
    ]

    print("Starting Frontier Campaign (Batch 4) - 8 advanced tests\n")

    for name, test_func in tests:
        test_start = time.time()
        print(f"Running {name}...")

        try:
            if asyncio.iscoroutinefunction(test_func):
                await test_func()
            else:
                test_func()

            runtime = time.time() - test_start
            summary["tests"][name] = {
                "status": "success",
                "runtime_seconds": round(runtime, 3),
            }
            print(f"{name} completed in {runtime:.2f}s\n")

        except Exception as e:
            runtime = time.time() - test_start
            summary["tests"][name] = {
                "status": "failed",
                "runtime_seconds": round(runtime, 3),
                "error": str(e),
            }
            print(f"{name} failed after {runtime:.2f}s - {e}\n")

    total_runtime = time.time() - start_time
    summary["total_runtime_seconds"] = round(total_runtime, 3)

    summary_path = campaign_dir / f"campaign_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Frontier Campaign finished in {total_runtime:.2f} seconds.")
    print(f"Summary saved to {summary_path}")
    print(
        "\nYou can now inspect the individual JSON results in results/ and results/frontier_campaign/"
    )


if __name__ == "__main__":
    asyncio.run(run_frontier_campaign())
