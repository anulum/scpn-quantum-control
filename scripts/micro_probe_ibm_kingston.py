#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Micro Probe
"""Micro-batch probe: 4 circuits to test cycle exhaust state.

After Phase 2.5, dashboard should be at or near 10m 0s. This submits
4 small circuits (depth 6, n=4, both sectors, 2 reps) to:
  1. Test whether the cycle is actually exhausted (submission would fail)
  2. If still allowed, add minimal reinforcement to depth=6 data

Expected cost: ~2-3 seconds. Minimal risk.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from phase1_mini_bench_ibm_kingston import (  # noqa: E402
    T_STEP,
    analyse_counts,
    build_xy_trotter_circuit,
    parse_vault,
)


def main() -> int:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    results_dir = REPO_ROOT / ".coordination" / "ibm_runs"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"micro_probe_{timestamp}.json"

    # 4 circuits: n=4, depth 6, 2 sectors, 2 reps
    circuits = []
    metas = []
    for rep in [21, 22]:  # continue rep index
        for sector, init in [("even", "0011"), ("odd", "0001")]:
            qc = build_xy_trotter_circuit(4, init, 6, T_STEP)
            qc.name = f"probe_n4_d6_{sector}_r{rep}"
            circuits.append(qc)
            metas.append(
                {
                    "experiment": "A_dla_parity_n4",
                    "n_qubits": 4,
                    "depth": 6,
                    "sector": sector,
                    "initial": init,
                    "rep": rep,
                    "shots": 2048,
                    "t_step": T_STEP,
                    "phase": "2.6_probe",
                }
            )

    print("=" * 60)
    print("Micro Probe — 4 circuits")
    print("=" * 60)
    print(f"Timestamp: {timestamp}")
    print()

    vault = Path("/media/anulum/724AA8E84AA8AA75/agentic-shared/CREDENTIALS.md")
    api_key, instance = parse_vault(vault)
    print("Credentials loaded.")
    print("Connecting to IBM Cloud...")

    from scpn_quantum_control.hardware.runner import HardwareRunner

    runner = HardwareRunner(
        token=api_key,
        channel="ibm_cloud",
        instance=instance,
        backend_name="ibm_kingston",
        use_simulator=False,
        optimization_level=2,
        resilience_level=0,
        results_dir=str(REPO_ROOT / "results/ibm_runs"),
    )
    try:
        runner.connect()
    except Exception as e:
        print(f"Connect failed: {e}")
        return 1

    print(f"Connected: {runner.backend_name}")
    print("Submitting 4 circuits, shots=2048...")
    t0 = time.time()
    try:
        results = runner.run_sampler(circuits, shots=2048, name="micro_probe", timeout_s=900)
    except Exception as e:
        err_msg = str(e)
        print(f"\nSUBMISSION FAILED: {err_msg}")
        with open(results_path, "w") as f:
            json.dump(
                {
                    "status": "failed",
                    "error": err_msg,
                    "interpretation": (
                        "If error mentions 'quota', 'exceeded', 'insufficient', "
                        "the cycle is exhausted → 180-min promo should be available."
                    ),
                    "timestamp": timestamp,
                },
                f,
                indent=2,
                default=str,
            )

        # Append to log
        log_path = REPO_ROOT / ".coordination" / "IBM_EXECUTION_LOG.md"
        with open(log_path, "a") as f:
            f.write(f"\n## {timestamp} — MICRO PROBE (failed)\n\n")
            f.write("- **Attempted:** 4 circuits (n=4, d=6, 2 sectors, 2 reps)\n")
            f.write("- **Result:** SUBMISSION FAILED\n")
            f.write(f"- **Error:** `{err_msg}`\n")
            f.write(
                "- **Interpretation:** Cycle likely exhausted. Check dashboard "
                "for 180-min promo availability.\n"
            )
        print(f"Log appended: {log_path}")
        return 2
    wall = time.time() - t0
    print(f"SUCCESS. Batch done in {wall:.1f}s.")
    print(f"Job: {results[0].job_id}")
    print()

    # Analyse
    for meta, jr in zip(metas, results):
        stats = analyse_counts(jr.counts or {}, meta)
        print(
            f"  {meta['sector']:<6} rep={meta['rep']}: leakage={stats.get('parity_leakage', 'N/A'):.4f}"
        )

    # Save
    output = {
        "status": "success",
        "timestamp_utc": timestamp,
        "job_id": results[0].job_id,
        "wall_time_s": wall,
        "circuits": [
            {"meta": m, "counts": r.counts, "stats": analyse_counts(r.counts or {}, m)}
            for m, r in zip(metas, results)
        ],
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")

    log_path = REPO_ROOT / ".coordination" / "IBM_EXECUTION_LOG.md"
    with open(log_path, "a") as f:
        f.write(f"\n## {timestamp} — MICRO PROBE (success)\n\n")
        f.write("- **Experiment:** micro_probe\n")
        f.write("- **Circuits:** 4 (n=4, d=6, 2 sectors, 2 reps)\n")
        f.write(f"- **Job ID:** `{results[0].job_id}`\n")
        f.write(f"- **Wall time:** {wall:.1f}s\n")
        f.write(
            "- **Interpretation:** Submission still accepted. Cycle not fully exhausted yet.\n"
        )
    print(f"Log appended: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
