#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Phase 2.5 final burn
"""Phase 2.5: final burn of remaining ~49s to fully exhaust cycle.

Dashboard state before this run:
  Usage: 9m 11s, Remaining: 49s, Cycle: Mar 13 – Apr 10, 2026

Observed QPU cost rate: ~0.557 s/circuit (from 117s for 210 circuits
across Phase 1.5 + Phase 2).

Target: 90 circuits at estimated ~50s → fully exhaust the 49s remaining.

Scientific value: reinforce the strongest-signal depths [4, 6, 8, 10, 14]
on n=4 with 9 additional reps each (rep 12-20). Combined with prior
12 reps, this gives 21 reps per point at the key depths — publication-
grade statistics for the primary DLA parity asymmetry claim.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from phase1_mini_bench_ibm_kingston import (  # noqa: E402
    T_STEP,
    aggregate_experiment_a,
    analyse_counts,
    build_xy_trotter_circuit,
    parse_vault,
)

BACKEND_NAME = "ibm_kingston"
EXPERIMENT_NAME = "phase2_5_final_burn"


def build_experiment_j() -> list[tuple[dict, object]]:
    """J: Final burn on strongest-signal depths, 9 reps."""
    circuits: list[tuple[dict, object]] = []
    depths = [4, 6, 8, 10, 14]
    sectors = {"even": "0011", "odd": "0001"}
    reps = list(range(12, 21))  # 9 new reps, continuing from Phase 2 (11)
    for depth in depths:
        for sector_name, init in sectors.items():
            for rep in reps:
                meta = {
                    "experiment": "A_dla_parity_n4",  # same label → merges
                    "n_qubits": 4,
                    "depth": depth,
                    "sector": sector_name,
                    "initial": init,
                    "rep": rep,
                    "shots": 2048,
                    "t_step": T_STEP,
                    "phase": "2.5",
                }
                qc = build_xy_trotter_circuit(4, init, depth, T_STEP)
                qc.name = f"J_n4_d{depth}_{sector_name}_r{rep}"
                circuits.append((meta, qc))
    return circuits


def load_prior_results(paths: list[Path]) -> list[dict]:
    combined: list[dict] = []
    for p in paths:
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)
        combined.extend(data.get("circuits", []))
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2.5 final burn")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backend", default=BACKEND_NAME)
    parser.add_argument("--shots", type=int, default=2048)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    results_dir = REPO_ROOT / ".coordination" / "ibm_runs"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"phase2_5_final_burn_{timestamp}.json"

    circuits_j = build_experiment_j()
    all_circuits = circuits_j

    print("=" * 60)
    print("Phase 2.5 — Final Cycle Burn")
    print("=" * 60)
    print(f"Timestamp: {timestamp}")
    print(f"Backend:   {args.backend}")
    print(f"Dry-run:   {args.dry_run}")
    print()
    print(f"Experiment J (n=4 final reps 12-20): {len(circuits_j)} circuits")
    print(f"TOTAL: {len(all_circuits)} circuits")
    print()
    print("Dashboard before: 9m 11s used, 49s remaining.")
    print(f"Estimated cost: {len(all_circuits) * 0.557:.1f}s")
    print()

    circuits_only = [c for _, c in all_circuits]
    metas = [m for m, _ in all_circuits]

    if args.dry_run:
        print("DRY RUN — transpile locally")
        from scpn_quantum_control.hardware.runner import HardwareRunner

        runner = HardwareRunner(
            use_simulator=True, results_dir=str(REPO_ROOT / "results/ibm_runs")
        )
        runner.connect()
        all_isa = [runner.transpile(qc) for qc in circuits_only]
        depths_isa = [c.depth() for c in all_isa]
        print(
            f"Transpile OK. ISA depths: min={min(depths_isa)}, "
            f"max={max(depths_isa)}, mean={np.mean(depths_isa):.1f}"
        )
        return 0

    vault = Path("/media/anulum/724AA8E84AA8AA75/agentic-shared/CREDENTIALS.md")
    api_key, instance = parse_vault(vault)
    print("Credentials loaded.")
    print("Connecting to IBM Cloud...")

    from scpn_quantum_control.hardware.runner import HardwareRunner

    runner = HardwareRunner(
        token=api_key,
        channel="ibm_cloud",
        instance=instance,
        backend_name=args.backend,
        use_simulator=False,
        optimization_level=2,
        resilience_level=0,
        results_dir=str(REPO_ROOT / "results/ibm_runs"),
    )
    runner.connect()
    print(f"Connected: {runner.backend_name}")
    print()

    with open(results_path, "w") as f:
        json.dump(
            {"status": "submitted", "timestamp": timestamp, "metas": metas},
            f,
            indent=2,
            default=str,
        )
    print(f"Pre-save: {results_path}")
    print()

    print(f"Submitting {len(circuits_only)} circuits (shots={args.shots})...")
    print("Expected to exhaust the cycle. Do not interrupt.")
    print()

    t0 = time.time()
    try:
        batch = runner.run_sampler(
            circuits_only,
            shots=args.shots,
            name=EXPERIMENT_NAME,
            timeout_s=1500,
        )
    except Exception as e:
        print(f"ERROR: Submission failed: {e}", file=sys.stderr)
        with open(results_path, "w") as f:
            json.dump(
                {"status": "failed", "error": str(e), "metas": metas},
                f,
                indent=2,
                default=str,
            )
        return 1
    wall = time.time() - t0
    job_id = batch[0].job_id if batch else "unknown"
    print(f"Batch done in {wall:.1f}s. Job: {job_id}")
    print()

    all_results_raw: list[dict] = []
    for meta, jr in zip(metas, batch):
        stats = analyse_counts(jr.counts or {}, meta)
        all_results_raw.append(
            {"meta": meta, "counts": jr.counts, "stats": stats, "job_id": jr.job_id}
        )

    # Joint aggregation across ALL phases
    prior_files = [
        REPO_ROOT / ".coordination/ibm_runs/phase1_bench_2026-04-10T183728Z.json",
        REPO_ROOT / ".coordination/ibm_runs/phase1_5_reinforce_2026-04-10T184909Z.json",
        REPO_ROOT / ".coordination/ibm_runs/phase2_exhaust_2026-04-10T185634Z.json",
    ]
    prior_circuits = load_prior_results(prior_files)
    combined = prior_circuits + all_results_raw
    print(
        f"Joint analysis: prior {len(prior_circuits)} + "
        f"Phase 2.5 {len(all_results_raw)} = {len(combined)} circuits"
    )

    agg_combined = aggregate_experiment_a(combined)
    print()
    print("=" * 70)
    print("FINAL JOINT Aggregation (all phases) — n=4 DLA parity")
    print("=" * 70)
    print(f"{'depth':<6}{'leak_even':<14}{'leak_odd':<14}{'asym_rel':<14}{'n_reps':<8}")
    for row in agg_combined:
        star = ""
        if row["depth"] in [4, 6, 8, 10, 14]:
            star = "  (**)"
        print(
            f"{row['depth']:<6}{row['mean_leakage_even']:<14.4f}"
            f"{row['mean_leakage_odd']:<14.4f}"
            f"{row['asymmetry_relative']:<+14.4f}{row['n_reps']:<8}{star}"
        )
    print()
    print("(**) = Phase 2.5 reinforced depths (21 reps total)")
    print()

    output = {
        "experiment": EXPERIMENT_NAME,
        "timestamp_utc": timestamp,
        "backend": args.backend,
        "job_ids": [job_id],
        "wall_time_s": wall,
        "n_circuits": len(all_circuits),
        "t_step": T_STEP,
        "circuits": all_results_raw,
        "aggregated_joint_all_phases_final": {"experiment_A_dla_parity_n4": agg_combined},
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved: {results_path}")

    log_path = REPO_ROOT / ".coordination" / "IBM_EXECUTION_LOG.md"
    with open(log_path, "a") as f:
        f.write(f"\n## {timestamp} — PHASE 2.5 FINAL BURN\n\n")
        f.write(f"- **Experiment:** {EXPERIMENT_NAME}\n")
        f.write(f"- **Backend:** {args.backend}\n")
        f.write(f"- **Circuits:** {len(all_circuits)} (n=4 strongest depths, 9 new reps)\n")
        f.write(f"- **Job ID:** `{job_id}`\n")
        f.write(f"- **Wall time:** {wall:.1f}s\n")
        f.write(f"- **Results file:** `{results_path.relative_to(REPO_ROOT)}`\n")
        f.write("- **Final joint DLA parity (all phases):**\n")
        for row in agg_combined:
            marker = " [REINFORCED]" if row["depth"] in [4, 6, 8, 10, 14] else ""
            f.write(
                f"  - depth {row['depth']:3d}: "
                f"leak_even={row['mean_leakage_even']:.4f}, "
                f"leak_odd={row['mean_leakage_odd']:.4f}, "
                f"asym_rel={row['asymmetry_relative']:+.4f} "
                f"(n={row['n_reps']} reps){marker}\n"
            )
        f.write("- **Purpose:** Complete cycle exhaust → 180-min promo unlock.\n")
    print(f"Log appended: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
