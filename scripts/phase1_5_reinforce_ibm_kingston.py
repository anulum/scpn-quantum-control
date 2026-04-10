#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Phase 1.5 DLA Parity Reinforcement
"""Phase 1.5 mini-bench: reps expansion + n=8 scaling probe.

Goals:
  D. Increase DLA parity reps from 2 → 6 per point at n=4 (64 circuits)
  E. Probe scaling behaviour at n=8 (8 circuits)

Combined with Phase 1 data, this should give 3σ-level statistics on the
DLA parity asymmetry signal at n=4 depths where it might exist.

Rep indexing starts at rep=2 (Phase 1 used rep 0,1) so Phase 1 + 1.5
combine cleanly.

Budget: ~55s estimated, ~166s available.
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

# Reuse helpers from Phase 1 script
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from phase1_mini_bench_ibm_kingston import (  # noqa: E402
    T_STEP,
    aggregate_experiment_a,
    analyse_counts,
    build_xy_trotter_circuit,
    parse_vault,
)

BACKEND_NAME = "ibm_kingston"
EXPERIMENT_NAME = "phase1_5_reinforce"


def build_experiment_d() -> list[tuple[dict, object]]:
    """D: DLA parity reinforcement — rep 2..5 for Phase 1 grid."""
    circuits: list[tuple[dict, object]] = []
    depths = [2, 4, 6, 8, 10, 14, 20, 30]
    sectors = {"even": "0011", "odd": "0001"}
    reps = [2, 3, 4, 5]  # Continue rep index from Phase 1 (0, 1)
    for depth in depths:
        for sector_name, init in sectors.items():
            for rep in reps:
                meta = {
                    "experiment": "A_dla_parity_n4",  # Same label so it merges with Phase 1
                    "n_qubits": 4,
                    "depth": depth,
                    "sector": sector_name,
                    "initial": init,
                    "rep": rep,
                    "shots": 2048,
                    "t_step": T_STEP,
                    "phase": "1.5",
                }
                qc = build_xy_trotter_circuit(4, init, depth, T_STEP)
                qc.name = f"D_n4_d{depth}_{sector_name}_r{rep}"
                circuits.append((meta, qc))
    return circuits


def build_experiment_e() -> list[tuple[dict, object]]:
    """E: n=8 scaling probe."""
    circuits: list[tuple[dict, object]] = []
    depths = [4, 8]
    sectors = {
        "even": "00000011",  # popcount=2, P=+1
        "odd": "00000001",  # popcount=1, P=-1
    }
    for depth in depths:
        for sector_name, init in sectors.items():
            for rep in range(2):
                meta = {
                    "experiment": "E_scaling_n8",
                    "n_qubits": 8,
                    "depth": depth,
                    "sector": sector_name,
                    "initial": init,
                    "rep": rep,
                    "shots": 2048,
                    "t_step": T_STEP,
                    "phase": "1.5",
                }
                qc = build_xy_trotter_circuit(8, init, depth, T_STEP)
                qc.name = f"E_n8_d{depth}_{sector_name}_r{rep}"
                circuits.append((meta, qc))
    return circuits


def load_phase1_results(
    phase1_path: Path,
) -> list[dict]:
    """Load Phase 1 circuit results for joint aggregation."""
    if not phase1_path.exists():
        return []
    with open(phase1_path) as f:
        data = json.load(f)
    return data.get("circuits", [])


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1.5 reinforcement")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backend", default=BACKEND_NAME)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument(
        "--phase1-file",
        default=".coordination/ibm_runs/phase1_bench_2026-04-10T183728Z.json",
        help="Phase 1 results JSON (for joint aggregation)",
    )
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    results_dir = REPO_ROOT / ".coordination" / "ibm_runs"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"phase1_5_reinforce_{timestamp}.json"

    circuits_d = build_experiment_d()
    circuits_e = build_experiment_e()
    all_circuits = circuits_d + circuits_e

    print("=" * 60)
    print("Phase 1.5 — DLA Parity Reinforcement + n=8 Probe")
    print("=" * 60)
    print(f"Timestamp: {timestamp}")
    print(f"Backend:   {args.backend}")
    print(f"Dry-run:   {args.dry_run}")
    print()
    print(f"Experiment D (n=4 DLA parity reps +4): {len(circuits_d)} circuits")
    print(f"Experiment E (n=8 scaling):           {len(circuits_e)} circuits")
    print(f"TOTAL:                                {len(all_circuits)} circuits")
    print()

    circuits_only = [c for _, c in all_circuits]
    metas = [m for m, _ in all_circuits]

    if args.dry_run:
        print("DRY RUN — transpile locally, run on simulator")
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

    # Real submission
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

    # Pre-save
    with open(results_path, "w") as f:
        json.dump(
            {"status": "submitted", "timestamp": timestamp, "metas": metas},
            f,
            indent=2,
            default=str,
        )
    print(f"Pre-save: {results_path}")
    print()

    # Single batch (all 72 circuits same shots)
    print(f"Submitting {len(circuits_only)} circuits (shots={args.shots})...")
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

    # Parse results
    all_results_raw: list[dict] = []
    for meta, jr in zip(metas, batch):
        stats = analyse_counts(jr.counts or {}, meta)
        all_results_raw.append(
            {"meta": meta, "counts": jr.counts, "stats": stats, "job_id": jr.job_id}
        )

    # Joint aggregation with Phase 1
    phase1_path = REPO_ROOT / args.phase1_file
    phase1_circuits = load_phase1_results(phase1_path)
    combined = phase1_circuits + all_results_raw
    print(
        f"Combined analysis: Phase 1 ({len(phase1_circuits)}) + "
        f"Phase 1.5 ({len(all_results_raw)}) = {len(combined)} circuits"
    )

    agg_combined = aggregate_experiment_a(combined)
    print()
    print("=" * 60)
    print("JOINT Aggregation (Phase 1 + 1.5) — Experiment A (DLA parity n=4):")
    print("=" * 60)
    print(f"{'depth':<6}{'leak_even':<12}{'leak_odd':<12}{'asym_rel':<12}{'n_reps':<8}")
    for row in agg_combined:
        print(
            f"{row['depth']:<6}{row['mean_leakage_even']:<12.4f}"
            f"{row['mean_leakage_odd']:<12.4f}"
            f"{row['asymmetry_relative']:<+12.4f}{row['n_reps']:<8}"
        )
    print()

    # Aggregate Experiment E (n=8)
    e_results = [r for r in all_results_raw if r["meta"]["experiment"] == "E_scaling_n8"]
    print("Experiment E (n=8 scaling):")
    print(f"{'depth':<6}{'sector':<8}{'leak':<12}{'rep':<6}")
    for r in e_results:
        m = r["meta"]
        s = r.get("stats", {})
        leak = s.get("parity_leakage", -1)
        print(f"{m['depth']:<6}{m['sector']:<8}{leak:<12.4f}{m['rep']:<6}")
    print()

    # Save full results
    output = {
        "experiment": EXPERIMENT_NAME,
        "timestamp_utc": timestamp,
        "backend": args.backend,
        "job_ids": [job_id],
        "wall_time_s": wall,
        "n_circuits": len(all_circuits),
        "t_step": T_STEP,
        "circuits": all_results_raw,
        "aggregated_joint_phase1_and_1_5": {"experiment_A_dla_parity_n4": agg_combined},
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved: {results_path}")

    # Append to log
    log_path = REPO_ROOT / ".coordination" / "IBM_EXECUTION_LOG.md"
    with open(log_path, "a") as f:
        f.write(f"\n## {timestamp} — PHASE 1.5 REINFORCEMENT\n\n")
        f.write(f"- **Experiment:** {EXPERIMENT_NAME}\n")
        f.write(f"- **Backend:** {args.backend}\n")
        f.write(
            f"- **Circuits:** {len(all_circuits)} (D: {len(circuits_d)}, E: {len(circuits_e)})\n"
        )
        f.write(f"- **Job ID:** `{job_id}`\n")
        f.write(f"- **Wall time:** {wall:.1f}s\n")
        f.write(f"- **Results file:** `{results_path.relative_to(REPO_ROOT)}`\n")
        f.write("- **Joint DLA parity (Phase 1 + 1.5):**\n")
        for row in agg_combined:
            f.write(
                f"  - depth {row['depth']:3d}: "
                f"leak_even={row['mean_leakage_even']:.4f}, "
                f"leak_odd={row['mean_leakage_odd']:.4f}, "
                f"asym_rel={row['asymmetry_relative']:+.4f} "
                f"(n={row['n_reps']} reps)\n"
            )
        f.write(
            "- **Outcome:** Phase 1 data reinforced with 4 extra reps; "
            "n=8 scaling data collected.\n"
        )
    print(f"Log appended: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
