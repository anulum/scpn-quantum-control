#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Phase 2 cycle exhaust
"""Phase 2: exhaust current Open Plan cycle to trigger 180-min promo.

Target: ~120 circuits at estimated ~85s QPU time. Actual remaining is
~2 minutes (~120s) based on Phase 1+1.5 estimates. We design for slight
under-exhaust (so the batch completes) and expect the cycle quota to
cut off if we overshoot.

Experiments:
  F. n=4 DLA parity reinforcement — 96 circuits (12 reps per point total)
  G. n=6 reinforcement            — 18 circuits (4 reps per point total)
  H. n=8 reinforcement            —  8 circuits (4 reps per point total)
  I. n=4 extended depths 40, 50   — 16 circuits (new data beyond Phase 1)

Total: 138 circuits, est ~95-100s.

This is the final action on the current cycle. After this runs, check
the IBM dashboard — if the cycle is exhausted (usage ≥ 10m), the 180-min
promo should be available for opt-in.
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
EXPERIMENT_NAME = "phase2_exhaust_cycle"


def build_experiment_f() -> list[tuple[dict, object]]:
    """F: n=4 DLA parity reinforcement, rep 6..11."""
    circuits: list[tuple[dict, object]] = []
    depths = [2, 4, 6, 8, 10, 14, 20, 30]
    sectors = {"even": "0011", "odd": "0001"}
    reps = list(range(6, 12))  # 6 new reps continuing from Phase 1.5
    for depth in depths:
        for sector_name, init in sectors.items():
            for rep in reps:
                meta = {
                    "experiment": "A_dla_parity_n4",
                    "n_qubits": 4,
                    "depth": depth,
                    "sector": sector_name,
                    "initial": init,
                    "rep": rep,
                    "shots": 2048,
                    "t_step": T_STEP,
                    "phase": "2",
                }
                qc = build_xy_trotter_circuit(4, init, depth, T_STEP)
                qc.name = f"F_n4_d{depth}_{sector_name}_r{rep}"
                circuits.append((meta, qc))
    return circuits


def build_experiment_g() -> list[tuple[dict, object]]:
    """G: n=6 reinforcement, rep 1..3."""
    circuits: list[tuple[dict, object]] = []
    depths = [4, 8, 16]
    sectors = {"even": "000011", "odd": "000001"}
    reps = list(range(1, 4))
    for depth in depths:
        for sector_name, init in sectors.items():
            for rep in reps:
                meta = {
                    "experiment": "B_scaling_n6",
                    "n_qubits": 6,
                    "depth": depth,
                    "sector": sector_name,
                    "initial": init,
                    "rep": rep,
                    "shots": 2048,
                    "t_step": T_STEP,
                    "phase": "2",
                }
                qc = build_xy_trotter_circuit(6, init, depth, T_STEP)
                qc.name = f"G_n6_d{depth}_{sector_name}_r{rep}"
                circuits.append((meta, qc))
    return circuits


def build_experiment_h() -> list[tuple[dict, object]]:
    """H: n=8 reinforcement, rep 2..3."""
    circuits: list[tuple[dict, object]] = []
    depths = [4, 8]
    sectors = {"even": "00000011", "odd": "00000001"}
    reps = [2, 3]
    for depth in depths:
        for sector_name, init in sectors.items():
            for rep in reps:
                meta = {
                    "experiment": "E_scaling_n8",
                    "n_qubits": 8,
                    "depth": depth,
                    "sector": sector_name,
                    "initial": init,
                    "rep": rep,
                    "shots": 2048,
                    "t_step": T_STEP,
                    "phase": "2",
                }
                qc = build_xy_trotter_circuit(8, init, depth, T_STEP)
                qc.name = f"H_n8_d{depth}_{sector_name}_r{rep}"
                circuits.append((meta, qc))
    return circuits


def build_experiment_i() -> list[tuple[dict, object]]:
    """I: n=4 extended deep depths 40, 50."""
    circuits: list[tuple[dict, object]] = []
    depths = [40, 50]
    sectors = {"even": "0011", "odd": "0001"}
    reps = list(range(4))
    for depth in depths:
        for sector_name, init in sectors.items():
            for rep in reps:
                meta = {
                    "experiment": "I_n4_extended_depth",
                    "n_qubits": 4,
                    "depth": depth,
                    "sector": sector_name,
                    "initial": init,
                    "rep": rep,
                    "shots": 2048,
                    "t_step": T_STEP,
                    "phase": "2",
                }
                qc = build_xy_trotter_circuit(4, init, depth, T_STEP)
                qc.name = f"I_n4_d{depth}_{sector_name}_r{rep}"
                circuits.append((meta, qc))
    return circuits


def load_prior_results(paths: list[Path]) -> list[dict]:
    """Load all prior phase results for joint aggregation."""
    combined: list[dict] = []
    for p in paths:
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)
        combined.extend(data.get("circuits", []))
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 cycle exhaust")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backend", default=BACKEND_NAME)
    parser.add_argument("--shots", type=int, default=2048)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    results_dir = REPO_ROOT / ".coordination" / "ibm_runs"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"phase2_exhaust_{timestamp}.json"

    circuits_f = build_experiment_f()
    circuits_g = build_experiment_g()
    circuits_h = build_experiment_h()
    circuits_i = build_experiment_i()
    all_circuits = circuits_f + circuits_g + circuits_h + circuits_i

    print("=" * 60)
    print("Phase 2 — Cycle Exhaust + Final Reinforcement")
    print("=" * 60)
    print(f"Timestamp: {timestamp}")
    print(f"Backend:   {args.backend}")
    print(f"Dry-run:   {args.dry_run}")
    print()
    print(f"Experiment F (n=4 DLA rep 6-11):  {len(circuits_f)} circuits")
    print(f"Experiment G (n=6 rep 1-3):       {len(circuits_g)} circuits")
    print(f"Experiment H (n=8 rep 2-3):       {len(circuits_h)} circuits")
    print(f"Experiment I (n=4 depth 40, 50):  {len(circuits_i)} circuits")
    print(f"TOTAL:                            {len(all_circuits)} circuits")
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

    # Pre-save metadata
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
    print("This should exhaust the current 10-minute cycle.")
    print("Expected duration: 60-90s wall time. Do not interrupt.")
    print()

    t0 = time.time()
    try:
        batch = runner.run_sampler(
            circuits_only,
            shots=args.shots,
            name=EXPERIMENT_NAME,
            timeout_s=1800,  # 30 min wall time headroom
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

    # Joint aggregation across ALL phases
    prior_files = [
        REPO_ROOT / ".coordination/ibm_runs/phase1_bench_2026-04-10T183728Z.json",
        REPO_ROOT / ".coordination/ibm_runs/phase1_5_reinforce_2026-04-10T184909Z.json",
    ]
    prior_circuits = load_prior_results(prior_files)
    combined = prior_circuits + all_results_raw
    print(
        f"Joint analysis: prior {len(prior_circuits)} + "
        f"Phase 2 {len(all_results_raw)} = {len(combined)} circuits"
    )

    agg_combined = aggregate_experiment_a(combined)
    print()
    print("=" * 70)
    print("JOINT Aggregation (Phase 1 + 1.5 + 2) — Experiment A (n=4 DLA parity)")
    print("=" * 70)
    print(f"{'depth':<6}{'leak_even':<14}{'leak_odd':<14}{'asym_rel':<14}{'n_reps':<8}")
    for row in agg_combined:
        print(
            f"{row['depth']:<6}{row['mean_leakage_even']:<14.4f}"
            f"{row['mean_leakage_odd']:<14.4f}"
            f"{row['asymmetry_relative']:<+14.4f}{row['n_reps']:<8}"
        )
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
        "aggregated_joint_all_phases": {"experiment_A_dla_parity_n4": agg_combined},
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved: {results_path}")

    # Append to log
    log_path = REPO_ROOT / ".coordination" / "IBM_EXECUTION_LOG.md"
    with open(log_path, "a") as f:
        f.write(f"\n## {timestamp} — PHASE 2 CYCLE EXHAUST\n\n")
        f.write(f"- **Experiment:** {EXPERIMENT_NAME}\n")
        f.write(f"- **Backend:** {args.backend}\n")
        f.write(
            f"- **Circuits:** {len(all_circuits)} "
            f"(F: {len(circuits_f)}, G: {len(circuits_g)}, "
            f"H: {len(circuits_h)}, I: {len(circuits_i)})\n"
        )
        f.write(f"- **Job ID:** `{job_id}`\n")
        f.write(f"- **Wall time:** {wall:.1f}s\n")
        f.write(f"- **Results file:** `{results_path.relative_to(REPO_ROOT)}`\n")
        f.write("- **Joint DLA parity (all phases):**\n")
        for row in agg_combined:
            f.write(
                f"  - depth {row['depth']:3d}: "
                f"leak_even={row['mean_leakage_even']:.4f}, "
                f"leak_odd={row['mean_leakage_odd']:.4f}, "
                f"asym_rel={row['asymmetry_relative']:+.4f} "
                f"(n={row['n_reps']} reps)\n"
            )
        f.write(
            "- **Purpose:** Exhaust current Open Plan cycle to trigger 180-minute promo unlock.\n"
        )
        f.write(
            "- **Next action:** Check IBM dashboard. If usage ≥ 10m, opt in to 180-min promo.\n"
        )
    print(f"Log appended: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
