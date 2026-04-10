#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Phase 2 Full Campaign (180-min promo)
"""Phase 2 — full DLA parity campaign on 180-minute promotional allocation.

DO NOT RUN until IBM confirms the 180-min promo is active on the account.
This script targets ~150 minutes of QPU time (50-minute safety margin).

Scientific goals (in priority order):
  1. PRIMARY: statistically robust DLA parity asymmetry at n = 4, 6, 8, 10
     with GUESS error mitigation applied — target p < 0.001.
  2. SECONDARY: n = 12, 14, 16 scaling law — first measurement of
     asymmetry vs system size.
  3. TERTIARY: independent replication on a different Heron r2 device
     (e.g. ibm_marrakesh) to rule out device-specific artefacts.

Sub-experiments:

  Exp A — High-statistics n = 4 (reference point)
    depths: [2, 4, 6, 8, 10, 14, 20, 30, 40, 50]
    sectors: even/odd parity
    reps: 30 (vs 21 in Phase 1 for the reinforced depths)
    shots: 4096 (vs 2048 in Phase 1)
    circuits: 10 * 2 * 30 = 600
    est: ~25 min

  Exp B — Scaling at n = 6
    depths: [4, 8, 14, 20]
    sectors: even/odd
    reps: 20
    shots: 4096
    circuits: 4 * 2 * 20 = 160
    est: ~15 min

  Exp C — Scaling at n = 8
    depths: [4, 8, 14, 20]
    sectors: even/odd
    reps: 15
    shots: 4096
    circuits: 4 * 2 * 15 = 120
    est: ~15 min

  Exp D — Scaling at n = 10
    depths: [4, 8, 14]
    sectors: even/odd
    reps: 12
    shots: 4096
    circuits: 3 * 2 * 12 = 72
    est: ~12 min

  Exp E — Scaling at n = 12
    depths: [4, 8, 14]
    sectors: even/odd
    reps: 8
    shots: 4096
    circuits: 3 * 2 * 8 = 48
    est: ~12 min

  Exp F — GUESS calibration — noise scale sweep at n = 4
    depths: [4, 8, 14]
    sectors: even/odd
    noise_scales (via circuit folding): [1, 3, 5]
    reps: 10
    shots: 4096
    circuits: 3 * 2 * 3 * 10 = 180
    est: ~25 min

  Exp G — Readout baseline (expanded)
    n = 4, 6, 8 computational basis states (16 total distinct preps)
    shots: 8192
    circuits: 16
    est: ~3 min

Total: ~1,196 circuits, ~107 minutes QPU time. Leaves ~70 min margin.

Usage:
    # Always dry-run first to verify transpile budget
    python scripts/phase2_full_campaign_ibm.py --dry-run

    # Submit to IBM (ONLY after 180-min promo confirmed active)
    python scripts/phase2_full_campaign_ibm.py \\
        --confirm-promo-active \\
        --backend ibm_kingston

    # Run on a different device for independent replication
    python scripts/phase2_full_campaign_ibm.py \\
        --confirm-promo-active \\
        --backend ibm_marrakesh
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit

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

BACKEND_NAME_DEFAULT = "ibm_kingston"
EXPERIMENT_NAME = "phase2_full_campaign"
SHOTS_MAIN = 4096
SHOTS_BASELINE = 8192


def fold_circuit(qc: QuantumCircuit, scale: int) -> QuantumCircuit:
    """Naive circuit folding: U → U U† U ... for noise amplification.

    Folds the core (non-measurement) gates scale-1 times. Preserves the
    ideal unitary action but multiplies the effective noise.
    """
    if scale < 1 or scale % 2 == 0:
        raise ValueError(f"scale must be odd integer >= 1, got {scale}")
    if scale == 1:
        return qc

    n = qc.num_qubits
    folded = QuantumCircuit(n, n)

    # Extract non-measurement gates
    core_ops = []
    meas_ops = []
    for inst in qc.data:
        if inst.operation.name == "measure":
            meas_ops.append(inst)
        else:
            core_ops.append(inst)

    # Initial state prep (all X's and core ops)
    for inst in core_ops:
        folded.append(inst.operation, inst.qubits, inst.clbits)

    # Fold (scale - 1) / 2 times: add U† U pairs
    n_folds = (scale - 1) // 2
    for _ in range(n_folds):
        for inst in reversed(core_ops):
            folded.append(inst.operation.inverse(), inst.qubits, inst.clbits)
        for inst in core_ops:
            folded.append(inst.operation, inst.qubits, inst.clbits)

    # Original measurements
    for inst in meas_ops:
        folded.append(inst.operation, inst.qubits, inst.clbits)

    return folded


def even_parity_init(n: int) -> str:
    """Smallest even-parity state with nontrivial M-block dynamics."""
    # '0011' for n=4, '000011' for n=6, etc. — popcount=2, even parity
    return "0" * (n - 2) + "11"


def odd_parity_init(n: int) -> str:
    """Smallest odd-parity state with nontrivial M-block dynamics."""
    # '0001' for n=4, etc. — popcount=1, odd parity
    return "0" * (n - 1) + "1"


def build_experiment_a() -> list[tuple[dict, QuantumCircuit]]:
    """A: High-statistics n = 4 DLA parity."""
    circuits: list[tuple[dict, QuantumCircuit]] = []
    n = 4
    depths = [2, 4, 6, 8, 10, 14, 20, 30, 40, 50]
    sectors = {"even": even_parity_init(n), "odd": odd_parity_init(n)}
    reps = range(30)
    for depth in depths:
        for sector, init in sectors.items():
            for rep in reps:
                meta = {
                    "experiment": "A_dla_parity_n4_phase2",
                    "n_qubits": n,
                    "depth": depth,
                    "sector": sector,
                    "initial": init,
                    "rep": rep,
                    "shots": SHOTS_MAIN,
                    "t_step": T_STEP,
                    "phase": "2_full",
                }
                qc = build_xy_trotter_circuit(n, init, depth, T_STEP)
                qc.name = f"A_n{n}_d{depth}_{sector}_r{rep}"
                circuits.append((meta, qc))
    return circuits


def build_scaling_experiment(
    label: str, n: int, depths: list[int], reps: int
) -> list[tuple[dict, QuantumCircuit]]:
    """Generic scaling experiment: n qubits, depth sweep, 2 sectors, reps."""
    circuits: list[tuple[dict, QuantumCircuit]] = []
    sectors = {"even": even_parity_init(n), "odd": odd_parity_init(n)}
    for depth in depths:
        for sector, init in sectors.items():
            for rep in range(reps):
                meta = {
                    "experiment": label,
                    "n_qubits": n,
                    "depth": depth,
                    "sector": sector,
                    "initial": init,
                    "rep": rep,
                    "shots": SHOTS_MAIN,
                    "t_step": T_STEP,
                    "phase": "2_full",
                }
                qc = build_xy_trotter_circuit(n, init, depth, T_STEP)
                qc.name = f"{label}_n{n}_d{depth}_{sector}_r{rep}"
                circuits.append((meta, qc))
    return circuits


def build_experiment_f() -> list[tuple[dict, QuantumCircuit]]:
    """F: GUESS noise-scale sweep at n = 4."""
    circuits: list[tuple[dict, QuantumCircuit]] = []
    n = 4
    depths = [4, 8, 14]
    sectors = {"even": even_parity_init(n), "odd": odd_parity_init(n)}
    noise_scales = [1, 3, 5]
    reps = 10
    for depth in depths:
        for sector, init in sectors.items():
            for scale in noise_scales:
                for rep in range(reps):
                    base = build_xy_trotter_circuit(n, init, depth, T_STEP)
                    try:
                        folded = fold_circuit(base, scale)
                    except ValueError:
                        folded = base
                    meta = {
                        "experiment": "F_guess_calibration_n4",
                        "n_qubits": n,
                        "depth": depth,
                        "sector": sector,
                        "initial": init,
                        "noise_scale": scale,
                        "rep": rep,
                        "shots": SHOTS_MAIN,
                        "t_step": T_STEP,
                        "phase": "2_full",
                    }
                    folded.name = f"F_n{n}_d{depth}_{sector}_g{scale}_r{rep}"
                    circuits.append((meta, folded))
    return circuits


def build_experiment_g() -> list[tuple[dict, QuantumCircuit]]:
    """G: Readout baseline across n = 4, 6, 8."""
    circuits: list[tuple[dict, QuantumCircuit]] = []
    preps = {
        4: ["0000", "1111", "0101", "1010"],
        6: ["000000", "111111", "010101", "101010"],
        8: ["00000000", "11111111", "01010101", "10101010"],
    }
    for n, states in preps.items():
        for init in states:
            qc = QuantumCircuit(n, n)
            for q, bit in enumerate(init):
                if bit == "1":
                    qc.x(q)
            qc.measure(range(n), range(n))
            qc.name = f"G_readout_n{n}_{init}"
            meta = {
                "experiment": "G_readout_baseline",
                "n_qubits": n,
                "depth": 0,
                "sector": "baseline",
                "initial": init,
                "rep": 0,
                "shots": SHOTS_BASELINE,
                "t_step": 0.0,
                "phase": "2_full",
            }
            circuits.append((meta, qc))
    return circuits


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 full campaign")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--confirm-promo-active",
        action="store_true",
        help="Required for real submission. Acknowledges that the 180-min "
        "promo is active and the user accepts the burn.",
    )
    parser.add_argument("--backend", default=BACKEND_NAME_DEFAULT)
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["A", "B", "C", "D", "E", "F", "G"],
        help="Skip specific experiments",
    )
    args = parser.parse_args()

    if not args.dry_run and not args.confirm_promo_active:
        print("ERROR: Real submission requires --confirm-promo-active flag.", file=sys.stderr)
        print("Run with --dry-run first to verify transpile budget.", file=sys.stderr)
        return 1

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    results_dir = REPO_ROOT / ".coordination" / "ibm_runs"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"phase2_full_{timestamp}.json"

    # Build all experiments
    blocks = []
    if "A" not in args.skip:
        blocks.append(("A", build_experiment_a()))
    if "B" not in args.skip:
        blocks.append(
            ("B", build_scaling_experiment("B_scaling_n6_phase2", 6, [4, 8, 14, 20], 20))
        )
    if "C" not in args.skip:
        blocks.append(
            ("C", build_scaling_experiment("C_scaling_n8_phase2", 8, [4, 8, 14, 20], 15))
        )
    if "D" not in args.skip:
        blocks.append(("D", build_scaling_experiment("D_scaling_n10_phase2", 10, [4, 8, 14], 12)))
    if "E" not in args.skip:
        blocks.append(("E", build_scaling_experiment("E_scaling_n12_phase2", 12, [4, 8, 14], 8)))
    if "F" not in args.skip:
        blocks.append(("F", build_experiment_f()))
    if "G" not in args.skip:
        blocks.append(("G", build_experiment_g()))

    all_circuits: list[tuple[dict, QuantumCircuit]] = []
    print("=" * 70)
    print("Phase 2 — Full DLA Parity Campaign (180-min promo)")
    print("=" * 70)
    print(f"Timestamp: {timestamp}")
    print(f"Backend:   {args.backend}")
    print(f"Dry-run:   {args.dry_run}")
    print()
    for label, block in blocks:
        print(f"Experiment {label}: {len(block)} circuits")
        all_circuits.extend(block)
    print(f"TOTAL: {len(all_circuits)} circuits")
    print(f"Estimated QPU cost at 0.55 s/circuit: ~{len(all_circuits) * 0.55 / 60:.1f} minutes")
    print()

    circuits_main = [c for m, c in all_circuits if m["shots"] == SHOTS_MAIN]
    metas_main = [m for m, c in all_circuits if m["shots"] == SHOTS_MAIN]
    circuits_baseline = [c for m, c in all_circuits if m["shots"] == SHOTS_BASELINE]
    metas_baseline = [m for m, c in all_circuits if m["shots"] == SHOTS_BASELINE]

    if args.dry_run:
        print("DRY RUN — transpile via HardwareRunner simulator")
        from scpn_quantum_control.hardware.runner import HardwareRunner

        runner = HardwareRunner(
            use_simulator=True, results_dir=str(REPO_ROOT / "results/ibm_runs")
        )
        runner.connect()
        all_isa = [runner.transpile(qc) for _, qc in all_circuits]
        depths_isa = [c.depth() for c in all_isa]
        gate_counts = [sum(c.count_ops().values()) for c in all_isa]
        print(
            f"Transpile OK. ISA depths: min={min(depths_isa)}, "
            f"max={max(depths_isa)}, mean={np.mean(depths_isa):.1f}"
        )
        print(
            f"Gate counts: min={min(gate_counts)}, "
            f"max={max(gate_counts)}, mean={np.mean(gate_counts):.1f}"
        )
        # Summary per experiment
        from collections import defaultdict

        by_exp: dict[str, int] = defaultdict(int)
        for meta, _ in all_circuits:
            by_exp[meta["experiment"]] += 1
        print("\nCircuit count per experiment:")
        for exp, count in sorted(by_exp.items()):
            print(f"  {exp}: {count}")
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
            {
                "status": "submitted",
                "timestamp": timestamp,
                "backend": args.backend,
                "n_circuits_main": len(circuits_main),
                "n_circuits_baseline": len(circuits_baseline),
                "metas_main": metas_main,
                "metas_baseline": metas_baseline,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"Pre-save: {results_path}")
    print()

    job_ids: list[str] = []
    all_results: list[dict] = []

    # Batch 1: main experiments
    print(f"Submitting Batch 1 (main, {len(circuits_main)} circuits, shots={SHOTS_MAIN})...")
    t0 = time.time()
    try:
        batch1 = runner.run_sampler(
            circuits_main,
            shots=SHOTS_MAIN,
            name=f"{EXPERIMENT_NAME}_main",
            timeout_s=7200,
        )
    except Exception as e:
        print(f"ERROR: Batch 1 failed: {e}", file=sys.stderr)
        return 2
    wall_1 = time.time() - t0
    if batch1:
        job_ids.append(batch1[0].job_id)
    print(f"Batch 1 done in {wall_1:.1f}s ({wall_1 / 60:.1f} min). Job: {job_ids[-1]}")
    for meta, jr in zip(metas_main, batch1):
        stats = analyse_counts(jr.counts or {}, meta)
        all_results.append(
            {"meta": meta, "counts": jr.counts, "stats": stats, "job_id": jr.job_id}
        )

    # Batch 2: readout baseline
    if circuits_baseline:
        print(
            f"\nSubmitting Batch 2 (readout, {len(circuits_baseline)} circuits, shots={SHOTS_BASELINE})..."
        )
        t0 = time.time()
        try:
            batch2 = runner.run_sampler(
                circuits_baseline,
                shots=SHOTS_BASELINE,
                name=f"{EXPERIMENT_NAME}_readout",
                timeout_s=1800,
            )
        except Exception as e:
            print(f"ERROR: Batch 2 failed: {e}", file=sys.stderr)
            # Save partial
            with open(results_path, "w") as f:
                json.dump(
                    {
                        "status": "partial_batch2_failed",
                        "error": str(e),
                        "job_ids": job_ids,
                        "circuits": all_results,
                    },
                    f,
                    indent=2,
                    default=str,
                )
            return 3
        wall_2 = time.time() - t0
        if batch2:
            job_ids.append(batch2[0].job_id)
        print(f"Batch 2 done in {wall_2:.1f}s. Job: {job_ids[-1]}")
        for meta, jr in zip(metas_baseline, batch2):
            stats = analyse_counts(jr.counts or {}, meta)
            all_results.append(
                {"meta": meta, "counts": jr.counts, "stats": stats, "job_id": jr.job_id}
            )

    # Aggregate Experiment A (n=4)
    agg_a = aggregate_experiment_a(all_results)
    print()
    print("=" * 70)
    print("Experiment A (n=4) — aggregated across 30 reps per point")
    print("=" * 70)
    print(f"{'depth':<6}{'leak_even':<14}{'leak_odd':<14}{'asym_rel':<14}{'n_reps':<8}")
    for row in agg_a:
        print(
            f"{row['depth']:<6}{row['mean_leakage_even']:<14.4f}"
            f"{row['mean_leakage_odd']:<14.4f}"
            f"{row['asymmetry_relative']:<+14.4f}{row['n_reps']:<8}"
        )

    # Save full results
    output = {
        "experiment": EXPERIMENT_NAME,
        "timestamp_utc": timestamp,
        "backend": args.backend,
        "job_ids": job_ids,
        "n_circuits": len(all_circuits),
        "t_step": T_STEP,
        "shots_main": SHOTS_MAIN,
        "shots_baseline": SHOTS_BASELINE,
        "circuits": all_results,
        "aggregated_experiment_A": agg_a,
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")

    # Append to log
    log_path = REPO_ROOT / ".coordination" / "IBM_EXECUTION_LOG.md"
    with open(log_path, "a") as f:
        f.write(f"\n## {timestamp} — PHASE 2 FULL CAMPAIGN\n\n")
        f.write(f"- **Experiment:** {EXPERIMENT_NAME}\n")
        f.write(f"- **Backend:** {args.backend}\n")
        f.write(f"- **Circuits:** {len(all_circuits)}\n")
        f.write(f"- **Job IDs:** {', '.join(f'`{j}`' for j in job_ids)}\n")
        f.write(f"- **Results file:** `{results_path.relative_to(REPO_ROOT)}`\n")
        if agg_a:
            f.write("- **Experiment A DLA parity (n=4, 30 reps):**\n")
            for row in agg_a:
                f.write(
                    f"  - depth {row['depth']:3d}: "
                    f"leak_even={row['mean_leakage_even']:.4f}, "
                    f"leak_odd={row['mean_leakage_odd']:.4f}, "
                    f"asym_rel={row['asymmetry_relative']:+.4f}\n"
                )
        f.write("- **Purpose:** Phase 2 full campaign on 180-min promo.\n")
    print(f"Log appended: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
