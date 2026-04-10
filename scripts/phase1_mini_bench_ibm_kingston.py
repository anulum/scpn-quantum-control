#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Phase 1 Mini DLA Parity Bench
"""Phase 1 mini-bench: DLA parity asymmetry on ibm_kingston.

42 circuits across three sub-experiments:
  A. DLA parity at n=4 — 32 circuits (8 depths × 2 sectors × 2 reps)
  B. Scaling at n=6    —  6 circuits (3 depths × 2 sectors × 1 rep)
  C. Readout baseline  —  4 circuits (4 bitstring preps)

Budget: target ~90s QPU time, margin ~100s.

Submitted as a single SamplerV2 batch. Job ID is written to disk
immediately after submission (before result wait) so it can be recovered
via scripts/retrieve_ibm_job.py if the client crashes.

Usage:
    python scripts/phase1_mini_bench_ibm_kingston.py --dry-run
    python scripts/phase1_mini_bench_ibm_kingston.py
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

# ============================================================
# Circuit builders
# ============================================================

T_STEP = 0.3  # meaningful rotation, ~11.5°/gate
BACKEND_NAME = "ibm_kingston"
EXPERIMENT_NAME = "phase1_dla_parity_mini_bench"


def prep_bitstring(qc: QuantumCircuit, bitstring: str) -> None:
    """Prepare computational basis state |bitstring⟩.

    Convention: bitstring[0] = qubit 0 (little-endian).
    E.g. bitstring='0011' → apply X to qubits 0 and 1.
    """
    for q, bit in enumerate(bitstring):
        if bit == "1":
            qc.x(q)


def build_kuramoto_k_matrix(n: int) -> np.ndarray:
    """Standard SCPN exponential-decay coupling matrix."""
    k = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                k[i, j] = 0.45 * np.exp(-0.3 * abs(i - j))
    return k


def build_xy_trotter_circuit(
    n: int,
    initial_bitstring: str,
    n_trotter_steps: int,
    t_step: float = T_STEP,
) -> QuantumCircuit:
    """Build a Kuramoto-XY Trotter circuit with a computational-basis
    initial state.

    H_XY = Σ K_nm (X_n X_m + Y_n Y_m) + Σ ω_n Z_n
    """
    qc = QuantumCircuit(n, n)
    prep_bitstring(qc, initial_bitstring)

    if n_trotter_steps > 0:
        k_matrix = build_kuramoto_k_matrix(n)
        omega = np.linspace(0.8, 1.2, n)
        for _ in range(n_trotter_steps):
            # Single-qubit Z rotations
            for i in range(n):
                qc.rz(2.0 * omega[i] * t_step, i)
            # XX+YY nearest-neighbour
            for i in range(n - 1):
                j = i + 1
                theta = 2.0 * k_matrix[i, j] * t_step
                qc.rxx(theta, i, j)
                qc.ryy(theta, i, j)

    qc.measure(range(n), range(n))
    return qc


def build_readout_baseline_circuit(n: int, bitstring: str) -> QuantumCircuit:
    """State prep + measure. No evolution."""
    qc = QuantumCircuit(n, n)
    prep_bitstring(qc, bitstring)
    qc.measure(range(n), range(n))
    return qc


# ============================================================
# Experiment plan
# ============================================================


def build_experiment_a() -> list[tuple[dict, QuantumCircuit]]:
    """A: DLA parity n=4, depth sweep, 2 sectors, 2 reps."""
    circuits: list[tuple[dict, QuantumCircuit]] = []
    depths = [2, 4, 6, 8, 10, 14, 20, 30]
    sectors = {
        "even": "0011",  # popcount=2, P=+1
        "odd": "0001",  # popcount=1, P=-1
    }
    reps = 2
    for depth in depths:
        for sector_name, init in sectors.items():
            for rep in range(reps):
                meta = {
                    "experiment": "A_dla_parity_n4",
                    "n_qubits": 4,
                    "depth": depth,
                    "sector": sector_name,
                    "initial": init,
                    "rep": rep,
                    "shots": 2048,
                    "t_step": T_STEP,
                }
                qc = build_xy_trotter_circuit(4, init, depth, T_STEP)
                qc.name = f"A_n4_d{depth}_{sector_name}_r{rep}"
                circuits.append((meta, qc))
    return circuits


def build_experiment_b() -> list[tuple[dict, QuantumCircuit]]:
    """B: Scaling at n=6, 3 depths, 2 sectors, 1 rep."""
    circuits: list[tuple[dict, QuantumCircuit]] = []
    depths = [4, 8, 16]
    sectors = {
        "even": "000011",  # popcount=2, P=+1
        "odd": "000001",  # popcount=1, P=-1
    }
    for depth in depths:
        for sector_name, init in sectors.items():
            meta = {
                "experiment": "B_scaling_n6",
                "n_qubits": 6,
                "depth": depth,
                "sector": sector_name,
                "initial": init,
                "rep": 0,
                "shots": 2048,
                "t_step": T_STEP,
            }
            qc = build_xy_trotter_circuit(6, init, depth, T_STEP)
            qc.name = f"B_n6_d{depth}_{sector_name}"
            circuits.append((meta, qc))
    return circuits


def build_experiment_c() -> list[tuple[dict, QuantumCircuit]]:
    """C: Readout baseline at n=4."""
    circuits: list[tuple[dict, QuantumCircuit]] = []
    for init in ("0000", "1111", "0101", "1010"):
        meta = {
            "experiment": "C_readout_baseline",
            "n_qubits": 4,
            "depth": 0,
            "sector": "baseline",
            "initial": init,
            "rep": 0,
            "shots": 4096,
            "t_step": 0.0,
        }
        qc = build_readout_baseline_circuit(4, init)
        qc.name = f"C_readout_{init}"
        circuits.append((meta, qc))
    return circuits


# ============================================================
# Analysis
# ============================================================


def parity_of_bitstring(s: str) -> int:
    """0 = even parity (even popcount), 1 = odd parity."""
    return s.replace(" ", "").count("1") % 2


def analyse_counts(counts: dict, meta: dict) -> dict:
    total = sum(counts.values())
    if total == 0:
        return {"error": "empty counts"}

    init_parity = parity_of_bitstring(meta["initial"])
    same_parity = 0
    opposite_parity = 0
    # Qiskit returns bitstrings in MSB-first (rightmost = qubit 0) convention,
    # while our `initial` stores qubit-0-first. Reverse for retention lookup.
    init_qiskit = meta["initial"][::-1]
    initial_state_count = counts.get(init_qiskit, 0)

    for bits, c in counts.items():
        clean = bits.replace(" ", "")
        if parity_of_bitstring(clean) == init_parity:
            same_parity += c
        else:
            opposite_parity += c

    return {
        "total_shots": total,
        "initial_parity": init_parity,
        "same_parity_count": same_parity,
        "opposite_parity_count": opposite_parity,
        "parity_leakage": opposite_parity / total,
        "same_parity_fraction": same_parity / total,
        "initial_state_retention": initial_state_count / total,
    }


def aggregate_experiment_a(results: list[dict]) -> list[dict]:
    """Compute mean leakage per (depth, sector) across reps."""
    buckets: dict[tuple[int, str], list[float]] = {}
    for r in results:
        if r["meta"]["experiment"] != "A_dla_parity_n4":
            continue
        stats = r.get("stats", {})
        if "parity_leakage" not in stats:
            continue
        key = (r["meta"]["depth"], r["meta"]["sector"])
        buckets.setdefault(key, []).append(stats["parity_leakage"])

    agg = []
    depths = sorted({k[0] for k in buckets})
    for d in depths:
        even_vals = buckets.get((d, "even"), [])
        odd_vals = buckets.get((d, "odd"), [])
        if even_vals and odd_vals:
            mean_even = float(np.mean(even_vals))
            mean_odd = float(np.mean(odd_vals))
            asymmetry = (mean_even - mean_odd) / max(mean_odd, 1e-12)
            agg.append(
                {
                    "depth": d,
                    "mean_leakage_even": mean_even,
                    "mean_leakage_odd": mean_odd,
                    "std_leakage_even": float(np.std(even_vals)),
                    "std_leakage_odd": float(np.std(odd_vals)),
                    "asymmetry_relative": asymmetry,
                    "n_reps": len(even_vals),
                }
            )
    return agg


# ============================================================
# Credentials
# ============================================================


def parse_vault(vault_path: Path) -> tuple[str, str]:
    api_key = None
    instance = None
    with open(vault_path) as f:
        in_ibm = False
        for line in f:
            if line.strip().startswith("### IBM Quantum"):
                in_ibm = True
                continue
            if in_ibm:
                if line.startswith("###"):
                    break
                if "API Key" in line:
                    if "`" in line:
                        api_key = line.split("`")[1]
                    else:
                        parts = line.split(":**")
                        if len(parts) >= 2:
                            api_key = parts[1].strip()
                elif ("CRN" in line or "Instance" in line) and "`" in line:
                    instance = line.split("`")[1]
    if not api_key or not instance:
        raise RuntimeError("Failed to parse vault")
    return api_key, instance


# ============================================================
# Main
# ============================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 DLA parity mini-bench")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backend", default=BACKEND_NAME)
    parser.add_argument("--shots-a", type=int, default=2048, help="Shots for exp A/B")
    parser.add_argument("--shots-c", type=int, default=4096, help="Shots for exp C")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    results_dir = REPO_ROOT / ".coordination" / "ibm_runs"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"phase1_bench_{timestamp}.json"

    # Build all circuits
    circuits_a = build_experiment_a()
    circuits_b = build_experiment_b()
    circuits_c = build_experiment_c()
    all_circuits = circuits_a + circuits_b + circuits_c

    print("=" * 60)
    print("Phase 1 DLA Parity Mini-Bench")
    print("=" * 60)
    print(f"Timestamp: {timestamp}")
    print(f"Backend:   {args.backend}")
    print(f"Dry-run:   {args.dry_run}")
    print()
    print(f"Experiment A (n=4 DLA parity):  {len(circuits_a)} circuits")
    print(f"Experiment B (n=6 scaling):     {len(circuits_b)} circuits")
    print(f"Experiment C (readout baseline):{len(circuits_c)} circuits")
    print(f"TOTAL:                          {len(all_circuits)} circuits")
    print()

    # Group by shots (SamplerV2 uses single shots value per run)
    # Exp A+B: shots_a
    # Exp C:   shots_c
    circuits_samp_a = [c for m, c in circuits_a + circuits_b]
    metas_samp_a = [m for m, c in circuits_a + circuits_b]
    circuits_samp_c = [c for m, c in circuits_c]
    metas_samp_c = [m for m, c in circuits_c]

    if args.dry_run:
        print("DRY RUN — transpile locally, run on simulator")
        from scpn_quantum_control.hardware.runner import HardwareRunner

        runner = HardwareRunner(
            use_simulator=True, results_dir=str(REPO_ROOT / "results/ibm_runs")
        )
        runner.connect()
        # Transpile all circuits
        all_isa = [runner.transpile(qc) for _, qc in all_circuits]
        depths_isa = [c.depth() for c in all_isa]
        print(
            f"Transpile OK. ISA depths: min={min(depths_isa)}, "
            f"max={max(depths_isa)}, mean={np.mean(depths_isa):.1f}"
        )
        # Quick sim run on first 4 circuits
        sample = all_circuits[:4]
        results = runner.run_sampler([c for _, c in sample], shots=1024, name="dry_run_sim")
        for (meta, _), res in zip(sample, results):
            stats = analyse_counts(res.counts or {}, meta)
            print(f"  {meta['experiment']} {meta['sector']} d={meta['depth']}: {stats}")
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
        resilience_level=0,  # raw counts, no runtime mitigation
        results_dir=str(REPO_ROOT / "results/ibm_runs"),
    )
    runner.connect()
    print(f"Connected: {runner.backend_name}")
    print()

    # Pre-save metadata so retrieval can reconstruct even if parsing crashes
    pre_save = {
        "experiment": EXPERIMENT_NAME,
        "timestamp_utc": timestamp,
        "backend": args.backend,
        "metas_a_b": metas_samp_a,
        "metas_c": metas_samp_c,
    }
    with open(results_path, "w") as f:
        json.dump({"status": "submitted", "pre_save": pre_save}, f, indent=2, default=str)
    print(f"Pre-save metadata: {results_path}")
    print()

    all_results_raw: list[dict] = []
    job_ids: list[str] = []

    # Batch 1: Experiments A + B (shots_a)
    print(
        f"Submitting Batch 1 (exp A+B, {len(circuits_samp_a)} circuits, shots={args.shots_a})..."
    )
    t0 = time.time()
    try:
        batch_a = runner.run_sampler(
            circuits_samp_a,
            shots=args.shots_a,
            name=f"{EXPERIMENT_NAME}_AB",
            timeout_s=1200,
        )
    except Exception as e:
        print(f"ERROR: Batch 1 submission failed: {e}", file=sys.stderr)
        # Save partial state
        with open(results_path, "w") as f:
            json.dump(
                {"status": "batch1_failed", "error": str(e), "pre_save": pre_save},
                f,
                indent=2,
                default=str,
            )
        return 1
    wall_1 = time.time() - t0
    if batch_a:
        job_ids.append(batch_a[0].job_id)
    print(f"Batch 1 done in {wall_1:.1f}s. Job: {batch_a[0].job_id if batch_a else 'N/A'}")

    for meta, jr in zip(metas_samp_a, batch_a):
        stats = analyse_counts(jr.counts or {}, meta)
        all_results_raw.append(
            {"meta": meta, "counts": jr.counts, "stats": stats, "job_id": jr.job_id}
        )

    # Batch 2: Experiment C (shots_c)
    print(
        f"\nSubmitting Batch 2 (exp C, {len(circuits_samp_c)} circuits, shots={args.shots_c})..."
    )
    t0 = time.time()
    try:
        batch_c = runner.run_sampler(
            circuits_samp_c,
            shots=args.shots_c,
            name=f"{EXPERIMENT_NAME}_C",
            timeout_s=600,
        )
    except Exception as e:
        print(f"ERROR: Batch 2 submission failed: {e}", file=sys.stderr)
        # Save partial with batch 1 results
        output_partial = {
            "experiment": EXPERIMENT_NAME,
            "timestamp_utc": timestamp,
            "backend": args.backend,
            "job_ids": job_ids,
            "status": "batch2_failed",
            "error": str(e),
            "circuits": all_results_raw,
        }
        with open(results_path, "w") as f:
            json.dump(output_partial, f, indent=2, default=str)
        return 2
    wall_2 = time.time() - t0
    if batch_c:
        job_ids.append(batch_c[0].job_id)
    print(f"Batch 2 done in {wall_2:.1f}s. Job: {batch_c[0].job_id if batch_c else 'N/A'}")

    for meta, jr in zip(metas_samp_c, batch_c):
        stats = analyse_counts(jr.counts or {}, meta)
        all_results_raw.append(
            {"meta": meta, "counts": jr.counts, "stats": stats, "job_id": jr.job_id}
        )

    total_wall = wall_1 + wall_2
    print(f"\nTotal wall time (incl. queue): {total_wall:.1f}s")
    print()

    # Aggregate Experiment A
    agg_a = aggregate_experiment_a(all_results_raw)
    print("=" * 60)
    print("Experiment A aggregated (DLA parity n=4):")
    print("=" * 60)
    print(f"{'depth':<6}{'leak_even':<12}{'leak_odd':<12}{'asym_rel':<12}")
    for row in agg_a:
        print(
            f"{row['depth']:<6}{row['mean_leakage_even']:<12.4f}"
            f"{row['mean_leakage_odd']:<12.4f}{row['asymmetry_relative']:<12.4f}"
        )
    print()

    # Save full results
    output = {
        "experiment": EXPERIMENT_NAME,
        "timestamp_utc": timestamp,
        "backend": args.backend,
        "job_ids": job_ids,
        "wall_time_s": total_wall,
        "n_circuits": len(all_circuits),
        "t_step": T_STEP,
        "circuits": all_results_raw,
        "aggregated": {"experiment_A_dla_parity_n4": agg_a},
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Full results saved: {results_path}")

    # Append to execution log
    log_path = REPO_ROOT / ".coordination" / "IBM_EXECUTION_LOG.md"
    with open(log_path, "a") as f:
        f.write(f"\n## {timestamp} — PHASE 1 MINI-BENCH\n\n")
        f.write(f"- **Experiment:** {EXPERIMENT_NAME}\n")
        f.write(f"- **Backend:** {args.backend}\n")
        f.write(
            f"- **Circuits:** {len(all_circuits)} "
            f"(A: {len(circuits_a)}, B: {len(circuits_b)}, C: {len(circuits_c)})\n"
        )
        f.write(f"- **Job IDs:** {', '.join(f'`{j}`' for j in job_ids)}\n")
        f.write(f"- **Wall time:** {total_wall:.1f}s\n")
        f.write(f"- **Results file:** `{results_path.relative_to(REPO_ROOT)}`\n")
        if agg_a:
            f.write("- **DLA parity summary (exp A):**\n")
            for row in agg_a:
                f.write(
                    f"  - depth {row['depth']:3d}: "
                    f"leak_even={row['mean_leakage_even']:.4f}, "
                    f"leak_odd={row['mean_leakage_odd']:.4f}, "
                    f"asym_rel={row['asymmetry_relative']:+.4f}\n"
                )
        f.write("- **Outcome:** Phase 1 primary DLA parity data on Heron r2.\n")
    print(f"Log appended: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
