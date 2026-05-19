#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Phase 2 popcount-control IBM runner
"""Run the preregistered Phase 2 popcount-control QPU experiment."""

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
    analyse_counts,
    build_xy_trotter_circuit,
    parse_vault,
)

EXPERIMENT_NAME = "phase2_popcount_control"
BACKEND_NAME_DEFAULT = "ibm_kingston"
SHOTS_MAIN = 4096
SHOTS_READOUT = 8192
DEPTHS = (4, 6, 8, 10, 14, 20)
REPS = 12
STATES = (
    ("E0_original_even", "0011", "even", 2),
    ("E1_even_swap", "0101", "even", 2),
    ("O0_original_odd", "0001", "odd", 1),
    ("O1_odd_swap", "0010", "odd", 1),
    ("O3_odd_high_excitation", "0111", "odd", 3),
)


def build_readout_circuit(bitstring: str) -> QuantumCircuit:
    """State preparation plus measurement only."""
    qc = QuantumCircuit(len(bitstring), len(bitstring))
    for q, bit in enumerate(bitstring):
        if bit == "1":
            qc.x(q)
    qc.measure(range(len(bitstring)), range(len(bitstring)))
    return qc


def build_popcount_control() -> list[tuple[dict, QuantumCircuit]]:
    """Build all preregistered parity-leakage and readout circuits."""
    circuits: list[tuple[dict, QuantumCircuit]] = []
    for depth in DEPTHS:
        for label, initial, sector, popcount in STATES:
            for rep in range(REPS):
                meta = {
                    "experiment": EXPERIMENT_NAME,
                    "block": "parity_leakage",
                    "n_qubits": 4,
                    "depth": depth,
                    "state_label": label,
                    "sector": sector,
                    "initial": initial,
                    "popcount": popcount,
                    "rep": rep,
                    "shots": SHOTS_MAIN,
                    "t_step": T_STEP,
                    "phase": "2_popcount_control",
                }
                qc = build_xy_trotter_circuit(4, initial, depth, T_STEP)
                qc.name = f"PC_n4_d{depth}_{label}_r{rep}"
                circuits.append((meta, qc))

    for label, initial, sector, popcount in STATES:
        meta = {
            "experiment": EXPERIMENT_NAME,
            "block": "readout",
            "n_qubits": 4,
            "depth": 0,
            "state_label": label,
            "sector": sector,
            "initial": initial,
            "popcount": popcount,
            "rep": 0,
            "shots": SHOTS_READOUT,
            "t_step": 0.0,
            "phase": "2_popcount_control",
        }
        qc = build_readout_circuit(initial)
        qc.name = f"PC_readout_{label}"
        circuits.append((meta, qc))
    return circuits


def summarise_leakage(rows: list[dict]) -> list[dict]:
    """Summarise parity leakage by depth and state."""
    buckets: dict[tuple[int, str], list[float]] = {}
    metadata: dict[str, dict] = {}
    for row in rows:
        meta = row["meta"]
        if meta["block"] != "parity_leakage":
            continue
        stats = row.get("stats", {})
        if "parity_leakage" not in stats:
            continue
        label = str(meta["state_label"])
        key = (int(meta["depth"]), label)
        buckets.setdefault(key, []).append(float(stats["parity_leakage"]))
        metadata[label] = {
            "initial": meta["initial"],
            "sector": meta["sector"],
            "popcount": meta["popcount"],
        }

    summary: list[dict] = []
    for depth, label in sorted(buckets):
        values = buckets[(depth, label)]
        sem = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        summary.append(
            {
                "depth": depth,
                "state_label": label,
                **metadata[label],
                "mean_parity_leakage": float(np.mean(values)),
                "sem_parity_leakage": sem,
                "n_reps": len(values),
            }
        )
    return summary


def main() -> int:
    """Submit or dry-run the Phase 2 popcount-control IBM batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-qpu", action="store_true")
    parser.add_argument("--backend", default=BACKEND_NAME_DEFAULT)
    parser.add_argument("--max-live-depth", type=int, default=700)
    args = parser.parse_args()

    if not args.dry_run and not args.confirm_qpu:
        print("ERROR: real QPU submission requires --confirm-qpu", file=sys.stderr)
        return 1

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    results_dir = REPO_ROOT / ".coordination" / "ibm_runs"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"phase2_popcount_control_{timestamp}.json"

    circuits = build_popcount_control()
    print("=" * 70)
    print("Phase 2 popcount-control IBM run")
    print("=" * 70)
    print(f"Timestamp: {timestamp}")
    print(f"Backend:   {args.backend}")
    print(f"Dry-run:   {args.dry_run}")
    print(f"Circuits:  {len(circuits)}")
    print(f"Shots:     main={SHOTS_MAIN}, readout={SHOTS_READOUT}")

    if args.dry_run:
        from scpn_quantum_control.hardware.runner import HardwareRunner

        runner = HardwareRunner(
            use_simulator=True,
            results_dir=str(REPO_ROOT / "results" / "ibm_runs"),
        )
        runner.connect()
        isa = [runner.transpile(qc) for _, qc in circuits]
        depths = [qc.depth() for qc in isa]
        gates = [sum(qc.count_ops().values()) for qc in isa]
        print(
            f"Dry transpile depths: min={min(depths)}, "
            f"max={max(depths)}, mean={float(np.mean(depths)):.1f}"
        )
        print(
            f"Dry gate counts: min={min(gates)}, "
            f"max={max(gates)}, mean={float(np.mean(gates)):.1f}"
        )
        return 0

    vault = Path("/media/anulum/724AA8E84AA8AA75/agentic-shared/CREDENTIALS.md")
    api_key, instance = parse_vault(vault)

    from scpn_quantum_control.hardware.runner import HardwareRunner

    runner = HardwareRunner(
        token=api_key,
        channel="ibm_cloud",
        instance=instance,
        backend_name=args.backend,
        use_simulator=False,
        optimization_level=2,
        resilience_level=0,
        results_dir=str(REPO_ROOT / "results" / "ibm_runs"),
    )
    runner.connect()
    if args.backend != runner.backend_name:
        print(f"ERROR: connected backend changed to {runner.backend_name}", file=sys.stderr)
        return 2

    print("Live transpilation budget check...")
    isa = [runner.transpile(qc) for _, qc in circuits]
    depths = [qc.depth() for qc in isa]
    gates = [sum(qc.count_ops().values()) for qc in isa]
    depth_summary = {"min": min(depths), "max": max(depths), "mean": float(np.mean(depths))}
    gate_summary = {"min": min(gates), "max": max(gates), "mean": float(np.mean(gates))}
    print(f"Live depths: {depth_summary}")
    print(f"Live gates:  {gate_summary}")
    if depth_summary["max"] > args.max_live_depth:
        results_path.write_text(
            json.dumps(
                {
                    "status": "aborted_live_depth_budget",
                    "timestamp_utc": timestamp,
                    "backend": args.backend,
                    "max_live_depth": args.max_live_depth,
                    "live_depth_summary": depth_summary,
                    "live_gate_summary": gate_summary,
                    "n_circuits": len(circuits),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("ERROR: live depth budget exceeded; no IBM job submitted.", file=sys.stderr)
        return 3

    main_items = [(m, c) for m, c in circuits if m["shots"] == SHOTS_MAIN]
    readout_items = [(m, c) for m, c in circuits if m["shots"] == SHOTS_READOUT]
    job_ids: list[str] = []
    rows: list[dict] = []

    print(f"Submitting main block: {len(main_items)} circuits")
    t0 = time.time()
    main_results = runner.run_sampler(
        [c for _, c in main_items],
        shots=SHOTS_MAIN,
        name=f"{EXPERIMENT_NAME}_main",
        timeout_s=7200,
    )
    print(f"Main block completed in {(time.time() - t0) / 60:.2f} minutes")
    if main_results:
        job_ids.append(main_results[0].job_id)
    for (meta, _), result in zip(main_items, main_results):
        rows.append(
            {
                "meta": meta,
                "counts": result.counts,
                "stats": analyse_counts(result.counts or {}, meta),
                "job_id": result.job_id,
            }
        )

    print(f"Submitting readout block: {len(readout_items)} circuits")
    t0 = time.time()
    readout_results = runner.run_sampler(
        [c for _, c in readout_items],
        shots=SHOTS_READOUT,
        name=f"{EXPERIMENT_NAME}_readout",
        timeout_s=1800,
    )
    print(f"Readout block completed in {(time.time() - t0) / 60:.2f} minutes")
    if readout_results:
        job_ids.append(readout_results[0].job_id)
    for (meta, _), result in zip(readout_items, readout_results):
        rows.append(
            {
                "meta": meta,
                "counts": result.counts,
                "stats": analyse_counts(result.counts or {}, meta),
                "job_id": result.job_id,
            }
        )

    output = {
        "experiment": EXPERIMENT_NAME,
        "status": "complete",
        "timestamp_utc": timestamp,
        "backend": args.backend,
        "job_ids": job_ids,
        "n_circuits": len(circuits),
        "depths": list(DEPTHS),
        "reps": REPS,
        "shots_main": SHOTS_MAIN,
        "shots_readout": SHOTS_READOUT,
        "max_live_depth": args.max_live_depth,
        "live_depth_summary": depth_summary,
        "live_gate_summary": gate_summary,
        "circuits": rows,
        "leakage_summary": summarise_leakage(rows),
    }
    results_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"Results saved: {results_path}")
    print(f"Job IDs: {', '.join(job_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
