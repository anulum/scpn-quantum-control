#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase 3 Multi-Device DLA Replication
"""Preregistered second-backend DLA parity replication.

This script implements the circuit matrix from
``docs/dla_multidevice_replication_prereg_2026-05-06.md``:

* backend must be account-visible, operational, Heron-class, and not
  ``ibm_kingston``;
* main circuits are n=4, states 0011/0001, depths 4,6,8,10,14,20,
  12 repetitions, 4096 shots;
* readout circuits are 0011/0001/0000/1111 at 8192 shots;
* readiness records live transpilation depth/gate summaries before any
  submission;
* submission requires ``--submit`` and ``--confirm-budget``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from phase1_mini_bench_ibm_kingston import (  # noqa: E402
    T_STEP,
    analyse_counts,
    build_readout_baseline_circuit,
    build_xy_trotter_circuit,
    parse_vault,
)

from scpn_quantum_control.hardware.runner import HardwareRunner  # noqa: E402

EXPERIMENT = "phase3_multidevice_dla"
DEFAULT_BACKEND = "ibm_marrakesh"
MAIN_SHOTS = 4096
READOUT_SHOTS = 8192
DEPTHS = [4, 6, 8, 10, 14, 20]
REPS = 12
STATES = {"even": "0011", "odd": "0001"}
READOUT_STATES = ["0011", "0001", "0000", "1111"]
PHASE2_AG_OBSERVED_MAX_DEPTH = 1014


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def _summary(values: list[float | int]) -> dict[str, float | int]:
    return {
        "min": min(values),
        "max": max(values),
        "mean": float(mean(values)),
    }


def _backend_name(backend: Any) -> str:
    return str(getattr(backend, "name", "unknown"))


def _backend_status(backend: Any) -> dict[str, Any]:
    status = backend.status() if hasattr(backend, "status") else None
    return {
        "name": _backend_name(backend),
        "num_qubits": getattr(backend, "num_qubits", None),
        "operational": getattr(status, "operational", None) if status else None,
        "pending_jobs": getattr(status, "pending_jobs", None) if status else None,
        "status_msg": getattr(status, "status_msg", None) if status else None,
    }


def _validate_backend(backend: Any) -> None:
    info = _backend_status(backend)
    name = str(info["name"])
    if name == "ibm_kingston":
        raise RuntimeError("phase3 replication must not run on ibm_kingston")
    if info["operational"] is not True:
        raise RuntimeError(f"backend is not operational: {info}")
    if info["num_qubits"] != 156:
        raise RuntimeError(f"backend is not a 156-qubit Heron-class target: {info}")


def build_circuits() -> tuple[
    list[tuple[dict[str, Any], QuantumCircuit]], list[tuple[dict[str, Any], QuantumCircuit]]
]:
    main: list[tuple[dict[str, Any], QuantumCircuit]] = []
    for depth in DEPTHS:
        for sector, initial in STATES.items():
            for rep in range(REPS):
                meta = {
                    "experiment": EXPERIMENT,
                    "block": "main",
                    "n_qubits": 4,
                    "depth": depth,
                    "sector": sector,
                    "initial": initial,
                    "rep": rep,
                    "shots": MAIN_SHOTS,
                    "t_step": T_STEP,
                }
                qc = build_xy_trotter_circuit(4, initial, depth, T_STEP)
                qc.name = f"phase3_n4_d{depth}_{sector}_r{rep}"
                main.append((meta, qc))

    readout: list[tuple[dict[str, Any], QuantumCircuit]] = []
    for initial in READOUT_STATES:
        meta = {
            "experiment": EXPERIMENT,
            "block": "readout",
            "n_qubits": 4,
            "depth": 0,
            "sector": "readout",
            "initial": initial,
            "rep": 0,
            "shots": READOUT_SHOTS,
            "t_step": 0.0,
        }
        qc = build_readout_baseline_circuit(4, initial)
        qc.name = f"phase3_readout_{initial}"
        readout.append((meta, qc))
    return main, readout


def readiness(
    runner: HardwareRunner,
    all_circuits: list[tuple[dict[str, Any], QuantumCircuit]],
    *,
    max_depth: int,
) -> dict[str, Any]:
    isa = [runner.transpile(qc) for _, qc in all_circuits]
    depths = [c.depth() for c in isa]
    total_gates = [sum(c.count_ops().values()) for c in isa]
    ecr_gates = [c.count_ops().get("ecr", 0) for c in isa]
    by_depth: dict[int, list[int]] = defaultdict(list)
    for meta, circuit in zip((m for m, _ in all_circuits), isa):
        by_depth[int(meta["depth"])].append(circuit.depth())

    depth_summary = _summary(depths)
    accepted = int(depth_summary["max"]) <= max_depth
    return {
        "accepted": accepted,
        "max_depth": max_depth,
        "reference_phase2_ag_observed_max_depth": PHASE2_AG_OBSERVED_MAX_DEPTH,
        "depth_summary": depth_summary,
        "total_gate_summary": _summary(total_gates),
        "ecr_gate_summary": _summary(ecr_gates),
        "per_preregistered_depth_max": {
            str(depth): max(vals) for depth, vals in sorted(by_depth.items())
        },
        "rejection_reason": None
        if accepted
        else f"max depth {depth_summary['max']} exceeds guard {max_depth}",
    }


def _append_execution_log(timestamp: str, payload: dict[str, Any], path: Path) -> None:
    log_path = REPO_ROOT / ".coordination" / "IBM_EXECUTION_LOG.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    job_ids = ", ".join(f"`{job}`" for job in payload.get("job_ids", []))
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"\n## {timestamp} — PHASE 3 MULTI-DEVICE DLA\n\n")
        handle.write(f"- **Backend:** {payload['backend']}\n")
        handle.write(f"- **Status:** {payload['status']}\n")
        handle.write(f"- **Circuits:** {payload['n_circuits']}\n")
        handle.write(f"- **Job IDs:** {job_ids or 'none'}\n")
        handle.write(f"- **Artefact:** `{path.relative_to(REPO_ROOT)}`\n")
        handle.write("- **Boundary:** second-backend n=4 DLA replication only.\n")


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-budget", action="store_true")
    parser.add_argument("--max-depth", type=int, default=int(PHASE2_AG_OBSERVED_MAX_DEPTH * 1.25))
    parser.add_argument("--timeout-main-s", type=int, default=3600)
    parser.add_argument("--timeout-readout-s", type=int, default=1200)
    args = parser.parse_args()

    if args.submit and not args.confirm_budget:
        print("ERROR: --submit requires --confirm-budget", file=sys.stderr)
        return 2

    main_circuits, readout_circuits = build_circuits()
    all_circuits = main_circuits + readout_circuits
    timestamp = _timestamp()
    out_dir = REPO_ROOT / "data" / "phase3_multidevice_dla"
    out_path = out_dir / f"phase3_multidevice_{args.backend}_{timestamp}.json"

    vault = Path("/media/anulum/724AA8E84AA8AA75/agentic-shared/CREDENTIALS.md")
    token, instance = parse_vault(vault)
    runner = HardwareRunner(
        token=token,
        channel="ibm_cloud",
        instance=instance,
        backend_name=args.backend,
        use_simulator=False,
        optimization_level=2,
        resilience_level=0,
        results_dir=str(REPO_ROOT / "results" / "ibm_runs"),
    )
    runner.connect()
    _validate_backend(runner.backend)

    ready = readiness(runner, all_circuits, max_depth=args.max_depth)
    est_qpu_minutes = len(all_circuits) * 0.55 / 60.0
    payload: dict[str, Any] = {
        "schema": "scpn_phase3_multidevice_dla_v1",
        "status": "readiness_passed" if ready["accepted"] else "readiness_rejected",
        "timestamp_utc": timestamp,
        "backend": runner.backend_name,
        "backend_status": _backend_status(runner.backend),
        "experiment": EXPERIMENT,
        "n_circuits": len(all_circuits),
        "n_main_circuits": len(main_circuits),
        "n_readout_circuits": len(readout_circuits),
        "shots_main": MAIN_SHOTS,
        "shots_readout": READOUT_SHOTS,
        "depths": DEPTHS,
        "repetitions": REPS,
        "states": STATES,
        "readout_states": READOUT_STATES,
        "estimated_qpu_minutes": est_qpu_minutes,
        "budget_ceiling_minutes": 10,
        "readiness": ready,
        "metas_main": [meta for meta, _ in main_circuits],
        "metas_readout": [meta for meta, _ in readout_circuits],
        "job_ids": [],
    }
    _save(out_path, payload)

    print(f"Backend: {runner.backend_name}")
    print(f"Readiness artefact: {out_path.relative_to(REPO_ROOT)}")
    print(
        f"Circuits: {len(all_circuits)} ({len(main_circuits)} main, {len(readout_circuits)} readout)"
    )
    print(f"Estimated QPU minutes: {est_qpu_minutes:.2f} (ceiling 10)")
    print(f"Depth summary: {ready['depth_summary']}")
    print(f"ECR summary: {ready['ecr_gate_summary']}")

    if not ready["accepted"]:
        print(f"READINESS REJECTED: {ready['rejection_reason']}", file=sys.stderr)
        _append_execution_log(timestamp, payload, out_path)
        return 3
    if not args.submit:
        print("Readiness passed. Re-run with --submit --confirm-budget to submit.")
        return 0

    if est_qpu_minutes > 10:
        payload["status"] = "aborted_estimated_qpu_ceiling"
        _save(out_path, payload)
        print("ERROR: estimate exceeds 10-minute ceiling; no job submitted.", file=sys.stderr)
        _append_execution_log(timestamp, payload, out_path)
        return 4

    print("Submitting main batch...")
    start = time.time()
    main_results = runner.run_sampler(
        [qc for _, qc in main_circuits],
        shots=MAIN_SHOTS,
        name=f"{EXPERIMENT}_main",
        timeout_s=args.timeout_main_s,
    )
    wall_main = time.time() - start
    main_job = main_results[0].job_id if main_results else None
    if main_job:
        payload["job_ids"].append(main_job)

    print("Submitting readout batch...")
    start = time.time()
    readout_results = runner.run_sampler(
        [qc for _, qc in readout_circuits],
        shots=READOUT_SHOTS,
        name=f"{EXPERIMENT}_readout",
        timeout_s=args.timeout_readout_s,
    )
    wall_readout = time.time() - start
    readout_job = readout_results[0].job_id if readout_results else None
    if readout_job:
        payload["job_ids"].append(readout_job)

    circuits_out: list[dict[str, Any]] = []
    for meta, result in zip([m for m, _ in main_circuits], main_results):
        circuits_out.append(
            {
                "meta": meta,
                "counts": result.counts,
                "stats": analyse_counts(result.counts or {}, meta),
                "job_id": result.job_id,
                "metadata": result.metadata,
            }
        )
    for meta, result in zip([m for m, _ in readout_circuits], readout_results):
        circuits_out.append(
            {
                "meta": meta,
                "counts": result.counts,
                "stats": analyse_counts(result.counts or {}, meta),
                "job_id": result.job_id,
                "metadata": result.metadata,
            }
        )

    payload.update(
        {
            "status": "completed",
            "wall_time_main_s": wall_main,
            "wall_time_readout_s": wall_readout,
            "circuits": circuits_out,
        }
    )
    _save(out_path, payload)
    _append_execution_log(timestamp, payload, out_path)
    print(f"Completed. Jobs: {payload['job_ids']}")
    print(f"Saved: {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
