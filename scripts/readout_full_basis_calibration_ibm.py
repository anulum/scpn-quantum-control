#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Full-Basis Readout Calibration
"""Approval-gated n=4 full-basis readout calibration for a fixed layout."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from phase1_mini_bench_ibm_kingston import parse_vault  # noqa: E402

from scpn_quantum_control.hardware.runner import HardwareRunner  # noqa: E402

EXPERIMENT = "readout_full_basis_n4"
DEFAULT_BACKEND = "ibm_marrakesh"
DEFAULT_PHYSICAL_QUBITS = [5, 6, 7, 8]
SHOTS = 8192


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def _bitstrings(n: int) -> list[str]:
    return [format(i, f"0{n}b") for i in range(2**n)]


def _summary(values: list[int]) -> dict[str, float | int]:
    return {"min": min(values), "max": max(values), "mean": float(mean(values))}


def _backend_status(backend: Any) -> dict[str, Any]:
    status = backend.status() if hasattr(backend, "status") else None
    return {
        "name": str(getattr(backend, "name", "unknown")),
        "num_qubits": getattr(backend, "num_qubits", None),
        "operational": getattr(status, "operational", None) if status else None,
        "pending_jobs": getattr(status, "pending_jobs", None) if status else None,
        "status_msg": getattr(status, "status_msg", None) if status else None,
    }


def build_calibration_circuits(
    *,
    physical_qubits: list[int],
    backend_qubits: int,
) -> list[tuple[dict[str, Any], QuantumCircuit]]:
    """Build full-basis readout calibration circuits for the physical layout."""
    circuits: list[tuple[dict[str, Any], QuantumCircuit]] = []
    for prepared in _bitstrings(len(physical_qubits)):
        qc = QuantumCircuit(backend_qubits, len(physical_qubits))
        for logical, bit in enumerate(prepared):
            if bit == "1":
                qc.x(physical_qubits[logical])
        for logical, physical in enumerate(physical_qubits):
            qc.measure(physical, logical)
        qc.name = f"readout_full_basis_{prepared}"
        meta = {
            "experiment": EXPERIMENT,
            "prepared": prepared,
            "n_logical_qubits": len(physical_qubits),
            "physical_qubits": physical_qubits,
            "shots": SHOTS,
        }
        circuits.append((meta, qc))
    return circuits


def _assignment_rows(circuits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for circuit in circuits:
        prepared = circuit["meta"]["prepared"]
        counts = circuit.get("counts") or {}
        total = sum(counts.values())
        retained = counts.get(prepared[::-1], 0)
        parity = prepared.count("1") % 2
        parity_flip = 0
        for observed, count in counts.items():
            if observed.replace(" ", "").count("1") % 2 != parity:
                parity_flip += count
        rows.append(
            {
                "prepared": prepared,
                "total_shots": total,
                "retention": retained / total if total else None,
                "parity_flip": parity_flip / total if total else None,
                "counts": counts,
            }
        )
    return rows


def main() -> int:
    """Run or prepare the full-basis IBM readout calibration batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument(
        "--physical-qubits",
        default=",".join(str(q) for q in DEFAULT_PHYSICAL_QUBITS),
        help="Comma-separated physical qubits matching the target data layout.",
    )
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-budget", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=1200)
    args = parser.parse_args()

    if args.submit and not args.confirm_budget:
        print("ERROR: --submit requires --confirm-budget", file=sys.stderr)
        return 2

    physical_qubits = [int(part) for part in args.physical_qubits.split(",") if part]
    if len(physical_qubits) != 4:
        print("ERROR: this preregistered runner is n=4 only", file=sys.stderr)
        return 2

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
    info = _backend_status(runner.backend)
    if info["operational"] is not True:
        raise RuntimeError(f"backend is not operational: {info}")
    backend_qubits = int(info["num_qubits"])

    timestamp = _timestamp()
    out_dir = REPO_ROOT / "data" / "readout_full_basis"
    out_path = out_dir / f"readout_full_basis_{args.backend}_4q_{timestamp}.json"
    circuits = build_calibration_circuits(
        physical_qubits=physical_qubits,
        backend_qubits=backend_qubits,
    )
    isa = [runner.transpile(qc) for _, qc in circuits]
    depths = [c.depth() for c in isa]
    gate_counts = [sum(c.count_ops().values()) for c in isa]
    estimated_qpu_minutes = len(circuits) * 0.55 / 60.0
    payload: dict[str, Any] = {
        "schema": "scpn_readout_full_basis_v1",
        "status": "readiness_passed",
        "timestamp_utc": timestamp,
        "backend": args.backend,
        "backend_status": info,
        "target_dataset": "data/phase3_multidevice_dla/phase3_multidevice_ibm_marrakesh_2026-05-06T171231Z.json",
        "physical_qubits": physical_qubits,
        "n_logical_qubits": 4,
        "shots": SHOTS,
        "n_circuits": len(circuits),
        "estimated_qpu_minutes": estimated_qpu_minutes,
        "budget_ceiling_minutes": 5,
        "depth_summary": _summary(depths),
        "gate_summary": _summary(gate_counts),
        "metas": [meta for meta, _ in circuits],
        "job_ids": [],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"Backend: {args.backend}")
    print(f"Physical qubits: {physical_qubits}")
    print(f"Readiness artefact: {out_path.relative_to(REPO_ROOT)}")
    print(f"Circuits: {len(circuits)}")
    print(f"Estimated QPU minutes: {estimated_qpu_minutes:.2f} (ceiling 5)")
    print(f"Depth summary: {payload['depth_summary']}")
    print(f"Gate summary: {payload['gate_summary']}")

    if not args.submit:
        print("Readiness passed. Re-run with --submit --confirm-budget to submit.")
        return 0
    if estimated_qpu_minutes > 5:
        payload["status"] = "aborted_estimated_qpu_ceiling"
        out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print("ERROR: estimate exceeds 5-minute ceiling; no job submitted.", file=sys.stderr)
        return 3

    start = time.time()
    results = runner.run_sampler(
        [qc for _, qc in circuits],
        shots=SHOTS,
        name=EXPERIMENT,
        timeout_s=args.timeout_s,
    )
    wall = time.time() - start
    if results:
        payload["job_ids"].append(results[0].job_id)
    circuit_rows: list[dict[str, Any]] = []
    for meta, result in zip([m for m, _ in circuits], results):
        circuit_rows.append(
            {
                "meta": meta,
                "counts": result.counts,
                "job_id": result.job_id,
                "metadata": result.metadata,
            }
        )
    payload.update(
        {
            "status": "completed",
            "wall_time_s": wall,
            "circuits": circuit_rows,
            "assignment_rows": _assignment_rows(circuit_rows),
        }
    )
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Completed. Jobs: {payload['job_ids']}")
    print(f"Saved: {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
