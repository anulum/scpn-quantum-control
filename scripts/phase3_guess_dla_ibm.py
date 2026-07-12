#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — phase3 guess DLA IBM script
# scpn-quantum-control -- Phase 3 GUESS DLA hardware calibration
"""Approval-gated Phase 3 GUESS / symmetry-decay hardware run.

Implements ``docs/campaigns/guess_symmetry_decay_prereg_2026-05-06.md``. The default
mode performs live readiness only and records no raw hardware counts. Hardware
submission requires both ``--submit`` and ``--confirm-budget``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from qiskit import QuantumCircuit, transpile

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
from phase2_full_campaign_ibm import fold_circuit  # noqa: E402

from scpn_quantum_control.hardware.runner import HardwareRunner, _extract_counts  # noqa: E402

EXPERIMENT = "phase3_guess_dla"
DEFAULT_BACKEND = "ibm_marrakesh"
MAIN_SHOTS = 4096
READOUT_SHOTS = 8192
DEPTHS = (6, 8, 10, 14)
NOISE_SCALES = (1, 3, 5)
REPS = 8
STATES = {
    "even": "0011",
    "odd": "0001",
}
READOUT_STATES = ("0011", "0001", "0000", "1111")
BUDGET_CEILING_MINUTES = 15.0
CONSERVATIVE_QPU_ESTIMATE_MINUTES = 12.0
MAX_DEPTH_SCALE5_OVERHEAD = 1.25
DEFAULT_LAYOUT = (5, 6, 7, 8)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def _summary(values: Sequence[float | int]) -> dict[str, float | int]:
    return {
        "min": min(values),
        "max": max(values),
        "mean": float(mean(values)),
    }


def _backend_name(backend: Any) -> str:
    name = getattr(backend, "name", "unknown")
    return name() if callable(name) else str(name)


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
    if info["operational"] is not True:
        raise RuntimeError(f"backend is not operational: {info}")
    if info["num_qubits"] != 156:
        raise RuntimeError(f"backend is not a 156-qubit Heron-class target: {info}")


def build_guess_circuits(
    *,
    physical_qubits: Sequence[int] = DEFAULT_LAYOUT,
) -> tuple[
    list[tuple[dict[str, Any], QuantumCircuit]], list[tuple[dict[str, Any], QuantumCircuit]]
]:
    """Build preregistered folded main circuits and four readout circuits."""
    main: list[tuple[dict[str, Any], QuantumCircuit]] = []
    for depth in DEPTHS:
        for sector, initial in STATES.items():
            base = build_xy_trotter_circuit(4, initial, depth, T_STEP)
            for scale in NOISE_SCALES:
                folded = fold_circuit(base, scale)
                for rep in range(REPS):
                    meta = {
                        "experiment": EXPERIMENT,
                        "block": "main",
                        "n_qubits": 4,
                        "physical_qubits": list(physical_qubits),
                        "depth": depth,
                        "sector": sector,
                        "initial": initial,
                        "noise_scale": scale,
                        "folding_rule": "global_unitary_folding_U_Udagger_U",
                        "rep": rep,
                        "shots": MAIN_SHOTS,
                        "t_step": T_STEP,
                    }
                    circuit = folded.copy()
                    circuit.name = f"p3_guess_d{depth}_{sector}_g{scale}_r{rep}"
                    main.append((meta, circuit))
    readout: list[tuple[dict[str, Any], QuantumCircuit]] = []
    for initial in READOUT_STATES:
        meta = {
            "experiment": EXPERIMENT,
            "block": "readout",
            "n_qubits": 4,
            "physical_qubits": list(physical_qubits),
            "depth": 0,
            "sector": "readout",
            "initial": initial,
            "noise_scale": 0,
            "folding_rule": "readout_only",
            "rep": 0,
            "shots": READOUT_SHOTS,
            "t_step": 0.0,
        }
        circuit = build_readout_baseline_circuit(4, initial)
        circuit.name = f"p3_guess_readout_{initial}"
        readout.append((meta, circuit))
    return main, readout


def transpile_with_layout(
    backend: Any,
    circuits: Sequence[tuple[dict[str, Any], QuantumCircuit]],
    *,
    optimization_level: int,
) -> list[QuantumCircuit]:
    """Transpile circuits with the preregistered fixed physical layout."""
    isa: list[QuantumCircuit] = []
    for meta, circuit in circuits:
        isa.append(
            transpile(
                circuit,
                backend=backend,
                initial_layout=meta["physical_qubits"],
                optimization_level=optimization_level,
            )
        )
    return isa


def _two_qubit_gate_count(circuit: QuantumCircuit) -> int:
    """Count operations acting on exactly two qubits."""
    return sum(1 for instruction in circuit.data if len(instruction.qubits) == 2)


def readiness(
    backend: Any,
    circuits: Sequence[tuple[dict[str, Any], QuantumCircuit]],
    isa_circuits: Sequence[QuantumCircuit],
) -> dict[str, Any]:
    """Return live folded-circuit readiness diagnostics."""
    depths = [circuit.depth() for circuit in isa_circuits]
    total_gates = [sum(circuit.count_ops().values()) for circuit in isa_circuits]
    two_qubit_gates = [_two_qubit_gate_count(circuit) for circuit in isa_circuits]
    by_depth_scale: dict[tuple[int, int], list[int]] = defaultdict(list)
    for (meta, _), circuit in zip(circuits, isa_circuits):
        if meta["block"] == "main":
            by_depth_scale[(int(meta["depth"]), int(meta["noise_scale"]))].append(circuit.depth())
    overhead_failures: list[dict[str, Any]] = []
    for depth in DEPTHS:
        scale1 = max(by_depth_scale[(depth, 1)])
        scale5 = max(by_depth_scale[(depth, 5)])
        allowed = scale1 * (5.0 * MAX_DEPTH_SCALE5_OVERHEAD)
        if scale5 > allowed:
            overhead_failures.append(
                {
                    "depth": depth,
                    "scale1_max_depth": scale1,
                    "scale5_max_depth": scale5,
                    "allowed_scale5_depth": allowed,
                }
            )
    accepted = not overhead_failures
    reason = None
    if overhead_failures:
        reason = "scale-5 folded depth exceeds preregistered overhead guard"
    return {
        "accepted": accepted,
        "backend_status": _backend_status(backend),
        "depth_summary": _summary(depths),
        "total_gate_summary": _summary(total_gates),
        "two_qubit_gate_summary": _summary(two_qubit_gates),
        "folded_depth_max_by_depth_scale": {
            f"d{depth}_g{scale}": max(values)
            for (depth, scale), values in sorted(by_depth_scale.items())
        },
        "overhead_failures": overhead_failures,
        "rejection_reason": reason,
    }


def _save(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    path.write_text(encoded, encoding="utf-8")
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _append_execution_log(timestamp: str, payload: dict[str, Any], path: Path) -> None:
    log_path = REPO_ROOT / ".coordination" / "IBM_EXECUTION_LOG.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    jobs = ", ".join(f"`{job}`" for job in payload.get("job_ids", []))
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {timestamp} - PHASE 3 GUESS DLA\n\n")
        handle.write(f"- **Backend:** {payload['backend']}\n")
        handle.write(f"- **Status:** {payload['status']}\n")
        handle.write(f"- **Circuits:** {payload['n_circuits']}\n")
        handle.write(f"- **Job IDs:** {jobs or 'none'}\n")
        handle.write(f"- **Artefact:** `{path.relative_to(REPO_ROOT)}`\n")
        handle.write("- **Boundary:** folded-noise witness calibration only.\n")


def _job_id(job: Any) -> str:
    value = job.job_id if hasattr(job, "job_id") else None
    return str(value() if callable(value) else value)


def _run_isa_sampler(
    backend: Any,
    isa_circuits: Sequence[QuantumCircuit],
    *,
    shots: int,
    timeout_s: float,
) -> tuple[str, list[dict[str, int]], float]:
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    sampler = Sampler(mode=backend)
    sampler.options.default_shots = shots
    start = time.time()
    job = sampler.run(list(isa_circuits))
    job_id = _job_id(job)
    result = job.result(timeout=timeout_s)
    wall = time.time() - start
    return job_id, [_extract_counts(pub_result) for pub_result in result], wall


def _result_rows(
    metas: Sequence[dict[str, Any]],
    counts_rows: Sequence[dict[str, int]],
    job_id: str,
    isa_circuits: Sequence[QuantumCircuit],
) -> list[dict[str, Any]]:
    rows = []
    for meta, counts, circuit in zip(metas, counts_rows, isa_circuits):
        rows.append(
            {
                "meta": meta,
                "counts": counts,
                "stats": analyse_counts(counts, meta),
                "job_id": job_id,
                "metadata": {
                    "depth": circuit.depth(),
                    "total_gates": sum(circuit.count_ops().values()),
                    "two_qubit_gates": _two_qubit_gate_count(circuit),
                },
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-budget", action="store_true")
    parser.add_argument("--timeout-main-s", type=int, default=5400)
    parser.add_argument("--timeout-readout-s", type=int, default=1800)
    parser.add_argument("--optimization-level", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    """Run readiness or approved submission."""
    args = parse_args()
    if args.submit and not args.confirm_budget:
        print("ERROR: --submit requires --confirm-budget", file=sys.stderr)
        return 2

    timestamp = _timestamp()
    out_dir = REPO_ROOT / "data" / "phase3_guess_dla"
    out_path = out_dir / f"phase3_guess_{args.backend}_{timestamp}.json"
    credential_value, instance = parse_vault(
        Path("~/.config/scpn-quantum-control/credentials.md").expanduser()
    )
    runner = HardwareRunner(
        credential_value,
        "ibm_cloud",
        instance,
        args.backend,
        False,
        args.optimization_level,
        0,
        True,
        str(REPO_ROOT / "results" / "ibm_runs"),
    )
    runner.connect()
    _validate_backend(runner.backend)

    main_circuits, readout_circuits = build_guess_circuits()
    all_circuits = main_circuits + readout_circuits
    isa_main = transpile_with_layout(
        runner.backend,
        main_circuits,
        optimization_level=args.optimization_level,
    )
    isa_readout = transpile_with_layout(
        runner.backend,
        readout_circuits,
        optimization_level=args.optimization_level,
    )
    isa_all = isa_main + isa_readout
    ready = readiness(runner.backend, all_circuits, isa_all)
    est_qpu_minutes = CONSERVATIVE_QPU_ESTIMATE_MINUTES
    payload: dict[str, Any] = {
        "schema": "scpn_phase3_guess_dla_v1",
        "status": "readiness_passed" if ready["accepted"] else "readiness_rejected",
        "timestamp_utc": timestamp,
        "backend": runner.backend_name,
        "experiment": EXPERIMENT,
        "n_circuits": len(all_circuits),
        "n_main_circuits": len(main_circuits),
        "n_readout_circuits": len(readout_circuits),
        "shots_main": MAIN_SHOTS,
        "shots_readout": READOUT_SHOTS,
        "depths": list(DEPTHS),
        "noise_scales": list(NOISE_SCALES),
        "repetitions": REPS,
        "states": STATES,
        "readout_states": list(READOUT_STATES),
        "physical_qubits": list(DEFAULT_LAYOUT),
        "estimated_qpu_minutes": est_qpu_minutes,
        "budget_ceiling_minutes": BUDGET_CEILING_MINUTES,
        "readiness": ready,
        "metas_main": [meta for meta, _ in main_circuits],
        "metas_readout": [meta for meta, _ in readout_circuits],
        "job_ids": [],
    }
    sha = _save(out_path, payload)
    print(f"Backend: {runner.backend_name}")
    print(f"Readiness artefact: {out_path.relative_to(REPO_ROOT)}")
    print(f"SHA256: {sha}")
    print(
        f"Circuits: {len(all_circuits)} ({len(main_circuits)} main, {len(readout_circuits)} readout)"
    )
    print(f"Estimated QPU minutes: {est_qpu_minutes:.2f} (ceiling {BUDGET_CEILING_MINUTES:.0f})")
    print(f"Depth summary: {ready['depth_summary']}")
    print(f"Two-qubit gate summary: {ready['two_qubit_gate_summary']}")
    if not ready["accepted"]:
        print(f"READINESS REJECTED: {ready['rejection_reason']}", file=sys.stderr)
        _append_execution_log(timestamp, payload, out_path)
        return 3
    if not args.submit:
        print("Readiness passed. Re-run with --submit --confirm-budget to submit.")
        return 0
    if est_qpu_minutes > BUDGET_CEILING_MINUTES:
        payload["status"] = "aborted_estimated_qpu_ceiling"
        _save(out_path, payload)
        _append_execution_log(timestamp, payload, out_path)
        print("ERROR: estimate exceeds QPU ceiling; no job submitted.", file=sys.stderr)
        return 4

    print("Submitting GUESS main batch...")
    main_job, main_counts, wall_main = _run_isa_sampler(
        runner.backend,
        isa_main,
        shots=MAIN_SHOTS,
        timeout_s=args.timeout_main_s,
    )
    payload["job_ids"].append(main_job)
    print("Submitting GUESS readout batch...")
    readout_job, readout_counts, wall_readout = _run_isa_sampler(
        runner.backend,
        isa_readout,
        shots=READOUT_SHOTS,
        timeout_s=args.timeout_readout_s,
    )
    payload["job_ids"].append(readout_job)
    payload.update(
        {
            "status": "completed",
            "wall_time_main_s": wall_main,
            "wall_time_readout_s": wall_readout,
            "circuits": _result_rows(
                [meta for meta, _ in main_circuits],
                main_counts,
                main_job,
                isa_main,
            )
            + _result_rows(
                [meta for meta, _ in readout_circuits],
                readout_counts,
                readout_job,
                isa_readout,
            ),
        }
    )
    final_sha = _save(out_path, payload)
    _append_execution_log(timestamp, payload, out_path)
    print(f"Completed. Jobs: {payload['job_ids']}")
    print(f"Saved: {out_path.relative_to(REPO_ROOT)} sha256={final_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
