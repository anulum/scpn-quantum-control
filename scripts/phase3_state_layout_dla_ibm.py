#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — phase3 state layout DLA IBM script
# scpn-quantum-control -- Phase 3 state/layout DLA randomisation
"""Preregistered state/layout randomisation control for DLA parity.

Implements ``docs/campaigns/dla_state_layout_randomisation_prereg_2026-05-06.md``.
The script is fail-closed: it records live readiness by default and submits
only when both ``--submit`` and ``--confirm-budget`` are supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
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

from scpn_quantum_control.hardware.runner import HardwareRunner, _extract_counts  # noqa: E402

EXPERIMENT = "phase3_state_layout_dla"
DEFAULT_BACKEND = "ibm_marrakesh"
MAIN_SHOTS = 4096
READOUT_SHOTS = 8192
DEPTHS = (6, 8, 10, 14)
REPS = 8
STATES = {
    "E0": {"initial": "0011", "sector": "even", "popcount": 2},
    "E1": {"initial": "0101", "sector": "even", "popcount": 2},
    "O0": {"initial": "0001", "sector": "odd", "popcount": 1},
    "O1": {"initial": "0010", "sector": "odd", "popcount": 1},
    "O3": {"initial": "0111", "sector": "odd", "popcount": 3},
}
POPCOUNT_CONTROL_MAX_DEPTH = 700
POPCOUNT_CONTROL_GATE_LIMIT = 1500
BUDGET_CEILING_MINUTES = 20.0


@dataclass(frozen=True)
class LayoutCandidate:
    """Connected four-qubit physical layout candidate."""

    layout_id: str
    physical_qubits: tuple[int, int, int, int]
    readout_error_mean: float | None
    two_qubit_error_mean: float | None
    score: float


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def _summary(values: Sequence[float | int]) -> dict[str, float | int]:
    return {"min": min(values), "max": max(values), "mean": float(mean(values))}


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


def _coupling_edges(backend: Any) -> set[tuple[int, int]]:
    coupling_map = getattr(backend, "coupling_map", None)
    if coupling_map is not None and hasattr(coupling_map, "get_edges"):
        raw_edges = coupling_map.get_edges()
    else:
        config = backend.configuration() if hasattr(backend, "configuration") else None
        raw_edges = getattr(config, "coupling_map", []) if config is not None else []
    edges: set[tuple[int, int]] = set()
    for edge in raw_edges:
        if len(edge) != 2:
            continue
        a, b = int(edge[0]), int(edge[1])
        edges.add((a, b))
        edges.add((b, a))
    return edges


def _is_connected_window(window: tuple[int, int, int, int], edges: set[tuple[int, int]]) -> bool:
    seen = {window[0]}
    frontier = [window[0]]
    allowed = set(window)
    while frontier:
        node = frontier.pop()
        neighbours = {b for a, b in edges if a == node and b in allowed}
        new_nodes = neighbours.difference(seen)
        seen.update(new_nodes)
        frontier.extend(new_nodes)
    return seen == allowed


def _readout_errors(backend: Any, qubits: Sequence[int]) -> list[float]:
    try:
        props = backend.properties()
    except Exception:
        return []
    values: list[float] = []
    for qubit in qubits:
        value = None
        try:
            value = props.readout_error(int(qubit))
        except Exception:
            value = None
        if value is not None:
            values.append(float(value))
    return values


def _two_qubit_errors(backend: Any, edges: Iterable[tuple[int, int]]) -> list[float]:
    try:
        props = backend.properties()
    except Exception:
        return []
    values: list[float] = []
    for edge in edges:
        value = None
        try:
            value = props.gate_error("ecr", list(edge))
        except Exception:
            try:
                value = props.gate_error("cx", list(edge))
            except Exception:
                value = None
        if value is not None:
            values.append(float(value))
    return values


def select_layouts(backend: Any, *, n_layouts: int = 3) -> list[LayoutCandidate]:
    """Select connected four-qubit windows before outcome data exists."""

    n_qubits = int(getattr(backend, "num_qubits", 0))
    edges = _coupling_edges(backend)
    if n_qubits < 4 or not edges:
        raise RuntimeError("backend does not expose enough coupling-map metadata")
    candidates: list[LayoutCandidate] = []
    for start in range(n_qubits - 3):
        window = (start, start + 1, start + 2, start + 3)
        if not _is_connected_window(window, edges):
            continue
        readout = _readout_errors(backend, window)
        local_edges = [(a, b) for a, b in edges if a in window and b in window and a < b]
        twoq = _two_qubit_errors(backend, local_edges)
        readout_mean = float(mean(readout)) if readout else None
        twoq_mean = float(mean(twoq)) if twoq else None
        score = (readout_mean if readout_mean is not None else 0.02) + (
            twoq_mean if twoq_mean is not None else 0.02
        )
        candidates.append(
            LayoutCandidate(
                layout_id=f"L{len(candidates)}",
                physical_qubits=window,
                readout_error_mean=readout_mean,
                two_qubit_error_mean=twoq_mean,
                score=score,
            )
        )
    if len(candidates) < n_layouts:
        raise RuntimeError(f"only {len(candidates)} connected windows found; need {n_layouts}")
    return sorted(candidates, key=lambda item: item.score)[:n_layouts]


def build_circuits(
    layouts: Sequence[LayoutCandidate],
) -> tuple[
    list[tuple[dict[str, Any], QuantumCircuit]], list[tuple[dict[str, Any], QuantumCircuit]]
]:
    """Build preregistered logical circuits and attach physical-layout metadata."""

    main: list[tuple[dict[str, Any], QuantumCircuit]] = []
    for layout in layouts:
        for depth in DEPTHS:
            for state_label, state in STATES.items():
                for rep in range(REPS):
                    initial = str(state["initial"])
                    meta = {
                        "experiment": EXPERIMENT,
                        "block": "main",
                        "n_qubits": 4,
                        "layout_id": layout.layout_id,
                        "physical_qubits": list(layout.physical_qubits),
                        "depth": depth,
                        "state_label": state_label,
                        "sector": state["sector"],
                        "popcount": state["popcount"],
                        "initial": initial,
                        "rep": rep,
                        "shots": MAIN_SHOTS,
                        "t_step": T_STEP,
                    }
                    qc = build_xy_trotter_circuit(4, initial, depth, T_STEP)
                    qc.name = f"p3_layout_{layout.layout_id}_d{depth}_{state_label}_r{rep}"
                    main.append((meta, qc))
    readout: list[tuple[dict[str, Any], QuantumCircuit]] = []
    for layout in layouts:
        for state_label, state in STATES.items():
            initial = str(state["initial"])
            meta = {
                "experiment": EXPERIMENT,
                "block": "readout",
                "n_qubits": 4,
                "layout_id": layout.layout_id,
                "physical_qubits": list(layout.physical_qubits),
                "depth": 0,
                "state_label": state_label,
                "sector": "readout",
                "popcount": state["popcount"],
                "initial": initial,
                "rep": 0,
                "shots": READOUT_SHOTS,
                "t_step": 0.0,
            }
            qc = build_readout_baseline_circuit(4, initial)
            qc.name = f"p3_layout_{layout.layout_id}_readout_{state_label}"
            readout.append((meta, qc))
    return main, readout


def transpile_with_layouts(
    backend: Any,
    circuits: Sequence[tuple[dict[str, Any], QuantumCircuit]],
    *,
    optimization_level: int,
) -> list[QuantumCircuit]:
    """Transpile each circuit with its preregistered physical-qubit layout."""
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


def readiness(
    backend: Any,
    circuits: Sequence[tuple[dict[str, Any], QuantumCircuit]],
    isa_circuits: Sequence[QuantumCircuit],
    *,
    max_depth: int,
    max_total_gates: int,
) -> dict[str, Any]:
    """Evaluate layout-specific depth and gate-count readiness guards."""
    depths = [circuit.depth() for circuit in isa_circuits]
    total_gates = [sum(circuit.count_ops().values()) for circuit in isa_circuits]
    ecr_gates = [circuit.count_ops().get("ecr", 0) for circuit in isa_circuits]
    by_layout: dict[str, list[int]] = defaultdict(list)
    by_preregistered_depth: dict[int, list[int]] = defaultdict(list)
    for (meta, _), circuit in zip(circuits, isa_circuits):
        by_layout[str(meta["layout_id"])].append(circuit.depth())
        by_preregistered_depth[int(meta["depth"])].append(circuit.depth())
    accepted = max(depths) <= max_depth and max(total_gates) <= max_total_gates
    reason = None
    if max(depths) > max_depth:
        reason = f"max depth {max(depths)} exceeds guard {max_depth}"
    if max(total_gates) > max_total_gates:
        reason = f"max total gates {max(total_gates)} exceeds guard {max_total_gates}"
    return {
        "accepted": accepted,
        "backend_status": _backend_status(backend),
        "max_depth": max_depth,
        "max_total_gates": max_total_gates,
        "reference_popcount_control_max_depth": POPCOUNT_CONTROL_MAX_DEPTH,
        "depth_summary": _summary(depths),
        "total_gate_summary": _summary(total_gates),
        "ecr_gate_summary": _summary(ecr_gates),
        "per_layout_depth_max": {layout: max(vals) for layout, vals in sorted(by_layout.items())},
        "per_preregistered_depth_max": {
            str(depth): max(vals) for depth, vals in sorted(by_preregistered_depth.items())
        },
        "rejection_reason": reason,
    }


def _layout_payload(layouts: Sequence[LayoutCandidate]) -> list[dict[str, Any]]:
    return [
        {
            "layout_id": layout.layout_id,
            "physical_qubits": list(layout.physical_qubits),
            "readout_error_mean": layout.readout_error_mean,
            "two_qubit_error_mean": layout.two_qubit_error_mean,
            "score": layout.score,
        }
        for layout in layouts
    ]


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
        handle.write(f"\n## {timestamp} - PHASE 3 STATE/LAYOUT DLA\n\n")
        handle.write(f"- **Backend:** {payload['backend']}\n")
        handle.write(f"- **Status:** {payload['status']}\n")
        handle.write(f"- **Circuits:** {payload['n_circuits']}\n")
        handle.write(f"- **Layouts:** {len(payload['layouts'])}\n")
        handle.write(f"- **Job IDs:** {jobs or 'none'}\n")
        handle.write(f"- **Artefact:** `{path.relative_to(REPO_ROOT)}`\n")
        handle.write("- **Boundary:** state/layout mechanism-separation control only.\n")


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
                    "ecr_gates": circuit.count_ops().get("ecr", 0),
                },
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    """Parse Phase 3 state/layout DLA command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-budget", action="store_true")
    parser.add_argument("--max-depth", type=int, default=POPCOUNT_CONTROL_MAX_DEPTH)
    parser.add_argument("--max-total-gates", type=int, default=POPCOUNT_CONTROL_GATE_LIMIT)
    parser.add_argument("--timeout-main-s", type=int, default=5400)
    parser.add_argument("--timeout-readout-s", type=int, default=1800)
    parser.add_argument("--optimization-level", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    """Run state/layout readiness checks and optionally submit ISA circuits."""
    args = parse_args()
    if args.submit and not args.confirm_budget:
        print("ERROR: --submit requires --confirm-budget", file=sys.stderr)
        return 2

    timestamp = _timestamp()
    out_dir = REPO_ROOT / "data" / "phase3_state_layout_dla"
    out_path = out_dir / f"phase3_state_layout_{args.backend}_{timestamp}.json"

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

    layouts = select_layouts(runner.backend)
    main_circuits, readout_circuits = build_circuits(layouts)
    all_circuits = main_circuits + readout_circuits
    isa_main = transpile_with_layouts(
        runner.backend, main_circuits, optimization_level=args.optimization_level
    )
    isa_readout = transpile_with_layouts(
        runner.backend, readout_circuits, optimization_level=args.optimization_level
    )
    isa_all = isa_main + isa_readout
    ready = readiness(
        runner.backend,
        all_circuits,
        isa_all,
        max_depth=args.max_depth,
        max_total_gates=args.max_total_gates,
    )
    est_qpu_minutes = len(all_circuits) * 0.55 / 60.0
    payload: dict[str, Any] = {
        "schema": "scpn_phase3_state_layout_dla_v1",
        "status": "readiness_passed" if ready["accepted"] else "readiness_rejected",
        "timestamp_utc": timestamp,
        "backend": runner.backend_name,
        "experiment": EXPERIMENT,
        "layouts": _layout_payload(layouts),
        "n_circuits": len(all_circuits),
        "n_main_circuits": len(main_circuits),
        "n_readout_circuits": len(readout_circuits),
        "shots_main": MAIN_SHOTS,
        "shots_readout": READOUT_SHOTS,
        "depths": list(DEPTHS),
        "repetitions": REPS,
        "states": STATES,
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
    print(f"ECR summary: {ready['ecr_gate_summary']}")
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

    print("Submitting state/layout main batch...")
    main_job, main_counts, wall_main = _run_isa_sampler(
        runner.backend,
        isa_main,
        shots=MAIN_SHOTS,
        timeout_s=args.timeout_main_s,
    )
    payload["job_ids"].append(main_job)
    print("Submitting state/layout readout batch...")
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
