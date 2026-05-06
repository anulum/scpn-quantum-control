#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S3 ansatz observable validation
"""Validate promoted S3 ansatz candidates against no-QPU observables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

from scpn_quantum_control.benchmarks.s3_design_protocol import (
    S3DesignCandidate,
    grid_s3_design_protocol,
    score_s3_candidates,
)
from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    knm_to_dense_matrix,
    knm_to_hamiltonian,
)
from scpn_quantum_control.control.structured_ansatz import StructuredAnsatz

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "s3_pulse_ansatz_design"
DATE = "2026-05-06"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="3,4", help="Comma-separated qubit sizes.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of lowest-resource ansatz candidates per size.",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _promoted_ansatz_candidates(n_qubits: int, top_k: int) -> list[S3DesignCandidate]:
    protocol = grid_s3_design_protocol()
    k_matrix = build_knm_paper27(n_qubits)
    omega = np.asarray(OMEGA_N_16[:n_qubits], dtype=np.float64)
    scored = score_s3_candidates(protocol, k_matrix, omega)
    labels = [
        row.candidate_label
        for row in sorted(
            (row for row in scored if row.family == "ansatz"),
            key=lambda row: row.score,
        )[:top_k]
    ]
    candidates = {candidate.label: candidate for candidate in protocol.candidates}
    return [candidates[label] for label in labels]


def _pauli_expectation(state: Statevector, label: str) -> float:
    value = state.expectation_value(SparsePauliOp.from_list([(label, 1.0)]))
    return float(np.real(value))


def _sync_proxy(state: Statevector, n_qubits: int) -> float:
    x_sum = 0.0
    y_sum = 0.0
    for qubit in range(n_qubits):
        x_label = ["I"] * n_qubits
        y_label = ["I"] * n_qubits
        x_label[n_qubits - 1 - qubit] = "X"
        y_label[n_qubits - 1 - qubit] = "Y"
        x_sum += _pauli_expectation(state, "".join(x_label))
        y_sum += _pauli_expectation(state, "".join(y_label))
    return float(np.hypot(x_sum, y_sum) / n_qubits)


def _validate_candidate(candidate: S3DesignCandidate, n_qubits: int) -> dict[str, Any]:
    k_matrix = build_knm_paper27(n_qubits)
    omega = np.asarray(OMEGA_N_16[:n_qubits], dtype=np.float64)
    params = candidate.parameters
    ansatz = StructuredAnsatz.from_kuramoto(
        k_matrix,
        omega=omega,
        trotter_depth=int(params["trotter_depth"]),
        time_step=float(params["time_step"]),
        coupling_scale=float(params["coupling_scale"]),
    )
    circuit = ansatz.build_circuit()
    state = Statevector.from_instruction(circuit)
    hamiltonian = knm_to_hamiltonian(k_matrix, omega)
    energy = float(np.real(state.expectation_value(hamiltonian)))
    exact_ground = float(np.linalg.eigvalsh(knm_to_dense_matrix(k_matrix, omega))[0])
    energy_error = abs(energy - exact_ground)
    return {
        "candidate_label": candidate.label,
        "n_qubits": n_qubits,
        "status": "ok",
        "parameters": dict(candidate.parameters),
        "depth": int(circuit.depth()),
        "size": int(circuit.size()),
        "two_qubit_gates": int(circuit.count_ops().get("rzz", 0)),
        "energy": energy,
        "exact_ground_energy": exact_ground,
        "energy_absolute_error": energy_error,
        "relative_energy_error_pct": energy_error / abs(exact_ground) * 100.0
        if abs(exact_ground) > 1e-15
        else None,
        "sync_proxy": _sync_proxy(state, n_qubits),
        "hardware_submission": False,
    }


def _rows(sizes: list[int], top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n_qubits in sizes:
        if n_qubits < 2 or n_qubits > len(OMEGA_N_16):
            raise ValueError("sizes must be between 2 and len(OMEGA_N_16)")
        for candidate in _promoted_ansatz_candidates(n_qubits, top_k):
            rows.append(_validate_candidate(candidate, n_qubits))
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = np.asarray([float(row["energy_absolute_error"]) for row in rows], dtype=np.float64)
    sync = np.asarray([float(row["sync_proxy"]) for row in rows], dtype=np.float64)
    best = min(rows, key=lambda row: float(row["energy_absolute_error"]))
    return {
        "row_count": len(rows),
        "mean_energy_absolute_error": float(np.mean(errors)),
        "best_energy_absolute_error": float(np.min(errors)),
        "mean_sync_proxy": float(np.mean(sync)),
        "best_candidate_by_energy": best["candidate_label"],
        "best_candidate_n_qubits": best["n_qubits"],
    }


def _markdown(summary: dict[str, Any]) -> str:
    aggregate = summary["aggregate"]
    if not isinstance(aggregate, dict):
        raise TypeError("aggregate must be a mapping")
    lines = [
        "# S3 Ansatz Observable Validation",
        "",
        "Submission state: no hardware submission; exact statevector observable validation only.",
        "",
        "## Aggregate",
        f"- row count: {aggregate['row_count']}",
        f"- mean energy absolute error: {aggregate['mean_energy_absolute_error']}",
        f"- best energy absolute error: {aggregate['best_energy_absolute_error']}",
        f"- mean synchronisation proxy: {aggregate['mean_sync_proxy']}",
        f"- best candidate: `{aggregate['best_candidate_by_energy']}` at n={aggregate['best_candidate_n_qubits']}",
        "",
        "## Claim Boundary",
        summary["claim_boundary"],
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    sizes = [int(item.strip()) for item in args.sizes.split(",") if item.strip()]
    rows = _rows(sizes, args.top_k)
    summary = {
        "date": DATE,
        "script": "scripts/validate_s3_ansatz_observables.py",
        "hardware_submission": False,
        "observable_validation_performed": True,
        "sizes": sizes,
        "top_k": args.top_k,
        "rows": rows,
        "aggregate": _aggregate(rows),
        "claim_boundary": (
            "This validates concrete ansatz candidates against exact no-QPU observables. "
            "It is not VQE optimisation, not pulse-level validation, not a hardware result, "
            "and not a quantum-advantage claim."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"s3_ansatz_observable_validation_{DATE}.json"
    md_path = args.out_dir / f"s3_ansatz_observable_validation_{DATE}.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(summary), encoding="utf-8")
    print(f"wrote_json={json_path}")
    print(f"wrote_markdown={md_path}")
    print(f"sha256_json={_sha256(json_path)}")
    print(f"sha256_markdown={_sha256(md_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
