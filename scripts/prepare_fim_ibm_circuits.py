#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- FIM IBM circuit preparation
"""Prepare and locally transpile non-submitting SCPN/FIM IBM pilot circuits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "scpn_fim_hamiltonian"
DATE = "2026-05-05"
T_STEP = 0.3
N_QUBITS = 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prep_bitstring(qc: QuantumCircuit, bitstring: str) -> None:
    for qubit, bit in enumerate(bitstring):
        if bit == "1":
            qc.x(qubit)


def _kuramoto_k_matrix(n_qubits: int) -> np.ndarray:
    k_matrix = np.zeros((n_qubits, n_qubits), dtype=np.float64)
    for i in range(n_qubits):
        for j in range(n_qubits):
            if i != j:
                k_matrix[i, j] = 0.45 * np.exp(-0.3 * abs(i - j))
    return k_matrix


def build_fim_trotter_circuit(
    initial_bitstring: str,
    depth: int,
    lambda_fim: float,
    t_step: float = T_STEP,
) -> QuantumCircuit:
    """Build an n=4 Kuramoto-XY + FIM Trotter pilot circuit.

    The FIM term is H_FIM = -lambda*M^2/n. Up to a global phase,
    M^2 = n I + 2 sum_{i<j} Z_i Z_j, so each Trotter step adds all-pair
    ZZ evolution with coefficient -2*lambda/n. Qiskit RZZ(theta) implements
    exp(-i theta ZZ / 2), hence theta = -4*lambda*t_step/n.
    """

    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    _prep_bitstring(qc, initial_bitstring)
    k_matrix = _kuramoto_k_matrix(N_QUBITS)
    omega = np.linspace(0.8, 1.2, N_QUBITS)
    fim_theta = -4.0 * float(lambda_fim) * t_step / float(N_QUBITS)

    for _ in range(depth):
        for qubit in range(N_QUBITS):
            qc.rz(2.0 * omega[qubit] * t_step, qubit)
        for i in range(N_QUBITS - 1):
            j = i + 1
            theta = 2.0 * k_matrix[i, j] * t_step
            qc.rxx(theta, i, j)
            qc.ryy(theta, i, j)
        if abs(fim_theta) > 1e-15:
            for i in range(N_QUBITS):
                for j in range(i + 1, N_QUBITS):
                    qc.rzz(fim_theta, i, j)

    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc


def build_readout_circuit(initial_bitstring: str) -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    _prep_bitstring(qc, initial_bitstring)
    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc


def _load_protocol(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _two_qubit_count(ops: dict[str, int]) -> int:
    return int(
        sum(
            count
            for gate, count in ops.items()
            if gate in {"cx", "cz", "ecr", "rxx", "ryy", "rzz"}
        )
    )


def generate(protocol_path: Path, optimisation_level: int) -> dict[str, object]:
    protocol = _load_protocol(protocol_path)
    rows: list[dict[str, object]] = []
    basis_gates = ["rz", "sx", "x", "cx", "measure"]
    for index, item in enumerate(protocol["rows"]):
        if item["protocol_arm"] == "readout_baseline":
            circuit = build_readout_circuit(str(item["initial_bitstring"]))
        else:
            circuit = build_fim_trotter_circuit(
                str(item["initial_bitstring"]),
                int(item["depth"]),
                float(item["lambda_fim"]),
            )
        circuit.name = f"fim_{index:03d}_{item['protocol_arm']}"
        transpiled = transpile(
            circuit,
            basis_gates=basis_gates,
            optimization_level=optimisation_level,
            seed_transpiler=20260505,
        )
        raw_ops = {gate: int(count) for gate, count in circuit.count_ops().items()}
        transpiled_ops = {gate: int(count) for gate, count in transpiled.count_ops().items()}
        rows.append(
            {
                **item,
                "circuit_index": index,
                "circuit_name": circuit.name,
                "raw_depth": circuit.depth(),
                "raw_size": circuit.size(),
                "raw_two_qubit_gates": _two_qubit_count(raw_ops),
                "transpiled_depth": transpiled.depth(),
                "transpiled_size": transpiled.size(),
                "transpiled_two_qubit_gates": _two_qubit_count(transpiled_ops),
                "raw_ops": json.dumps(raw_ops, sort_keys=True),
                "transpiled_ops": json.dumps(transpiled_ops, sort_keys=True),
                "submission_status": "not_submitted",
            }
        )
    total_shots = int(sum(int(row["shots"]) for row in rows))
    return {
        "schema": "scpn_fim_ibm_circuit_preparation_v1",
        "date": DATE,
        "command": "python scripts/prepare_fim_ibm_circuits.py",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "protocol_path": str(protocol_path.relative_to(REPO_ROOT)),
        "basis_gates": basis_gates,
        "optimisation_level": optimisation_level,
        "submission_status": "not_submitted",
        "requires_live_backend_transpile_before_qpu": True,
        "requires_user_approval_before_qpu": True,
        "total_circuits": len(rows),
        "total_shots": total_shots,
        "max_transpiled_depth": max(int(row["transpiled_depth"]) for row in rows),
        "max_transpiled_two_qubit_gates": max(
            int(row["transpiled_two_qubit_gates"]) for row in rows
        ),
        "scientific_boundary": (
            "Local basis-gate transpilation only. This is not a backend-calibrated "
            "Heron ISA transpile and not a QPU submission. Live backend selection, "
            "layout, calibration metadata, and QPU-time estimate remain required."
        ),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=OUT_DIR / f"fim_ibm_candidate_protocol_{DATE}.json",
    )
    parser.add_argument("--optimisation-level", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    ns = parser.parse_args()

    ns.output_dir.mkdir(parents=True, exist_ok=True)
    summary = generate(ns.protocol, ns.optimisation_level)
    json_path = ns.output_dir / f"fim_ibm_circuit_preparation_{DATE}.json"
    csv_path = ns.output_dir / f"fim_ibm_circuit_preparation_{DATE}.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(csv_path, list(summary["rows"]))
    print(f"wrote_json={json_path}")
    print(f"wrote_csv={csv_path}")
    print(f"sha256_json={_sha256(json_path)}")
    print(f"sha256_csv={_sha256(csv_path)}")
    print(f"total_circuits={summary['total_circuits']}")
    print(f"total_shots={summary['total_shots']}")
    print(f"max_transpiled_depth={summary['max_transpiled_depth']}")
    print(f"max_transpiled_two_qubit_gates={summary['max_transpiled_two_qubit_gates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
