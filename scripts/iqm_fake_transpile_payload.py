#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — isolated IQM fake-transpile helper
"""Transpile QASM2 circuit payloads inside an IQM-specific Python environment."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile

FAKE_BACKENDS = {
    "adonis": ("iqm.qiskit_iqm.fake_backends.fake_adonis", "IQMFakeAdonis"),
    "deneb": ("iqm.qiskit_iqm.fake_backends.fake_deneb", "IQMFakeDeneb"),
    "apollo": ("iqm.qiskit_iqm.fake_backends.fake_apollo", "IQMFakeApollo"),
    "garnet": ("iqm.qiskit_iqm.fake_backends.fake_garnet", "IQMFakeGarnet"),
    "aphrodite": ("iqm.qiskit_iqm.fake_backends.fake_aphrodite", "IQMFakeAphrodite"),
}


def main() -> int:
    """Transpile each payload circuit against the requested IQM fake backend."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--fake-backend", default="garnet")
    parser.add_argument("--optimisation-level", type=int, default=1)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    module_name, class_name = FAKE_BACKENDS[args.fake_backend]
    backend = getattr(importlib.import_module(module_name), class_name)()
    rows = []
    for row in payload["circuits"]:
        circuit = _build_circuit(row)
        transpiled = transpile(
            circuit, backend=backend, optimization_level=args.optimisation_level
        )
        rows.append(
            {
                "circuit_name": row["circuit_name"],
                "iqm_fake_backend": args.fake_backend,
                "iqm_fake_status": "passed",
                "iqm_transpiled_depth": transpiled.depth(),
                "iqm_transpiled_size": transpiled.size(),
                "iqm_transpiled_ops": json.dumps(
                    {name: int(count) for name, count in transpiled.count_ops().items()},
                    sort_keys=True,
                ),
            }
        )
    print(json.dumps({"status": "passed", "rows": rows}, sort_keys=True))
    return 0


def _build_circuit(row: dict) -> QuantumCircuit:
    meta = row["meta"]
    name = row["circuit_name"]
    experiment = meta.get("experiment", "")
    protocol_arm = meta.get("protocol_arm", "")
    if experiment == "iqm_smoke_bell":
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])
    elif experiment == "A_dla_parity_n4":
        circuit = _build_xy_trotter_circuit(4, meta["initial"], int(meta["depth"]), meta["t_step"])
    elif experiment in {"C_readout_baseline", "fim_readout_full_basis"}:
        circuit = _build_readout_baseline_circuit(4, meta["initial"])
    elif protocol_arm == "fim_sector_survival_pilot":
        circuit = _build_fim_trotter_circuit(
            meta["initial_bitstring"],
            int(meta["depth"]),
            float(meta["lambda_fim"]),
        )
    else:
        raise ValueError(f"Unsupported IQM payload row: {meta}")
    circuit.name = name
    return circuit


def _prep_bitstring(circuit: QuantumCircuit, bitstring: str) -> None:
    for qubit, bit in enumerate(bitstring):
        if bit == "1":
            circuit.x(qubit)


def _kuramoto_k_matrix(n_qubits: int) -> np.ndarray:
    k_matrix = np.zeros((n_qubits, n_qubits), dtype=np.float64)
    for i in range(n_qubits):
        for j in range(n_qubits):
            if i != j:
                k_matrix[i, j] = 0.45 * np.exp(-0.3 * abs(i - j))
    return k_matrix


def _build_xy_trotter_circuit(
    n_qubits: int,
    initial_bitstring: str,
    depth: int,
    t_step: float,
) -> QuantumCircuit:
    circuit = QuantumCircuit(n_qubits, n_qubits)
    _prep_bitstring(circuit, initial_bitstring)
    k_matrix = _kuramoto_k_matrix(n_qubits)
    omega = np.linspace(0.8, 1.2, n_qubits)
    for _ in range(depth):
        for qubit in range(n_qubits):
            circuit.rz(2.0 * omega[qubit] * t_step, qubit)
        for i in range(n_qubits - 1):
            j = i + 1
            theta = 2.0 * k_matrix[i, j] * t_step
            circuit.rxx(theta, i, j)
            circuit.ryy(theta, i, j)
    circuit.measure(range(n_qubits), range(n_qubits))
    return circuit


def _build_readout_baseline_circuit(n_qubits: int, initial_bitstring: str) -> QuantumCircuit:
    circuit = QuantumCircuit(n_qubits, n_qubits)
    _prep_bitstring(circuit, initial_bitstring)
    circuit.measure(range(n_qubits), range(n_qubits))
    return circuit


def _build_fim_trotter_circuit(
    initial_bitstring: str,
    depth: int,
    lambda_fim: float,
    t_step: float = 0.3,
) -> QuantumCircuit:
    n_qubits = 4
    circuit = QuantumCircuit(n_qubits, n_qubits)
    _prep_bitstring(circuit, initial_bitstring)
    k_matrix = _kuramoto_k_matrix(n_qubits)
    omega = np.linspace(0.8, 1.2, n_qubits)
    fim_theta = -4.0 * lambda_fim * t_step / n_qubits
    for _ in range(depth):
        for qubit in range(n_qubits):
            circuit.rz(2.0 * omega[qubit] * t_step, qubit)
        for i in range(n_qubits - 1):
            j = i + 1
            theta = 2.0 * k_matrix[i, j] * t_step
            circuit.rxx(theta, i, j)
            circuit.ryy(theta, i, j)
        if abs(fim_theta) > 1e-15:
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    circuit.rzz(fim_theta, i, j)
    circuit.measure(range(n_qubits), range(n_qubits))
    return circuit


if __name__ == "__main__":
    raise SystemExit(main())
