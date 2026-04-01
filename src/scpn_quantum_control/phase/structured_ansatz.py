# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""General-purpose structured VQE ansatz based on physical coupling matrices.

The Hamiltonian-structured ansatz places entangling gates (CZ, CNOT, or parameterized
two-qubit gates) exclusively across qubit pairs that have non-zero interaction terms
in the physical Hamiltonian.

This module abstracts the K_nm-informed ansatz technique (initially developed
for Kuramoto networks) into a generalized tool for any structured matrix,
such as molecular interaction graphs, power grids, or neural connectomes.
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit


def build_structured_ansatz(
    coupling_matrix: np.ndarray,
    reps: int = 2,
    entanglement_gate: str = "cz",
    threshold: float = 1e-6,
) -> QuantumCircuit:
    """Construct a topology-informed variational quantum circuit.

    Places single-qubit Ry and Rz rotations on all qubits, followed by
    two-qubit entangling gates only between qubits with |coupling_matrix[i,j]| >= threshold.

    Args:
        coupling_matrix: An n x n symmetric matrix defining the interaction graph.
        reps: Number of ansatz layers.
        entanglement_gate: The two-qubit gate to use ("cz" or "cx").
        threshold: Minimum absolute coupling strength to warrant an entangling gate.

    Returns:
        QuantumCircuit: The parameterized ansatz circuit.
    """
    n = coupling_matrix.shape[0]
    if coupling_matrix.shape[1] != n:
        raise ValueError("coupling_matrix must be square.")

    # Enforce symmetry just in case
    K = (coupling_matrix + coupling_matrix.T) / 2.0

    params = ParameterVector("θ", n * 2 * reps)
    qc = QuantumCircuit(n)

    idx = 0
    for _ in range(reps):
        # Single-qubit rotations
        for q in range(n):
            qc.ry(params[idx], q)
            idx += 1
        for q in range(n):
            qc.rz(params[idx], q)
            idx += 1

        # Two-qubit entangling layer
        for i in range(n):
            for j in range(i + 1, n):
                if abs(K[i, j]) >= threshold:
                    if entanglement_gate.lower() == "cz":
                        qc.cz(i, j)
                    elif entanglement_gate.lower() == "cx" or entanglement_gate.lower() == "cnot":
                        qc.cx(i, j)
                    else:
                        raise ValueError(f"Unsupported entanglement gate: {entanglement_gate}")

    return qc
