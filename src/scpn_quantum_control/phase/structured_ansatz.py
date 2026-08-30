# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Structured Ansatz
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
from numpy.typing import ArrayLike
from qiskit.circuit import ParameterVector, QuantumCircuit


def build_structured_ansatz(
    coupling_matrix: ArrayLike,
    reps: int = 2,
    entanglement_gate: str = "cz",
    threshold: float = 1e-6,
) -> QuantumCircuit:
    """Construct a topology-informed variational quantum circuit.

    Place single-qubit Ry and Rz rotations on all qubits, followed by
    two-qubit gates only where the symmetrised coupling magnitude meets the
    requested threshold.

    Parameters
    ----------
    coupling_matrix
        Square, non-empty, finite matrix defining the interaction graph.
    reps
        Positive number of ansatz layers.
    entanglement_gate
        Two-qubit gate to use: ``"cz"``, ``"cx"``, or the ``"cnot"`` alias.
    threshold
        Non-negative minimum coupling magnitude for an entangling gate.

    Returns
    -------
    QuantumCircuit
        Parameterised ansatz circuit.

    Raises
    ------
    ValueError
        If an input violates the matrix, layer, threshold, or gate contract.

    """
    matrix = np.asarray(coupling_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("coupling_matrix must be a two-dimensional square matrix")
    n = matrix.shape[0]
    if n == 0:
        raise ValueError("coupling_matrix must contain at least one qubit")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("coupling_matrix must contain only finite values")
    if isinstance(reps, bool) or not isinstance(reps, int) or reps <= 0:
        raise ValueError("reps must be a positive integer")
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative")
    gate = entanglement_gate.lower()
    if gate not in {"cz", "cx", "cnot"}:
        raise ValueError(f"Unsupported entanglement gate: {entanglement_gate}")

    K = (matrix + matrix.T) / 2.0

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
                    if gate == "cz":
                        qc.cz(i, j)
                    else:
                        qc.cx(i, j)

    return qc
