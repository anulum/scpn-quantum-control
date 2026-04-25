# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts & Code 2020–2026 Miroslav Šotek. All rights reserved.

from __future__ import annotations

from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


class StructuredAnsatz:
    """
    Physically-informed ansatz for heterogeneous Kuramoto-XY model.
    Builds Trotterized circuits from coupling matrix K_nm and frequencies omega.
    """

    def __init__(self):
        self.circuit: QuantumCircuit | None = None
        self.params: dict[str, Any] = {}

    @staticmethod
    def from_kuramoto(
        K_nm: np.ndarray,
        omega: np.ndarray | None = None,
        trotter_depth: int = 6,
        time_step: float = 0.1,
        informed_topology: bool = True,
        non_hermitian_gain: float = 0.0,
        mediated_couplings: bool = False,
        lambda_fim: float = 0.0,
        **kwargs: Any,
    ) -> StructuredAnsatz:
        """
        Creates a StructuredAnsatz from Kuramoto coupling matrix and frequencies.

        Args:
            K_nm: (N x N) symmetric coupling matrix
            omega: (N,) natural frequencies (optional)
            trotter_depth: Number of Trotter steps
            time_step: dt per Trotter step
            informed_topology: Use K_nm topology for entangling gates
            non_hermitian_gain: For PT-symmetric extensions
            mediated_couplings: For distributed/multi-node experiments
            lambda_fim: FIM feedback strength (strange loop)
        """
        N = K_nm.shape[0]
        ansatz = StructuredAnsatz()
        ansatz.params = {
            "N": N,
            "trotter_depth": trotter_depth,
            "time_step": time_step,
            "lambda_fim": lambda_fim,
        }

        qc = QuantumCircuit(N)

        # Initial state: all qubits in |+> (uniform phase)
        qc.h(range(N))

        dt = time_step
        for _ in range(trotter_depth):
            # Single-qubit Z rotations from natural frequencies
            if omega is not None:
                for i in range(N):
                    qc.rz(2 * omega[i] * dt, i)

            # Two-qubit XY interactions from K_nm
            for i in range(N):
                for j in range(i + 1, N):
                    if abs(K_nm[i, j]) > 1e-8:
                        # XY interaction via RZZ + single-qubit rotations (standard decomposition)
                        theta = 2 * K_nm[i, j] * dt
                        qc.rzz(theta, i, j)

            # Optional FIM feedback term (strange loop)
            if lambda_fim > 0:
                # Global phase feedback approximation
                global_phase = Parameter("lambda_fim")
                for i in range(N):
                    qc.rz(global_phase * dt, i)

        ansatz.circuit = qc
        return ansatz

    def build_circuit(self) -> QuantumCircuit:
        """Returns the built Qiskit circuit for submission."""
        if self.circuit is None:
            raise ValueError("Ansatz not initialized. Call from_kuramoto first.")
        return self.circuit.copy()

    def __repr__(self) -> str:
        return f"StructuredAnsatz(N={self.params.get('N')}, trotter_depth={self.params.get('trotter_depth')})"
