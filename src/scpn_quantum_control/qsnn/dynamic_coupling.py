# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Dynamic Quantum-Classical Co-Evolution (Quantum Hebbian Learning).

Traditional quantum simulation takes a static physical coupling matrix K_nm
and computes the resulting quantum state. However, in biological systems
(like the brain or the SCPN framework), the coupling matrix is dynamic:
synapses rewire based on the coherence of the states they connect.

This module implements a 'Strange Loop' where:
1. Classical K_nm defines a Quantum XY Hamiltonian.
2. The quantum system evolves (e.g., via fast sparse evolution).
3. We measure the quantum correlation matrix C_nm = ⟨X_n X_m + Y_n Y_m⟩.
4. The classical coupling K_nm is updated via a Hebbian rule:
   K_nm(t+1) = (1 - decay) * K_nm(t) + learning_rate * C_nm
5. Repeat.

This creates a system whose macroscopic topology evolves according to its
microscopic quantum entanglement—a fundamentally novel simulation paradigm
for quantum biology and neuromorphic computing.
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

from scpn_quantum_control.hardware.fast_classical import fast_sparse_evolution


class DynamicCouplingEngine:
    """Engine for Quantum-Classical Co-evolution."""

    def __init__(
        self,
        n_qubits: int,
        initial_K: np.ndarray,
        omega: np.ndarray,
        learning_rate: float = 0.1,
        decay_rate: float = 0.05,
    ):
        self.n = n_qubits
        self.K = np.array(initial_K, dtype=float)
        self.omega = np.array(omega, dtype=float)
        self.lr = learning_rate
        self.decay = decay_rate

        # Enforce symmetry on initial K
        self.K = (self.K + self.K.T) / 2.0
        np.fill_diagonal(self.K, 0.0)

    def _measure_correlation_matrix(self, statevector: np.ndarray) -> np.ndarray:
        """Measure the XY correlation matrix from the quantum state."""
        sv = Statevector(statevector)
        C = np.zeros((self.n, self.n))

        for i in range(self.n):
            for j in range(i + 1, self.n):
                # XX
                x_str = ["I"] * self.n
                x_str[i] = "X"
                x_str[j] = "X"
                xx = sv.expectation_value(SparsePauliOp("".join(reversed(x_str)))).real

                # YY
                y_str = ["I"] * self.n
                y_str[i] = "Y"
                y_str[j] = "Y"
                yy = sv.expectation_value(SparsePauliOp("".join(reversed(y_str)))).real

                # Total XY correlation
                corr = xx + yy
                C[i, j] = corr
                C[j, i] = corr

        return C

    def step(self, dt: float) -> dict:
        """Perform one cycle of co-evolution.

        Evolve state for time dt -> Measure Correlators -> Update K.
        """
        # 1. Quantum Evolution (using the high-performance sparse engine)
        res = fast_sparse_evolution(self.K, self.omega, t_total=dt, n_steps=1)
        psi_final = res["final_state"]

        # 2. Measurement
        C_nm = self._measure_correlation_matrix(psi_final)

        # 3. Classical Hebbian Update
        # Increase coupling where quantum correlation is high; decay everywhere
        self.K = (1.0 - self.decay) * self.K + self.lr * C_nm

        # Enforce physical constraints (symmetry, no self-loops, non-negative)
        self.K = (self.K + self.K.T) / 2.0
        np.fill_diagonal(self.K, 0.0)
        self.K = np.maximum(self.K, 0.0)

        return {"K_updated": self.K.copy(), "correlation_matrix": C_nm, "statevector": psi_final}

    def run_coevolution(self, steps: int, dt: float) -> list[dict]:
        """Run the strange loop for a given number of steps."""
        history = []
        for _ in range(steps):
            history.append(self.step(dt))
        return history
