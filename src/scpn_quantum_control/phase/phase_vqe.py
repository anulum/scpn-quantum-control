"""VQE for Kuramoto/XY Hamiltonian ground state.

Finds the maximum synchronization configuration of the coupled oscillator
system using a variational quantum eigensolver with physics-informed ansatz
where entanglement topology matches Knm sparsity.
"""
from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

from ..bridge.knm_hamiltonian import knm_to_hamiltonian, knm_to_ansatz


class PhaseVQE:
    """VQE solver for the XY Kuramoto Hamiltonian ground state.

    The ground state corresponds to maximum phase synchronization
    (minimum energy = strongest coupling alignment).
    """

    def __init__(
        self,
        K: np.ndarray,
        omega: np.ndarray,
        ansatz_reps: int = 2,
        threshold: float = 0.01,
    ):
        self.K = K
        self.omega = omega
        self.hamiltonian = knm_to_hamiltonian(K, omega)
        self.ansatz = knm_to_ansatz(K, reps=ansatz_reps, threshold=threshold)
        self.n_params = self.ansatz.num_parameters
        self._optimal_params: np.ndarray | None = None
        self._ground_energy: float | None = None

    def _cost(self, params: np.ndarray) -> float:
        bound = self.ansatz.assign_parameters(params)
        sv = Statevector.from_instruction(bound)
        return float(sv.expectation_value(self.hamiltonian).real)

    def solve(self, optimizer: str = "COBYLA", maxiter: int = 200) -> dict:
        """Run VQE optimization.

        Returns dict with ground_energy, optimal_params, n_evals.
        """
        x0 = np.random.default_rng().uniform(-np.pi, np.pi, self.n_params)

        result = minimize(
            self._cost,
            x0,
            method=optimizer,
            options={"maxiter": maxiter},
        )

        self._optimal_params = result.x
        self._ground_energy = float(result.fun)

        return {
            "ground_energy": self._ground_energy,
            "optimal_params": self._optimal_params,
            "n_evals": result.nfev,
            "converged": result.success,
        }

    def ground_state(self) -> Statevector | None:
        """Return the optimized ground state vector (call solve first)."""
        if self._optimal_params is None:
            return None
        bound = self.ansatz.assign_parameters(self._optimal_params)
        return Statevector.from_instruction(bound)
