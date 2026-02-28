"""VQLS for Grad-Shafranov equilibrium.

Discretized Laplacian del^2(Psi) = source on N grid points -> sparse linear
system Ax=b.  VQLS finds |x> such that A|x> ~ |b> using variational cost
C = 1 - |<b|A|x>|^2 / <x|A^dag A|x>.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize


class VQLS_GradShafranov:
    """Variational Quantum Linear Solver for 1D Grad-Shafranov.

    Grid: N = 2^n_qubits points, uniform spacing on [0, 1].
    Source: peaked Gaussian J(x) = exp(-(x - 0.5)^2 / 0.05).
    """

    def __init__(self, n_qubits: int = 4, source_width: float = 0.05):
        self.n_qubits = n_qubits
        self.grid_size = 2**n_qubits
        self.source_width = source_width
        self._A: np.ndarray | None = None
        self._b: np.ndarray | None = None
        self._optimal_params: np.ndarray | None = None

    def discretize(self) -> tuple[np.ndarray, np.ndarray]:
        """Build finite-difference Laplacian and source vector."""
        N = self.grid_size
        dx = 1.0 / (N + 1)
        x = np.linspace(dx, 1.0 - dx, N)

        # 1D Laplacian: [-1, 2, -1] / dx^2
        A = np.zeros((N, N))
        for i in range(N):
            A[i, i] = 2.0
            if i > 0:
                A[i, i - 1] = -1.0
            if i < N - 1:
                A[i, i + 1] = -1.0
        A /= dx**2

        # Gaussian source profile
        b = np.exp(-((x - 0.5) ** 2) / self.source_width)
        b /= np.linalg.norm(b)

        self._A = A
        self._b = b
        return A, b

    def build_ansatz(self, reps: int = 2) -> QuantumCircuit:
        return TwoLocal(
            self.n_qubits,
            rotation_blocks=["ry"],
            entanglement_blocks="cz",
            reps=reps,
            entanglement="linear",
        )

    def solve(self, reps: int = 2, maxiter: int = 200) -> np.ndarray:
        """VQLS optimization -> Psi profile on grid.

        Cost: C = 1 - |<b|A|x>|^2 / <x|A^dag A|x>
        """
        if self._A is None:
            self.discretize()

        ansatz = self.build_ansatz(reps)
        A = self._A
        b = self._b
        AtA = A.T @ A

        def cost_fn(params):
            bound = ansatz.assign_parameters(params)
            sv = Statevector.from_instruction(bound)
            x_vec = np.array(sv)

            Ax = A @ x_vec
            bAx = np.vdot(b, Ax)
            xAtAx = np.vdot(x_vec, AtA @ x_vec).real

            if xAtAx < 1e-15:
                return 1.0
            return 1.0 - abs(bAx) ** 2 / xAtAx

        n_params = ansatz.num_parameters
        x0 = np.random.default_rng().uniform(-np.pi, np.pi, n_params)
        result = minimize(cost_fn, x0, method="COBYLA", options={"maxiter": maxiter})
        self._optimal_params = result.x

        bound = ansatz.assign_parameters(result.x)
        sv = Statevector.from_instruction(bound)
        psi = np.real(np.array(sv))

        # Normalize to physical scale
        A_psi = A @ psi
        scale = np.vdot(b, b) / np.vdot(b, A_psi) if abs(np.vdot(b, A_psi)) > 1e-15 else 1.0
        return psi * float(np.real(scale))
