"""Quantum 16-layer UPDE solver: multi-site spin chain.

The 16-layer SCPN UPDE with Knm coupling becomes a 16-qubit system
where each qubit encodes one layer's phase. Inter-layer coupling K[n,m]
maps to XY interaction strength; natural frequencies Omega_n map to Z fields.
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

from ..bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from .xy_kuramoto import QuantumKuramotoSolver


class QuantumUPDESolver:
    """Full 16-layer UPDE as quantum spin chain.

    Wraps QuantumKuramotoSolver with canonical SCPN parameters.
    """

    def __init__(
        self,
        K: np.ndarray | None = None,
        omega: np.ndarray | None = None,
    ):
        if K is None:
            K = build_knm_paper27()
        if omega is None:
            omega = OMEGA_N_16.copy()

        self.K = K
        self.omega = omega
        self.n_layers = len(omega)
        self._solver = QuantumKuramotoSolver(self.n_layers, K, omega)
        self._solver.build_hamiltonian()

    def step(self, dt: float = 0.1, trotter_steps: int = 5) -> dict:
        """Single Trotter step, return per-layer expectations and global R."""
        if not hasattr(self, "_sv"):
            from qiskit import QuantumCircuit

            qc = QuantumCircuit(self.n_layers)
            for i in range(self.n_layers):
                qc.ry(float(self.omega[i]) % (2 * np.pi), i)
            self._sv = Statevector.from_instruction(qc)

        evo_qc = self._solver.evolve(dt, trotter_steps)
        self._sv = self._sv.evolve(evo_qc)
        R, psi = self._solver.measure_order_parameter(self._sv)

        return {"R_global": R, "psi": psi, "dt": dt}

    def run(self, n_steps: int = 50, dt: float = 0.1, trotter_per_step: int = 5) -> dict:
        """Full trajectory returning R(t) over n_steps."""
        result = self._solver.run(n_steps * dt, dt, trotter_per_step)
        return {
            "times": result["times"],
            "R": result["R"],
            "n_layers": self.n_layers,
        }

    def reset(self) -> None:
        """Reset statevector so the next step() reinitialises from omega."""
        if hasattr(self, "_sv"):
            del self._sv

    def hamiltonian(self) -> SparsePauliOp | None:
        return self._solver._hamiltonian
