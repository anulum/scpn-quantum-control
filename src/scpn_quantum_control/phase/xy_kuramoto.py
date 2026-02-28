"""Quantum Kuramoto solver via XY spin Hamiltonian + Trotter evolution.

The Kuramoto model d(theta_i)/dt = omega_i + K*sum_j sin(theta_j - theta_i)
is isomorphic to the XY spin Hamiltonian:
    H = -sum_{i<j} K_ij (X_iX_j + Y_iY_j) - sum_i omega_i Z_i

Quantum hardware simulates this natively via Trotterized time evolution.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import LieTrotter, SuzukiTrotter

from ..bridge.knm_hamiltonian import knm_to_hamiltonian


class QuantumKuramotoSolver:
    """Trotterized quantum simulation of Kuramoto oscillators.

    Each oscillator maps to one qubit. The XY coupling simulates
    the sin(theta_j - theta_i) interaction natively.
    """

    def __init__(
        self,
        n_oscillators: int,
        K_coupling: np.ndarray,
        omega_natural: np.ndarray,
        trotter_order: int = 1,
    ):
        self.n = n_oscillators
        self.K = np.asarray(K_coupling, dtype=np.float64)
        self.omega = np.asarray(omega_natural, dtype=np.float64)
        self.trotter_order = trotter_order
        self._hamiltonian: SparsePauliOp | None = None

    def build_hamiltonian(self) -> SparsePauliOp:
        self._hamiltonian = knm_to_hamiltonian(self.K, self.omega)
        return self._hamiltonian

    def evolve(self, time: float, trotter_steps: int = 10) -> QuantumCircuit:
        """Build Trotterized evolution circuit U(t) = exp(-iHt).

        Uses LieTrotter (order=1, O(t²/reps)) or SuzukiTrotter (order=2,
        O(t³/reps²)) depending on self.trotter_order.
        """
        if self._hamiltonian is None:
            self.build_hamiltonian()
        if self.trotter_order == 2:
            synth = SuzukiTrotter(order=2, reps=trotter_steps)
        else:
            synth = LieTrotter(reps=trotter_steps)
        evo_gate = PauliEvolutionGate(self._hamiltonian, time=time, synthesis=synth)
        qc = QuantumCircuit(self.n)
        qc.append(evo_gate, range(self.n))
        return qc

    def measure_order_parameter(self, sv: Statevector) -> tuple[float, float]:
        """Compute Kuramoto R from qubit X,Y expectations.

        R*exp(i*psi) = (1/N) sum_j (<X_j> + i<Y_j>)
        """
        z_complex = 0.0 + 0.0j
        for j in range(self.n):
            exp_x = float(sv.expectation_value(self._pauli_op("X", j)).real)
            exp_y = float(sv.expectation_value(self._pauli_op("Y", j)).real)
            z_complex += exp_x + 1j * exp_y
        z_complex /= self.n
        R = float(abs(z_complex))
        psi = float(np.angle(z_complex))
        return R, psi

    def run(self, t_max: float, dt: float, trotter_per_step: int = 5) -> dict:
        """Time-stepped evolution returning R(t) and per-qubit expectations."""
        if self._hamiltonian is None:
            self.build_hamiltonian()

        n_steps = max(1, int(t_max / dt))
        R_history = np.zeros(n_steps + 1)
        times = np.linspace(0, t_max, n_steps + 1)

        # Initial state: each qubit at angle ~ omega_i (Ry rotation)
        init_qc = QuantumCircuit(self.n)
        for i in range(self.n):
            angle = float(self.omega[i]) % (2 * np.pi)
            init_qc.ry(angle, i)

        sv = Statevector.from_instruction(init_qc)
        R_history[0], _ = self.measure_order_parameter(sv)

        for step in range(1, n_steps + 1):
            evo_qc = self.evolve(dt, trotter_per_step)
            sv = sv.evolve(evo_qc)
            R_history[step], _ = self.measure_order_parameter(sv)

        return {"times": times, "R": R_history}

    def energy_expectation(self, sv: Statevector) -> float:
        """Compute <H> for a given statevector."""
        if self._hamiltonian is None:
            self.build_hamiltonian()
        return float(sv.expectation_value(self._hamiltonian).real)

    def _pauli_op(self, pauli: str, qubit: int) -> SparsePauliOp:
        label = ["I"] * self.n
        label[qubit] = pauli
        return SparsePauliOp("".join(reversed(label)))
