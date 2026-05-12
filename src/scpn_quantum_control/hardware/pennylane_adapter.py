# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Pennylane Adapter
"""PennyLane backend adapter for cross-platform quantum execution.

Translates the Kuramoto-XY Hamiltonian and circuits into PennyLane's
framework, unlocking 20+ hardware backends:

    IBM (via pennylane-qiskit), IonQ, Rigetti, Quantinuum,
    Amazon Braket, Google Cirq, Xanadu (photonic), simulators.

Usage:
    from scpn_quantum_control.hardware.pennylane_adapter import PennyLaneRunner
    runner = PennyLaneRunner(K, omega, device="default.qubit")
    result = runner.run_trotter(t=1.0, reps=5)
    result = runner.run_vqe(maxiter=100)

For hardware:
    runner = PennyLaneRunner(K, omega, device="qiskit.ibmq", backend="ibm_fez")
    runner = PennyLaneRunner(K, omega, device="ionq.simulator")
    runner = PennyLaneRunner(K, omega, device="braket.aws.qubit")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

try:
    import pennylane as qml  # type: ignore[import-untyped,import-not-found]

    _PL_AVAILABLE = True
except Exception:
    _PL_AVAILABLE = False
    qml = None  # type: ignore[assignment]


@dataclass
class PennyLaneResult:
    """Result from PennyLane execution."""

    energy: float
    order_parameter: float
    statevector: np.ndarray | None
    device_name: str
    n_qubits: int


def is_pennylane_available() -> bool:
    """Check if PennyLane is installed."""
    return _PL_AVAILABLE


def _xy_hamiltonian_pl(K: np.ndarray, omega: np.ndarray) -> Any:
    """Build XY Hamiltonian as PennyLane Hamiltonian.

    H = -Σ_{ij} K_ij (X_i X_j + Y_i Y_j) - Σ_i (ω_i/2) Z_i
    """
    if not _PL_AVAILABLE:
        raise ImportError("PennyLane not installed: pip install pennylane")

    n = K.shape[0]
    coeffs: list[float] = []
    ops: list[Any] = []

    for i in range(n):
        for j in range(i + 1, n):
            if abs(K[i, j]) > 1e-15:
                coeffs.append(-float(K[i, j]))
                ops.append(qml.PauliX(i) @ qml.PauliX(j))
                coeffs.append(-float(K[i, j]))
                ops.append(qml.PauliY(i) @ qml.PauliY(j))

    for i in range(n):
        if abs(omega[i]) > 1e-15:
            coeffs.append(-float(omega[i]) / 2.0)
            ops.append(qml.PauliZ(i))

    return qml.Hamiltonian(coeffs, ops)


class PennyLaneRunner:
    """Cross-platform quantum runner via PennyLane.

    Supports any PennyLane-compatible device:
        "default.qubit" — statevector simulator
        "default.mixed" — density matrix simulator
        "lightning.qubit" — fast C++ simulator
        "qiskit.ibmq" — IBM Quantum (via pennylane-qiskit)
        "ionq.simulator" — IonQ (via pennylane-ionq)
        "braket.aws.qubit" — Amazon Braket
    """

    def __init__(
        self,
        K: np.ndarray,
        omega: np.ndarray,
        device: str = "default.qubit",
        shots: int | None = None,
        **device_kwargs: Any,
    ):
        if not _PL_AVAILABLE:
            raise ImportError("PennyLane not installed: pip install pennylane")

        self.K = K
        self.omega = omega
        self.n = K.shape[0]
        self.H = _xy_hamiltonian_pl(K, omega)
        self.device_name = device
        self.shots = shots
        self.dev = qml.device(device, wires=self.n, shots=shots, **device_kwargs)

    def _measure_order_parameter(self, prepare_state: Callable[[], None]) -> float:
        """Measure Kuramoto R from local transverse Bloch-vector phases."""
        phases = np.zeros(self.n)
        for i in range(self.n):

            @qml.qnode(self.dev)
            def measure_x(qubit=i):
                prepare_state()
                return qml.expval(qml.PauliX(qubit))

            @qml.qnode(self.dev)
            def measure_y(qubit=i):
                prepare_state()
                return qml.expval(qml.PauliY(qubit))

            ex = float(measure_x())
            ey = float(measure_y())
            phases[i] = np.arctan2(ey, ex)

        z = np.mean(np.exp(1j * phases))
        return float(np.clip(np.abs(z), 0.0, 1.0))

    def _apply_vqe_ansatz(self, params: np.ndarray, ansatz_depth: int) -> None:
        """Apply the hardware-efficient VQE ansatz used by this runner."""
        idx = 0
        for _layer in range(ansatz_depth):
            for q in range(self.n):
                qml.Rot(params[idx], params[idx + 1], params[idx + 2], wires=q)
                idx += 3
            for q in range(self.n - 1):
                qml.CNOT(wires=[q, q + 1])

    def run_trotter(
        self,
        t: float = 1.0,
        reps: int = 5,
    ) -> PennyLaneResult:
        """Run Trotter evolution and measure energy + order parameter."""
        n = self.n
        H = self.H
        dt = t / reps

        @qml.qnode(self.dev)
        def circuit():
            for _r in range(reps):
                qml.ApproxTimeEvolution(H, dt, 1)
            return qml.expval(H)

        energy = float(circuit())

        def prepare_state() -> None:
            for _r in range(reps):
                qml.ApproxTimeEvolution(H, dt, 1)

        r_global = self._measure_order_parameter(prepare_state)

        return PennyLaneResult(
            energy=energy,
            order_parameter=r_global,
            statevector=None,
            device_name=self.device_name,
            n_qubits=n,
        )

    def run_vqe(
        self,
        ansatz_depth: int = 2,
        maxiter: int = 100,
        seed: int | None = None,
    ) -> PennyLaneResult:
        """Run VQE with hardware-efficient ansatz."""
        n = self.n
        H = self.H
        rng = np.random.default_rng(seed)

        n_params = n * ansatz_depth * 3  # Rot(phi, theta, omega) per qubit per layer

        @qml.qnode(self.dev)
        def cost_fn(params):
            self._apply_vqe_ansatz(params, ansatz_depth)
            return qml.expval(H)

        opt = qml.GradientDescentOptimizer(stepsize=0.1)
        initial_params = rng.normal(0, 0.1, size=n_params)
        pl_np = getattr(qml, "numpy", np)
        try:
            params = pl_np.array(initial_params, requires_grad=True)
        except TypeError:
            params = pl_np.array(initial_params)

        for _step in range(maxiter):
            params = opt.step(cost_fn, params)

        energy = float(cost_fn(params))
        r_global = self._measure_order_parameter(
            lambda: self._apply_vqe_ansatz(params, ansatz_depth)
        )

        return PennyLaneResult(
            energy=energy,
            order_parameter=r_global,
            statevector=None,
            device_name=self.device_name,
            n_qubits=n,
        )
