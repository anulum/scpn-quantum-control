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
from typing import Any, Callable, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..differentiable import GradientResult

try:
    import pennylane as _pennylane

    _PL_AVAILABLE = True
except Exception:
    _PL_AVAILABLE = False
    _pennylane = None

qml: Any = _pennylane

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]


@dataclass
class PennyLaneResult:
    """Result from PennyLane execution."""

    energy: float
    order_parameter: float
    statevector: ComplexArray | None
    device_name: str
    n_qubits: int


def is_pennylane_available() -> bool:
    """Check if PennyLane is installed."""
    return _PL_AVAILABLE


def _xy_hamiltonian_pl(K: FloatArray, omega: FloatArray) -> Any:
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


def _as_finite_matrix(name: str, values: ArrayLike) -> FloatArray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix.astype(np.float64, copy=True)


def _as_finite_vector(name: str, values: ArrayLike, *, width: int | None = None) -> FloatArray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if width is not None and vector.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.astype(np.float64, copy=True)


def _normalise_pennylane_device_name(device: str) -> str:
    normalised = str(device).strip()
    if not normalised:
        raise ValueError("PennyLane device name must not be empty")
    if any(ord(character) < 32 or ord(character) == 127 for character in normalised):
        raise ValueError("PennyLane device name must not contain control characters")
    return normalised


def _normalise_shots(shots: int | None) -> int | None:
    if shots is None:
        return None
    if isinstance(shots, bool) or not isinstance(shots, int):
        raise ValueError("shots must be a positive integer or None")
    if shots <= 0:
        raise ValueError("shots must be a positive integer or None")
    return shots


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
        K: ArrayLike,
        omega: ArrayLike,
        device: str = "default.qubit",
        shots: int | None = None,
        **device_kwargs: Any,
    ) -> None:
        if not _PL_AVAILABLE:
            raise ImportError("PennyLane not installed: pip install pennylane")

        self.K = _as_finite_matrix("K", K)
        if self.K.shape[0] != self.K.shape[1]:
            raise ValueError("K must be a square coupling matrix")
        self.omega = _as_finite_vector("omega", omega, width=self.K.shape[0])
        self.n = self.K.shape[0]
        self.H = _xy_hamiltonian_pl(self.K, self.omega)
        self.device_name = _normalise_pennylane_device_name(device)
        self.shots = _normalise_shots(shots)
        self.dev = qml.device(
            self.device_name,
            wires=self.n,
            shots=self.shots,
            **device_kwargs,
        )

    def _measure_order_parameter(self, prepare_state: Callable[[], None]) -> float:
        """Measure Kuramoto R from local transverse Bloch-vector phases."""
        phases = np.zeros(self.n)
        for i in range(self.n):

            def _measure_x(qubit: int = i) -> Any:
                prepare_state()
                return qml.expval(qml.PauliX(qubit))

            def _measure_y(qubit: int = i) -> Any:
                prepare_state()
                return qml.expval(qml.PauliY(qubit))

            measure_x = qml.qnode(self.dev)(_measure_x)
            measure_y = qml.qnode(self.dev)(_measure_y)
            ex = float(measure_x())
            ey = float(measure_y())
            phases[i] = np.arctan2(ey, ex)

        z = np.mean(np.exp(1j * phases))
        return float(np.clip(np.abs(z), 0.0, 1.0))

    def _apply_vqe_ansatz(self, params: Any, ansatz_depth: int) -> None:
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

        def _circuit() -> Any:
            for _r in range(reps):
                qml.ApproxTimeEvolution(H, dt, 1)
            return qml.expval(H)

        circuit = qml.qnode(self.dev)(_circuit)
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

        def _cost_fn(params: Any) -> Any:
            self._apply_vqe_ansatz(params, ansatz_depth)
            return qml.expval(H)

        cost_fn = qml.qnode(self.dev)(_cost_fn)
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

    def vqe_value_and_grad(
        self,
        params: ArrayLike,
        *,
        ansatz_depth: int = 2,
    ) -> GradientResult:
        """Return VQE energy and PennyLane autodiff gradient for ansatz parameters."""
        n_params = self.n * ansatz_depth * 3
        raw_params = _as_finite_vector("params", params, width=n_params)

        pl_np = getattr(qml, "numpy", np)
        try:
            diff_params = pl_np.array(raw_params, requires_grad=True)
        except TypeError:
            diff_params = pl_np.array(raw_params)

        def _cost_fn(current_params: Any) -> Any:
            self._apply_vqe_ansatz(current_params, ansatz_depth)
            return qml.expval(self.H)

        cost_fn = qml.qnode(self.dev)(_cost_fn)
        gradient_fn = qml.grad(cost_fn)
        value = float(cost_fn(diff_params))
        gradient = _as_finite_vector("PennyLane VQE gradient", gradient_fn(diff_params))
        return GradientResult(
            value=value,
            gradient=gradient,
            method="pennylane_autodiff",
            shift=None,
            coefficient=None,
            evaluations=2,
            parameter_names=tuple(f"vqe_{index}" for index in range(n_params)),
            trainable=tuple(True for _ in range(n_params)),
        )
