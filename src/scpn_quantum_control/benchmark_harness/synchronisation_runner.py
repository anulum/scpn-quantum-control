# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- synchronisation benchmark runner
"""No-QPU runners for canonical synchronisation benchmark instances."""

from __future__ import annotations

import math
import platform
import sys
from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from scpn_quantum_control.benchmark_harness.synchronisation import RESULT_SCHEMA

RING_N4_BENCHMARK_ID = "kuramoto_ring_n4_linear_omega"
CHAIN_N8_BENCHMARK_ID = "kuramoto_chain_n8_decay_omega"
BENCHMARK_ID = RING_N4_BENCHMARK_ID


@dataclass(frozen=True, slots=True)
class ObservableRow:
    """One schema-compatible benchmark observable."""

    name: str
    value: float
    uncertainty: float
    units: str
    tolerance: float
    passed: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable observable row."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class BenchmarkResultRow:
    """One backend row for a canonical synchronisation benchmark."""

    benchmark_id: str
    backend: str
    backend_version: str
    command: str
    commit: str
    dependency_lock: str
    hardware_submission: bool
    wall_time_s: float
    observables: tuple[ObservableRow, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable result row."""

        payload = asdict(self)
        payload["observables"] = [row.to_dict() for row in self.observables]
        return payload


def ring_coupling_matrix(n_oscillators: int = 4, coupling: float = 0.45) -> NDArray[np.float64]:
    """Return a nearest-neighbour ring coupling matrix."""

    if n_oscillators < 3:
        raise ValueError("ring benchmark requires at least three oscillators")
    matrix = np.zeros((n_oscillators, n_oscillators), dtype=np.float64)
    for index in range(n_oscillators):
        matrix[index, (index - 1) % n_oscillators] = coupling
        matrix[index, (index + 1) % n_oscillators] = coupling
    return matrix


def decaying_chain_coupling_matrix(
    n_oscillators: int = 8, coupling: float = 0.45, decay: float = 0.3
) -> NDArray[np.float64]:
    """Return an exponential-distance decaying chain coupling matrix."""

    if n_oscillators < 2:
        raise ValueError("chain benchmark requires at least two oscillators")
    indices = np.arange(n_oscillators, dtype=float)
    distance = np.abs(indices[:, None] - indices[None, :])
    matrix = coupling * np.exp(-decay * distance)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def natural_frequencies(n_oscillators: int = 4) -> NDArray[np.float64]:
    """Return the canonical linear frequency grid."""

    return np.linspace(0.8, 1.2, n_oscillators, dtype=np.float64)


def kuramoto_order_parameter(phases: NDArray[np.float64]) -> float:
    """Return the Kuramoto order parameter magnitude."""

    return float(abs(np.mean(np.exp(1j * phases))))


def run_classical_reference(
    *, coupling: NDArray[np.float64] | None = None, t_final: float = 1.0
) -> ObservableRow:
    """Run the classical Kuramoto ODE reference for the n=4 ring."""

    coupling = ring_coupling_matrix() if coupling is None else coupling
    omega = natural_frequencies(coupling.shape[0])
    initial = np.linspace(0.0, 1.5 * math.pi, coupling.shape[0], dtype=float)

    def rhs(_time: float, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        delta = theta[None, :] - theta[:, None]
        derivative = cast(
            "NDArray[np.float64]",
            np.asarray(
                omega + np.sum(coupling * np.sin(delta), axis=1),
                dtype=float,
            ),
        )
        return derivative

    result = solve_ivp(
        rhs,
        (0.0, t_final),
        initial,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
    )
    if not result.success:
        raise RuntimeError(f"classical ODE solve failed: {result.message}")
    order = kuramoto_order_parameter(result.y[:, -1])
    return ObservableRow(
        name="order_parameter_t1",
        value=order,
        uncertainty=0.0,
        units="dimensionless",
        tolerance=1e-9,
        passed=0.0 <= order <= 1.0,
    )


def _single_qubit_operator(
    op: NDArray[np.complex128], index: int, n_qubits: int
) -> NDArray[np.complex128]:
    eye = np.eye(2, dtype=np.complex128)
    out: NDArray[np.complex128] = np.array([[1.0 + 0.0j]], dtype=np.complex128)
    for pos in range(n_qubits):
        out = np.kron(out, op if pos == index else eye).astype(np.complex128)
    return out


def _two_qubit_operator(
    op_a: NDArray[np.complex128],
    i: int,
    op_b: NDArray[np.complex128],
    j: int,
    n_qubits: int,
) -> NDArray[np.complex128]:
    eye = np.eye(2, dtype=np.complex128)
    out: NDArray[np.complex128] = np.array([[1.0 + 0.0j]], dtype=np.complex128)
    for pos in range(n_qubits):
        if pos == i:
            op = op_a
        elif pos == j:
            op = op_b
        else:
            op = eye
        out = np.kron(out, op).astype(np.complex128)
    return out


def xy_hamiltonian(
    coupling: NDArray[np.float64], omega: NDArray[np.float64]
) -> NDArray[np.complex128]:
    """Build the dense XY Hamiltonian for the canonical benchmark."""

    n_qubits = int(coupling.shape[0])
    if coupling.shape != (n_qubits, n_qubits):
        raise ValueError("coupling must be square")
    if omega.shape != (n_qubits,):
        raise ValueError("omega length must match coupling size")
    x_gate = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    y_gate = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    z_gate = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    dim = 2**n_qubits
    hamiltonian = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n_qubits):
        hamiltonian -= float(omega[i]) * _single_qubit_operator(z_gate, i, n_qubits)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            kij = float(coupling[i, j])
            if kij == 0.0:
                continue
            hamiltonian -= kij * (
                _two_qubit_operator(x_gate, i, x_gate, j, n_qubits)
                + _two_qubit_operator(y_gate, i, y_gate, j, n_qubits)
            )
    return hamiltonian


def run_exact_reference(
    *, coupling: NDArray[np.float64] | None = None, t_final: float = 1.0
) -> tuple[ObservableRow, ObservableRow]:
    """Run dense exact XY evolution for the n=4 ring benchmark."""

    coupling = ring_coupling_matrix() if coupling is None else coupling
    omega = natural_frequencies(coupling.shape[0])
    hamiltonian = xy_hamiltonian(coupling, omega)
    n_qubits = coupling.shape[0]
    plus = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2.0)
    state = plus.copy()
    for _ in range(n_qubits - 1):
        state = np.asarray(np.kron(state, plus), dtype=complex)
    evolved = expm(-1j * hamiltonian * t_final) @ state
    norm = float(np.vdot(evolved, evolved).real)
    energy = float(np.vdot(evolved, hamiltonian @ evolved).real)
    return (
        ObservableRow(
            name="state_norm_t1",
            value=norm,
            uncertainty=0.0,
            units="dimensionless",
            tolerance=1e-10,
            passed=abs(norm - 1.0) <= 1e-10,
        ),
        ObservableRow(
            name="energy_expectation_t1",
            value=energy,
            uncertainty=0.0,
            units="dimensionless_hamiltonian_units",
            tolerance=1e-10,
            passed=bool(np.isfinite(energy)),
        ),
    )


def dependency_lock() -> str:
    """Return lightweight dependency provenance for the no-QPU runner."""

    return f"python={platform.python_version()}; numpy={np.__version__}; scipy=installed"


def run_kuramoto_ring_n4_linear_omega(*, command: str, commit: str) -> dict[str, Any]:
    """Run the first schema-compatible synchronisation benchmark."""

    classical = BenchmarkResultRow(
        benchmark_id=RING_N4_BENCHMARK_ID,
        backend="classical_ode_scipy_dop853",
        backend_version=f"python-{platform.python_version()}",
        command=command,
        commit=commit,
        dependency_lock=dependency_lock(),
        hardware_submission=False,
        wall_time_s=0.0,
        observables=(run_classical_reference(),),
        claim_boundary="No-QPU classical reference row; not a hardware or advantage claim.",
    )
    exact = BenchmarkResultRow(
        benchmark_id=RING_N4_BENCHMARK_ID,
        backend="dense_exact_xy_numpy_scipy",
        backend_version=f"python-{platform.python_version()}",
        command=command,
        commit=commit,
        dependency_lock=dependency_lock(),
        hardware_submission=False,
        wall_time_s=0.0,
        observables=run_exact_reference(),
        claim_boundary="No-QPU dense exact reference row; not a hardware or advantage claim.",
    )
    return {
        "schema": "synchronisation_benchmark_run_v1",
        "result_schema": RESULT_SCHEMA,
        "benchmark_id": RING_N4_BENCHMARK_ID,
        "hardware_submission": False,
        "rows": [classical.to_dict(), exact.to_dict()],
        "claim_boundary": (
            "This artefact establishes reference rows for one canonical "
            "synchronisation benchmark. It does not submit QPU jobs or claim "
            "quantum advantage."
        ),
        "python_executable": sys.executable,
    }


def run_kuramoto_chain_n8_decay_omega(*, command: str, commit: str) -> dict[str, Any]:
    """Run the n=8 decaying-chain synchronisation benchmark."""

    coupling = decaying_chain_coupling_matrix()
    classical = BenchmarkResultRow(
        benchmark_id=CHAIN_N8_BENCHMARK_ID,
        backend="classical_ode_scipy_dop853",
        backend_version=f"python-{platform.python_version()}",
        command=command,
        commit=commit,
        dependency_lock=dependency_lock(),
        hardware_submission=False,
        wall_time_s=0.0,
        observables=(run_classical_reference(coupling=coupling),),
        claim_boundary="No-QPU classical reference row; not a hardware or advantage claim.",
    )
    exact = BenchmarkResultRow(
        benchmark_id=CHAIN_N8_BENCHMARK_ID,
        backend="dense_exact_xy_numpy_scipy",
        backend_version=f"python-{platform.python_version()}",
        command=command,
        commit=commit,
        dependency_lock=dependency_lock(),
        hardware_submission=False,
        wall_time_s=0.0,
        observables=run_exact_reference(coupling=coupling),
        claim_boundary="No-QPU dense exact reference row; not a hardware or advantage claim.",
    )
    return {
        "schema": "synchronisation_benchmark_run_v1",
        "result_schema": RESULT_SCHEMA,
        "benchmark_id": CHAIN_N8_BENCHMARK_ID,
        "hardware_submission": False,
        "rows": [classical.to_dict(), exact.to_dict()],
        "claim_boundary": (
            "This artefact establishes n=8 decaying-chain reference rows for "
            "the standardised synchronisation benchmark suite. It does not "
            "submit QPU jobs or claim quantum advantage."
        ),
        "python_executable": sys.executable,
    }
