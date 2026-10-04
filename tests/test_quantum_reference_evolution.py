# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — public quantum reference conformance
"""Compare original density, sparse-action and Trotter APIs with independent states."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray
from qiskit.quantum_info import Statevector
from scipy.linalg import expm

from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix
from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.execution_reservations import active_reserved_bytes
from scpn_quantum_control.phase.lindblad import LindbladKuramotoSolver
from scpn_quantum_control.phase.lindblad_engine import LindbladSyncEngine
from scpn_quantum_control.phase.xy_kuramoto import QuantumKuramotoSolver


@pytest.mark.parametrize("time", [0.0, 0.17, 1.1, 3.3])
def test_quantum_reference_evolution_01(time: float) -> None:
    """The three real routes rotate a plus state with the declared Hamiltonian sign."""
    frequency = 1.3
    coupling = np.zeros((1, 1))
    omega = np.array([-frequency / 2])
    initial = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
    rho = np.outer(initial, initial.conj())
    expected = initial * np.exp(np.array([-1j, 1j]) * frequency * time / 2)
    expected_rho = np.outer(expected, expected.conj())
    unitary = QuantumKuramotoSolver(1, coupling, omega)
    state = Statevector(initial).evolve(unitary.evolve(time).decompose(reps=2))
    result = LindbladKuramotoSolver(1, coupling, omega).run(
        time, 0.1, initial_density_matrix=rho, atol=1e-12, rtol=1e-11
    )
    np.testing.assert_allclose(state.data, expected, atol=1e-10, rtol=1e-8)
    actual_rho = cast(NDArray[np.complex128], result["rho_final"])
    np.testing.assert_allclose(actual_rho, expected_rho, atol=1e-10, rtol=1e-8)
    for pauli, value in (
        (np.array([[0, 1], [1, 0]]), np.cos(frequency * time)),
        (np.array([[0, -1j], [1j, 0]]), np.sin(frequency * time)),
    ):
        assert np.trace(pauli @ actual_rho).real == pytest.approx(value, abs=1e-10, rel=1e-8)
    sparse = LindbladSyncEngine(coupling, omega, gamma=0).evolve(
        time, 4, initial_state=initial, n_traj=1
    )
    np.testing.assert_allclose(sparse["final_state"], expected_rho, atol=1e-10, rtol=1e-8)
    np.testing.assert_array_equal(rho, np.outer(initial, initial.conj()))


@pytest.mark.parametrize("order", [1, 2])
def test_quantum_reference_evolution_02(order: int) -> None:
    """Norm conservation and product-formula refinement are distinct observable checks."""
    coupling = np.array([[0, 0.37], [0.37, 0]])
    omega = np.array([-0.4, 0.8])
    hamiltonian = np.array(
        [[-0.4, 0, 0, 0], [0, -1.2, -0.74, 0], [0, -0.74, 1.2, 0], [0, 0, 0, 0.4]]
    )
    initial = np.array([1, 2j, -0.5, 0.3j], dtype=np.complex128)
    initial /= np.linalg.norm(initial)
    time = 0.83
    expected = expm(-1j * time * hamiltonian) @ initial
    solver = QuantumKuramotoSolver(2, coupling, omega, trotter_order=order)
    errors = []
    for reps in (4, 8, 16):
        circuit = solver.evolve(time, reps).decompose(reps=2)
        state = Statevector(initial).evolve(circuit)
        assert np.vdot(state.data, state.data).real == pytest.approx(1, abs=1e-12)
        errors.append(float(np.linalg.norm(state.data - expected)))
    assert errors[0] > errors[1] > errors[2] > 1e-12
    expected_ratio = 2**order
    assert errors[0] / errors[1] == pytest.approx(expected_ratio, rel=0.15)
    assert errors[1] / errors[2] == pytest.approx(expected_ratio, rel=0.15)
    density = LindbladKuramotoSolver(2, coupling, omega).run(
        time,
        0.07,
        initial_density_matrix=np.outer(initial, initial.conj()),
        atol=1e-12,
        rtol=1e-11,
    )["rho_final"]
    np.testing.assert_allclose(density, np.outer(expected, expected.conj()), atol=1e-10, rtol=1e-8)
    assert np.trace(density) == pytest.approx(1, abs=1e-10)
    np.testing.assert_allclose(density, density.conj().T, atol=1e-12)
    assert np.linalg.eigvalsh(density).min() >= -1e-10


@pytest.mark.parametrize("route", ["canonical", "sync"])
def test_quantum_reference_evolution_03(route: str) -> None:
    """Both public density routes reject a unit-trace matrix with negative population."""
    coupling = np.zeros((1, 1))
    omega = np.zeros(1)
    rho = np.diag([1.1, -0.1]).astype(np.complex128)
    original = rho.copy()
    if route == "canonical":
        solver = LindbladKuramotoSolver(1, coupling, omega)
        with pytest.raises(ValueError, match="positive semidefinite"):
            solver.run(0.1, 0.1, initial_density_matrix=rho)
        assert solver._H is None
    else:
        engine = LindbladSyncEngine(coupling, omega)
        with pytest.raises(ValueError, match="positive semidefinite"):
            engine.evolve(0.1, 1, method="density_matrix", initial_state=rho)
        assert engine.H_dense is None
    np.testing.assert_array_equal(rho, original)


def test_quantum_reference_evolution_04() -> None:
    """Original public matrix exports and cached density runs refuse before allocation."""
    count = 32
    coupling = np.zeros((count, count))
    omega = np.zeros(count)
    with pytest.raises(DenseAllocationError):
        knm_to_dense_matrix(coupling, omega, max_dense_gib=0.001)
    large = LindbladKuramotoSolver(count, coupling, omega, max_dense_gib=0.001)
    with pytest.raises(DenseAllocationError):
        large.build()
    assert large._H is None
    solver = LindbladKuramotoSolver(1, np.zeros((1, 1)), np.zeros(1))
    solver.build()
    previous = solver._H
    charged = active_reserved_bytes()
    with pytest.raises(DenseAllocationError):
        solver.run(0.1, 0.1, max_dense_gib=1e-12)
    assert solver._H is previous
    with pytest.raises(DenseAllocationError):
        solver.run(1e308, 1e-308, max_dense_gib=0.001)
    assert solver._H is previous
    assert active_reserved_bytes() == charged
    engine = LindbladSyncEngine(np.zeros((11, 11)), np.zeros(11))
    with pytest.raises(RuntimeError, match="N <= 10"):
        engine.liouvillian(np.array([1], dtype=np.complex128))
    assert engine.H_dense is None


@pytest.mark.parametrize("step", [0.2, 0.1, 0.05])
def test_trotter_trajectory_grid_matches_an_independent_exact_state(step: float) -> None:
    """Real trajectory samples retain their time and little-endian observable meaning."""
    coupling = np.array([[0, 0.37], [0.37, 0]])
    omega = np.array([-0.4, 0.8])
    hamiltonian = np.array(
        [[-0.4, 0, 0, 0], [0, -1.2, -0.74, 0], [0, -0.74, 1.2, 0], [0, 0, 0, 0.4]]
    )
    zero = np.array([np.cos(omega[0] / 2), np.sin(omega[0] / 2)])
    one = np.array([np.cos(omega[1] / 2), np.sin(omega[1] / 2)])
    initial = np.kron(one, zero).astype(np.complex128)
    pauli_x = np.array([[0, 1], [1, 0]])
    pauli_y = np.array([[0, -1j], [1j, 0]])
    mean_x = (np.kron(pauli_x, np.eye(2)) + np.kron(np.eye(2), pauli_x)) / 2
    mean_y = (np.kron(pauli_y, np.eye(2)) + np.kron(np.eye(2), pauli_y)) / 2
    solver = QuantumKuramotoSolver(2, coupling, omega, trotter_order=2)
    result = solver.run(0.8, step, trotter_per_step=8)
    for time, value in zip(result.times, result.R, strict=True):
        state = expm(-1j * hamiltonian * time) @ initial
        expected = abs(np.vdot(state, (mean_x + 1j * mean_y) @ state))
        assert value == pytest.approx(expected, abs=2e-5)
    assert result.times[-1] == pytest.approx(0.8, abs=1e-15)


@pytest.mark.parametrize(
    ("atol", "rtol"), [(0.0, 1e-6), (1e-8, 0.0), (-1.0, 1e-6), (1e-8, np.nan), (np.inf, 1e-6)]
)
def test_invalid_integration_tolerance_preserves_unbuilt_state(atol: float, rtol: float) -> None:
    """Nonpositive or nonfinite explicit tolerances refuse before density allocation."""
    solver = LindbladKuramotoSolver(1, np.zeros((1, 1)), np.zeros(1))
    charged = active_reserved_bytes()
    with pytest.raises(ValueError):
        solver.run(0.1, 0.1, atol=atol, rtol=rtol)
    assert solver._H is None
    assert active_reserved_bytes() == charged


@pytest.mark.parametrize("time", [1e20, 1e308])
def test_unrepresentable_finite_density_history_refuses(time: float) -> None:
    """Finite arguments can still exceed native addressability before a time array exists."""
    solver = LindbladKuramotoSolver(1, np.zeros((1, 1)), np.zeros(1))
    with pytest.raises(DenseAllocationError, match="density history"):
        solver.run(time, 1.0, max_dense_gib=0.001)
    assert solver._H is None


def test_native_subkernels_use_the_independent_little_endian_hamiltonian() -> None:
    """Actual compiled dense/sparse operators and expectations agree with an explicit matrix."""
    import scpn_quantum_engine as engine

    coupling = np.array([[0, 0.37], [0.37, 0]])
    omega = np.array([-0.4, 0.8])
    expected = np.array([[-0.4, 0, 0, 0], [0, -1.2, -0.74, 0], [0, -0.74, 1.2, 0], [0, 0, 0, 0.4]])
    for backend in ("python", "rust"):
        np.testing.assert_allclose(
            knm_to_dense_matrix(coupling, omega, backend=backend), expected, atol=1e-14
        )
    rows, columns, values = engine.build_sparse_xy_hamiltonian(coupling.ravel(), omega, 2)
    sparse = np.zeros((4, 4))
    sparse[rows, columns] = values
    np.testing.assert_allclose(sparse, expected, atol=1e-14)
    state = np.array([1, 1j, 0, 0], dtype=np.complex128) / np.sqrt(2)
    x, y = engine.all_xy_expectations(state.real.copy(), state.imag.copy(), 2)
    np.testing.assert_allclose(x, [0, 0], atol=1e-14)
    np.testing.assert_allclose(y, [1, 0], atol=1e-14)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_physical_input_refusal_retains_a_previously_built_density_cache(
    dtype: type[np.float64] | type[np.complex128],
) -> None:
    """A rejected input cannot replace a prior admitted density run or its cached operators."""
    rho = np.diag([1.1, -0.1]).astype(np.complex128)
    solver = LindbladKuramotoSolver(1, np.zeros((1, 1)), np.zeros(1))
    first = solver.run(0.0, 0.1)
    hamiltonian = solver._H
    charged = active_reserved_bytes()
    with pytest.raises(ValueError, match="positive semidefinite"):
        solver.run(0.1, 0.1, initial_density_matrix=rho)
    assert solver._H is hamiltonian
    assert active_reserved_bytes() == charged
    np.testing.assert_array_equal(first["rho_final"], [[1, 0], [0, 0]])
    engine = LindbladSyncEngine(np.zeros((1, 1)), np.zeros(1), gamma=0)
    engine.evolve(0.01, 1, method="density_matrix")
    previous = engine.H_dense
    mixed = cast(NDArray[np.float64] | NDArray[np.complex128], np.diag([0.3, 0.7]).astype(dtype))
    reused = engine.evolve(0.02, 2, method="density_matrix", initial_state=mixed)
    np.testing.assert_array_equal(reused["final_state"], mixed)
    assert engine.H_dense is previous
    with pytest.raises(ValueError, match="positive semidefinite"):
        engine.evolve(0.1, 1, method="density_matrix", initial_state=rho)
    assert engine.H_dense is previous
    with pytest.raises(DenseAllocationError):
        engine.evolve(0.1, 1, method="density_matrix", max_dense_gib=1e-12)
    assert engine.H_dense is previous
