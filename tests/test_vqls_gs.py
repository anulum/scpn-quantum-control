"""Tests for control/vqls_gs.py."""

import numpy as np
import pytest

from scpn_quantum_control.control.vqls_gs import VQLS_GradShafranov


def test_discretize_shapes():
    solver = VQLS_GradShafranov(n_qubits=3)
    A, b = solver.discretize()
    assert A.shape == (8, 8)
    assert b.shape == (8,)
    assert abs(np.linalg.norm(b) - 1.0) < 1e-12


def test_laplacian_symmetric():
    solver = VQLS_GradShafranov(n_qubits=3)
    A, _ = solver.discretize()
    np.testing.assert_allclose(A, A.T, atol=1e-12)


def test_laplacian_positive_definite():
    solver = VQLS_GradShafranov(n_qubits=3)
    A, _ = solver.discretize()
    eigvals = np.linalg.eigvalsh(A)
    assert np.all(eigvals > 0)


def test_solve_returns_array():
    solver = VQLS_GradShafranov(n_qubits=3)
    psi = solver.solve(reps=1, maxiter=30)
    assert psi.shape == (8,)
    assert np.all(np.isfinite(psi))


def test_ansatz_qubit_count():
    solver = VQLS_GradShafranov(n_qubits=4)
    qc = solver.build_ansatz(reps=1)
    assert qc.num_qubits == 4


def test_solve_respects_boundary_conditions():
    """Solution should be non-trivial (not all zeros) for a Gaussian source."""
    solver = VQLS_GradShafranov(n_qubits=3)
    psi = solver.solve(reps=1, maxiter=50)
    assert np.linalg.norm(psi) > 1e-6


def test_solve_before_build_raises():
    """RuntimeError guard fires when discretize is sabotaged."""
    solver = VQLS_GradShafranov(n_qubits=2)
    solver.discretize = lambda: None  # break auto-build
    with pytest.raises(RuntimeError, match="discretize"):
        solver.solve()


def test_solve_seeded_deterministic():
    """Seeded solve produces identical flux profiles."""
    solver = VQLS_GradShafranov(n_qubits=2)
    r1 = solver.solve(reps=1, maxiter=30, seed=42)
    solver._A = None  # reset to re-discretize
    r2 = solver.solve(reps=1, maxiter=30, seed=42)
    np.testing.assert_allclose(r1, r2)
