"""Tests for control/vqls_gs.py."""

import numpy as np

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
