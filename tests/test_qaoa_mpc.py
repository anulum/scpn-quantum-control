"""Tests for control/qaoa_mpc.py."""

import numpy as np

from scpn_quantum_control.control.qaoa_mpc import QAOA_MPC


def test_build_cost_hamiltonian():
    B = np.array([[1.0, 0.0], [0.0, 1.0]])
    target = np.array([0.5, 0.5])
    mpc = QAOA_MPC(B, target, horizon=3, p_layers=1)
    H = mpc.build_cost_hamiltonian()
    assert H.num_qubits == 3


def test_optimize_returns_binary():
    B = np.array([[1.0]])
    target = np.array([1.0])
    mpc = QAOA_MPC(B, target, horizon=4, p_layers=1)
    actions = mpc.optimize()
    assert actions.shape == (4,)
    assert set(np.unique(actions)).issubset({0, 1})


def test_cost_hamiltonian_hermitian():
    B = np.eye(2)
    target = np.array([1.0, 0.0])
    mpc = QAOA_MPC(B, target, horizon=3, p_layers=1)
    H = mpc.build_cost_hamiltonian()
    mat = H.to_matrix()
    np.testing.assert_allclose(mat, mat.conj().T, atol=1e-12)
