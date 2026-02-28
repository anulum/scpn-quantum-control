"""Tests for control/qaoa_mpc.py."""

import numpy as np
from qiskit.quantum_info import Statevector

from scpn_quantum_control.control.qaoa_mpc import QAOA_MPC
from scpn_quantum_control.hardware.classical import classical_brute_mpc


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


def test_hamiltonian_matches_classical_cost():
    """QAOA Hamiltonian diagonal must match classical_brute_mpc cost on each bitstring."""
    B = np.eye(2)
    target = np.array([0.8, 0.6])
    horizon = 4

    mpc = QAOA_MPC(B, target, horizon=horizon, p_layers=1)
    H = mpc.build_cost_hamiltonian()
    H_mat = np.real(np.diag(np.array(H.to_matrix())))

    a = float(np.linalg.norm(B))
    b = float(np.linalg.norm(target)) / horizon

    for idx in range(2**horizon):
        actions = np.array([(idx >> bit) & 1 for bit in range(horizon)])
        classical_cost = sum((a * actions[t] - b) ** 2 for t in range(horizon))
        assert abs(H_mat[idx] - classical_cost) < 1e-10, (
            f"bitstring {idx:04b}: H={H_mat[idx]:.6f}, classical={classical_cost:.6f}"
        )


def test_optimal_bitstring_matches_brute_force():
    """QAOA Hamiltonian minimum eigenvalue must correspond to brute-force optimal."""
    B = np.eye(2)
    target = np.array([0.8, 0.6])
    horizon = 3

    mpc = QAOA_MPC(B, target, horizon=horizon, p_layers=1)
    H = mpc.build_cost_hamiltonian()
    H_diag = np.real(np.diag(np.array(H.to_matrix())))

    brute = classical_brute_mpc(B, target, horizon=horizon)
    qaoa_min_idx = int(np.argmin(H_diag))
    qaoa_min_actions = np.array([(qaoa_min_idx >> bit) & 1 for bit in range(horizon)])

    np.testing.assert_array_equal(qaoa_min_actions, brute["optimal_actions"])
