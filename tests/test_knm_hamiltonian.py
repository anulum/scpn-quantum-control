"""Tests for bridge/knm_hamiltonian.py."""
import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    knm_to_ansatz,
    knm_to_hamiltonian,
)


def test_omega_shape():
    assert OMEGA_N_16.shape == (16,)
    assert OMEGA_N_16[0] == pytest.approx(1.329)


def test_knm_paper27_symmetric():
    K = build_knm_paper27()
    assert K.shape == (16, 16)
    np.testing.assert_allclose(K, K.T, atol=1e-12)


def test_knm_paper27_calibration_anchors():
    K = build_knm_paper27()
    assert K[0, 1] == pytest.approx(0.302)
    assert K[1, 2] == pytest.approx(0.201)
    assert K[2, 3] == pytest.approx(0.252)
    assert K[3, 4] == pytest.approx(0.154)


def test_knm_paper27_cross_hierarchy():
    K = build_knm_paper27()
    assert K[0, 15] >= 0.05
    assert K[4, 6] >= 0.15


def test_hamiltonian_hermitian():
    K = np.array([[0, 0.3], [0.3, 0]])
    omega = np.array([1.0, 2.0])
    H = knm_to_hamiltonian(K, omega)
    mat = H.to_matrix()
    np.testing.assert_allclose(mat, mat.conj().T, atol=1e-12)


def test_hamiltonian_small_system():
    K = np.array([[0, 0.5], [0.5, 0]])
    omega = np.array([1.0, 0.0])
    H = knm_to_hamiltonian(K, omega)
    assert H.num_qubits == 2


def test_ansatz_qubit_count():
    K = build_knm_paper27(L=4)
    qc = knm_to_ansatz(K, reps=1)
    assert qc.num_qubits == 4


def test_ansatz_has_parameters():
    K = build_knm_paper27(L=4)
    qc = knm_to_ansatz(K, reps=2)
    assert qc.num_parameters == 4 * 2 * 2  # n * 2 * reps
