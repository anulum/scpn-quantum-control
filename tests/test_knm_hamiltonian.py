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


def test_pauli_ordering_energy_on_zero_state():
    """<0...0|H|0...0> must equal -sum(omega) to verify qubit labeling."""
    from qiskit.quantum_info import Statevector

    n = 4
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    H = knm_to_hamiltonian(K, omega)

    sv = Statevector.from_int(0, dims=2**n)
    E = float(sv.expectation_value(H).real)
    # H = -sum(omega_i * Z_i) - sum(K_ij * (XX+YY))
    # |0...0>: <Z_i>=+1, <XX>=<YY>=0
    np.testing.assert_allclose(E, -np.sum(omega), atol=1e-12)


def test_pauli_ordering_single_flip():
    """Flipping qubit 0 should change energy by +2*omega[0]."""
    from qiskit import QuantumCircuit as QC
    from qiskit.quantum_info import Statevector

    n = 3
    omega = np.array([1.0, 2.0, 3.0])
    K = np.zeros((n, n))  # no coupling → only Z terms
    H = knm_to_hamiltonian(K, omega)

    # |0,0,0>: E = -(1+2+3) = -6
    sv0 = Statevector.from_int(0, dims=2**n)
    E0 = float(sv0.expectation_value(H).real)
    np.testing.assert_allclose(E0, -6.0, atol=1e-12)

    # Flip qubit 0: <Z_0> = -1, others still +1 → E = +1 -2 -3 = -4
    qc = QC(n)
    qc.x(0)
    sv1 = Statevector.from_instruction(qc)
    E1 = float(sv1.expectation_value(H).real)
    np.testing.assert_allclose(E1, -4.0, atol=1e-12)
