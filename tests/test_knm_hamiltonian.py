# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Knm Hamiltonian
"""Tests for bridge/knm_hamiltonian.py."""

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    build_kuramoto_ring,
    knm_to_ansatz,
    knm_to_hamiltonian,
)

SIZES = [2, 3, 4, 6, 8, 16]


@pytest.mark.parametrize("n", SIZES)
def test_knm_paper27_symmetric(n):
    K = build_knm_paper27(L=n)
    assert K.shape == (n, n)
    np.testing.assert_allclose(K, K.T, atol=1e-12)


def test_knm_paper27_cross_hierarchy():
    K = build_knm_paper27()
    assert K[0, 15] >= 0.05
    assert K[4, 6] >= 0.15


@pytest.mark.parametrize("n", [2, 3, 4, 6])
def test_hamiltonian_hermitian(n):
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    H = knm_to_hamiltonian(K, omega)
    mat = H.to_matrix()
    np.testing.assert_allclose(mat, mat.conj().T, atol=1e-12)


@pytest.mark.parametrize("n", [2, 3, 4, 6])
def test_hamiltonian_qubit_count(n):
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    H = knm_to_hamiltonian(K, omega)
    assert H.num_qubits == n


@pytest.mark.parametrize("n", [2, 3, 4, 6])
def test_ansatz_qubit_count(n):
    K = build_knm_paper27(L=n)
    qc = knm_to_ansatz(K, reps=1)
    assert qc.num_qubits == n


@pytest.mark.parametrize("n,reps", [(2, 1), (3, 2), (4, 2), (6, 1)])
def test_ansatz_has_parameters(n, reps):
    K = build_knm_paper27(L=n)
    qc = knm_to_ansatz(K, reps=reps)
    assert qc.num_parameters == n * 2 * reps


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


def test_knm_omega_shape_mismatch():
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:3]  # 3 != 4
    with pytest.raises(ValueError, match="rows but omega has"):
        knm_to_hamiltonian(K, omega)


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


@pytest.mark.parametrize("n", [3, 4, 6, 8])
def test_kuramoto_ring_symmetric(n):
    K, omega = build_kuramoto_ring(n, coupling=0.5, rng_seed=0)
    assert K.shape == (n, n)
    np.testing.assert_allclose(K, K.T, atol=1e-15)
    assert len(omega) == n
    assert np.count_nonzero(K) == 2 * n


@pytest.mark.parametrize("n", [3, 4, 6])
def test_kuramoto_ring_hamiltonian(n):
    K, omega = build_kuramoto_ring(n, coupling=1.0, rng_seed=42)
    H = knm_to_hamiltonian(K, omega)
    assert H.num_qubits == n
    mat = H.to_matrix()
    np.testing.assert_allclose(mat, mat.conj().T, atol=1e-12)


def test_kuramoto_ring_custom_omega():
    omega_in = np.array([1.0, 2.0, 3.0])
    K, omega_out = build_kuramoto_ring(3, omega=omega_in)
    np.testing.assert_array_equal(omega_out, omega_in)
