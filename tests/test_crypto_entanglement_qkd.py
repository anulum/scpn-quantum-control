"""Tests for entanglement_qkd: SCPN-QKD protocol."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.bridge import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.crypto.entanglement_qkd import (
    bell_inequality_test,
    correlator_matrix,
    scpn_qkd_protocol,
)
from scpn_quantum_control.crypto.knm_key import prepare_key_state


def test_qkd_protocol_returns_keys():
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    result = scpn_qkd_protocol(K, omega, alice_qubits=[0, 1], bob_qubits=[2, 3])
    assert "raw_key_alice" in result
    assert "raw_key_bob" in result
    assert "qber" in result
    assert 0 <= result["qber"] <= 1
    assert result["ground_energy"] < 0


def test_correlator_matrix_shape():
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    state = prepare_key_state(K, omega, ansatz_reps=1, maxiter=20)
    sv = state["statevector"]
    corr = correlator_matrix(sv, [0, 1], [2, 3])
    assert corr.shape == (2, 2)


def test_correlator_matrix_has_nonzero_entries():
    """Ground state of coupled H should have cross-correlations."""
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    state = prepare_key_state(K, omega, ansatz_reps=2, maxiter=50)
    sv = state["statevector"]
    corr = correlator_matrix(sv, [0, 1], [2, 3])
    assert np.any(np.abs(corr) > 1e-6), "Expected nonzero cross-correlations"


def test_bell_test_returns_S():
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    state = prepare_key_state(K, omega, ansatz_reps=1, maxiter=20)
    sv = state["statevector"]
    result = bell_inequality_test(sv, qubit_a=0, qubit_b=1, n_total=4)
    assert "S" in result
    assert "violates_classical" in result
    assert "correlators" in result
    assert result["S"] >= 0
