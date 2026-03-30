# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Identity Key
"""Tests for identity/identity_key.py."""

import os

import numpy as np

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.identity.identity_key import (
    identity_fingerprint,
    identity_fingerprint_from_binding_spec,
    prove_identity,
    verify_identity,
)


def test_fingerprint_returns_spectral():
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    fp = identity_fingerprint(K, omega, ansatz_reps=1, maxiter=30)
    assert "spectral" in fp
    assert "fiedler" in fp["spectral"]
    assert fp["spectral"]["fiedler"] > 0


def test_fingerprint_returns_ground_energy():
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    fp = identity_fingerprint(K, omega, ansatz_reps=1, maxiter=20)
    assert "ground_energy" in fp
    assert np.isfinite(fp["ground_energy"])


def test_fingerprint_commitment_is_hex():
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    fp = identity_fingerprint(K, omega, ansatz_reps=1, maxiter=20)
    assert len(fp["commitment"]) == 64  # SHA-256 hex


def test_fingerprint_n_parameters():
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    fp = identity_fingerprint(K, omega, ansatz_reps=1, maxiter=10)
    # 4*(4-1)/2 = 6 independent parameters
    assert fp["n_parameters"] == 6


def test_different_k_different_fingerprint():
    omega = OMEGA_N_16[:3]
    K1 = build_knm_paper27(L=3, K_base=0.3)
    K2 = build_knm_paper27(L=3, K_base=0.8)
    fp1 = identity_fingerprint(K1, omega, ansatz_reps=1, maxiter=20)
    fp2 = identity_fingerprint(K2, omega, ansatz_reps=1, maxiter=20)
    assert fp1["commitment"] != fp2["commitment"]
    assert fp1["spectral"]["fiedler"] != fp2["spectral"]["fiedler"]


def test_challenge_response_correct_k():
    K = build_knm_paper27(L=3)
    challenge = os.urandom(32)
    response = prove_identity(K, challenge)
    assert verify_identity(K, challenge, response)


def test_challenge_response_wrong_k():
    K_real = build_knm_paper27(L=3, K_base=0.45)
    K_fake = build_knm_paper27(L=3, K_base=0.90)
    challenge = os.urandom(32)
    response = prove_identity(K_fake, challenge)
    assert not verify_identity(K_real, challenge, response)


def test_from_binding_spec():
    spec = {
        "layers": [
            {"oscillator_ids": ["a", "b"], "natural_frequency": 1.0},
            {"oscillator_ids": ["c"], "natural_frequency": 2.0},
        ],
        "coupling": {"base_strength": 0.5, "decay_alpha": 0.2},
    }
    fp = identity_fingerprint_from_binding_spec(spec, ansatz_reps=1, maxiter=20)
    assert fp["n_qubits"] == 3
    assert fp["n_parameters"] == 3
