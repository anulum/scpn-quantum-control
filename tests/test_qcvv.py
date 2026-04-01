# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Qcvv
"""Tests for QCVV certification."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.hardware.qcvv import (
    QCVVResult,
    cross_entropy_score,
    mirror_circuit_fidelity,
    qcvv_certify,
    simulate_xeb,
    state_fidelity,
)


class TestStateFidelity:
    def test_identical_states(self):
        psi = np.array([1, 0, 0, 0], dtype=complex)
        assert state_fidelity(psi, psi) == pytest.approx(1.0)

    def test_orthogonal_states(self):
        psi1 = np.array([1, 0, 0, 0], dtype=complex)
        psi2 = np.array([0, 1, 0, 0], dtype=complex)
        assert state_fidelity(psi1, psi2) == pytest.approx(0.0)

    def test_bounded(self):
        rng = np.random.default_rng(42)
        psi1 = rng.normal(size=4) + 0j
        psi1 /= np.linalg.norm(psi1)
        psi2 = rng.normal(size=4) + 0j
        psi2 /= np.linalg.norm(psi2)
        f = state_fidelity(psi1, psi2)
        assert 0 <= f <= 1.0


class TestMirrorCircuit:
    def test_perfect_return(self):
        """Statevector simulator should return to |0> perfectly."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        f = mirror_circuit_fidelity(K, omega)
        assert f > 0.99

    def test_bounded(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        f = mirror_circuit_fidelity(K, omega)
        assert 0 <= f <= 1.0


class TestXEB:
    def test_perfect_xeb_near_one(self):
        """No noise → XEB near 1."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        xeb = simulate_xeb(K, omega, noise_level=0.0, n_shots=5000)
        assert xeb > 0.8

    def test_cross_entropy_score_uniform(self):
        """Uniform distribution → XEB ≈ 0."""
        n = 8
        probs = np.ones(n) / n
        counts = np.ones(n) * 100
        xeb = cross_entropy_score(probs, counts, 800)
        assert abs(xeb) < 0.2


class TestQCVVCertify:
    def test_returns_result(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = qcvv_certify(K, omega)
        assert isinstance(result, QCVVResult)

    def test_simulator_certified(self):
        """Statevector simulator should pass QCVV."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = qcvv_certify(K, omega)
        assert result.certified

    def test_n_qubits(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = qcvv_certify(K, omega)
        assert result.n_qubits == 4

    def test_scpn_qcvv(self):
        """Record QCVV certification."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = qcvv_certify(K, omega)
        print("\n  QCVV (3 osc):")
        print(f"  State fidelity: {result.state_fidelity:.4f}")
        print(f"  Mirror fidelity: {result.mirror_fidelity:.4f}")
        print(f"  XEB score: {result.xeb_score:.4f}")
        print(f"  Certified: {result.certified}")
        assert result.certified
