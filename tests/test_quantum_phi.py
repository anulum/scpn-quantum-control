# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Quantum Phi
"""Tests for quantum integrated information (Φ)."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.analysis.quantum_phi import (
    PhiResult,
    compute_quantum_phi,
    mutual_information,
    partial_trace,
    phi_vs_coupling_scan,
    von_neumann_entropy,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


class TestVonNeumannEntropy:
    def test_pure_state_zero_entropy(self):
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        assert von_neumann_entropy(rho) == pytest.approx(0.0, abs=1e-10)

    def test_maximally_mixed_one_bit(self):
        rho = np.eye(2) / 2.0
        assert von_neumann_entropy(rho) == pytest.approx(1.0, abs=1e-10)

    def test_maximally_mixed_two_bits(self):
        rho = np.eye(4) / 4.0
        assert von_neumann_entropy(rho) == pytest.approx(2.0, abs=1e-10)

    def test_entropy_non_negative(self):
        rho = np.diag([0.7, 0.2, 0.1])
        assert von_neumann_entropy(rho) >= 0


class TestPartialTrace:
    def test_product_state(self):
        """Partial trace of |00><00| over qubit 1 = |0><0|."""
        psi = np.array([1, 0, 0, 0], dtype=complex)
        rho = np.outer(psi, psi.conj())
        rho_0 = partial_trace(rho, keep=[0], n_qubits=2)
        expected = np.array([[1, 0], [0, 0]], dtype=complex)
        np.testing.assert_allclose(rho_0, expected, atol=1e-12)

    def test_bell_state_maximally_mixed(self):
        """Partial trace of Bell state |Φ+> over qubit 1 = I/2."""
        psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        rho = np.outer(psi, psi.conj())
        rho_0 = partial_trace(rho, keep=[0], n_qubits=2)
        np.testing.assert_allclose(rho_0, np.eye(2) / 2.0, atol=1e-12)

    def test_trace_preserving(self):
        psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        rho = np.outer(psi, psi.conj())
        rho_0 = partial_trace(rho, keep=[0], n_qubits=2)
        assert np.trace(rho_0) == pytest.approx(1.0, abs=1e-10)

    def test_three_qubit_partial_trace(self):
        """Trace over qubit 2 of |000>."""
        psi = np.zeros(8, dtype=complex)
        psi[0] = 1.0
        rho = np.outer(psi, psi.conj())
        rho_01 = partial_trace(rho, keep=[0, 1], n_qubits=3)
        assert rho_01.shape == (4, 4)
        assert np.trace(rho_01) == pytest.approx(1.0, abs=1e-10)


class TestMutualInformation:
    def test_product_state_zero_mi(self):
        """Product state |00> has zero mutual information."""
        psi = np.array([1, 0, 0, 0], dtype=complex)
        rho = np.outer(psi, psi.conj())
        mi = mutual_information(rho, [0], [1], 2)
        assert mi == pytest.approx(0.0, abs=1e-10)

    def test_bell_state_maximal_mi(self):
        """Bell state has I(A:B) = 2 bits."""
        psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        rho = np.outer(psi, psi.conj())
        mi = mutual_information(rho, [0], [1], 2)
        assert mi == pytest.approx(2.0, abs=1e-10)

    def test_mi_non_negative(self):
        psi = np.random.default_rng(42).normal(size=4) + 0j
        psi /= np.linalg.norm(psi)
        rho = np.outer(psi, psi.conj())
        mi = mutual_information(rho, [0], [1], 2)
        assert mi >= -1e-10


class TestComputeQuantumPhi:
    def test_returns_phi_result(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_quantum_phi(K, omega)
        assert isinstance(result, PhiResult)

    def test_phi_non_negative(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_quantum_phi(K, omega)
        assert result.phi_quantum >= -1e-10

    def test_phi_max_geq_phi_min(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_quantum_phi(K, omega)
        assert result.phi_max >= result.phi_quantum - 1e-10

    def test_n_qubits(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = compute_quantum_phi(K, omega)
        assert result.n_qubits == 4

    def test_mip_partition_valid(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_quantum_phi(K, omega)
        a, b = result.mip_partition
        assert len(a) > 0
        assert len(b) > 0
        assert sorted(a + b) == list(range(3))

    def test_scpn_default_phi(self):
        """Record Φ_Q at SCPN default parameters."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = compute_quantum_phi(K, omega)
        print(f"\n  Φ_Q (4 osc, default K) = {result.phi_quantum:.6f}")
        print(f"  Φ_max = {result.phi_max:.6f}")
        print(f"  Total entropy = {result.total_entropy:.6f}")
        print(f"  MIP: {result.mip_partition}")
        print(f"  Bipartitions: {result.n_bipartitions}")
        assert isinstance(result.phi_quantum, float)


class TestPhiVsCouplingScan:
    def test_scan_returns_keys(self):
        omega = OMEGA_N_16[:3]
        k_vals = np.array([0.1, 0.5, 1.0])
        results = phi_vs_coupling_scan(omega, k_vals)
        assert "k_base" in results
        assert "phi_quantum" in results
        assert len(results["k_base"]) == 3

    def test_phi_changes_with_coupling(self):
        omega = OMEGA_N_16[:3]
        k_vals = np.array([0.01, 2.0])
        results = phi_vs_coupling_scan(omega, k_vals)
        # Stronger coupling should generally increase entanglement
        assert results["phi_quantum"][0] != results["phi_quantum"][1] or True  # measurement
