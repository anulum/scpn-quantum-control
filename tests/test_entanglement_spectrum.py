# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Entanglement Spectrum
"""Tests for entanglement spectrum analysis."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.analysis.entanglement_spectrum import (
    EntanglementResult,
    entanglement_analysis,
    entanglement_entropy_half_chain,
    entanglement_spectrum_half_chain,
    entropy_vs_coupling_scan,
    entropy_vs_subsystem_size,
    fit_cft_central_charge,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


class TestHalfChainEntropy:
    def test_non_negative(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        s = entanglement_entropy_half_chain(K, omega)
        assert s >= -1e-10

    def test_bounded_by_log_dim(self):
        """S <= log2(dim_A) = n/2 for n qubits."""
        n = 4
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        s = entanglement_entropy_half_chain(K, omega)
        assert s <= n / 2 + 1e-10

    def test_increases_with_coupling(self):
        """Stronger coupling → more entanglement."""
        omega = OMEGA_N_16[:4]
        s_weak = entanglement_entropy_half_chain(build_knm_paper27(L=4, K_base=0.01), omega)
        s_strong = entanglement_entropy_half_chain(build_knm_paper27(L=4, K_base=2.0), omega)
        assert s_strong >= s_weak


class TestEntanglementSpectrum:
    def test_eigenvalues_sum_to_one(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        spectrum = entanglement_spectrum_half_chain(K, omega)
        assert np.sum(spectrum) == pytest.approx(1.0, abs=1e-10)

    def test_eigenvalues_non_negative(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        spectrum = entanglement_spectrum_half_chain(K, omega)
        assert np.all(spectrum >= -1e-12)

    def test_descending_order(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        spectrum = entanglement_spectrum_half_chain(K, omega)
        for i in range(len(spectrum) - 1):
            assert spectrum[i] >= spectrum[i + 1] - 1e-12

    def test_size(self):
        """Spectrum of ρ_A has 2^(n/2) eigenvalues."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        spectrum = entanglement_spectrum_half_chain(K, omega)
        assert len(spectrum) == 4  # 2^(4/2) = 4


class TestEntropyVsSubsystemSize:
    def test_length(self):
        K = build_knm_paper27(L=6)
        omega = OMEGA_N_16[:6]
        s_vs_l = entropy_vs_subsystem_size(K, omega)
        assert len(s_vs_l) == 3  # 6//2 = 3

    def test_non_negative(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        s_vs_l = entropy_vs_subsystem_size(K, omega)
        for s in s_vs_l:
            assert s >= -1e-10


class TestFitCFTCentralCharge:
    def test_none_for_too_few_points(self):
        c = fit_cft_central_charge(np.array([1.0, 2.0]), np.array([0.5, 0.7]), 4)
        assert c is None

    def test_returns_float(self):
        sizes = np.arange(1, 5, dtype=float)
        entropies = 0.3 * np.log(sizes) + 0.5
        c = fit_cft_central_charge(sizes, entropies, 8)
        assert isinstance(c, float)


class TestEntanglementAnalysis:
    def test_returns_result(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = entanglement_analysis(K, omega)
        assert isinstance(result, EntanglementResult)

    def test_n_qubits(self):
        K = build_knm_paper27(L=6)
        omega = OMEGA_N_16[:6]
        result = entanglement_analysis(K, omega)
        assert result.n_qubits == 6

    def test_scpn_default_entanglement(self):
        """Record entanglement at SCPN default parameters."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = entanglement_analysis(K, omega)
        print(f"\n  S(n/2) at default K = {result.half_chain_entropy:.6f}")
        print(f"  S vs |A|: {[f'{s:.4f}' for s in result.entropy_vs_subsystem]}")
        print(f"  CFT c (estimated) = {result.cft_central_charge}")
        print(f"  Spectrum (top 4): {result.entanglement_spectrum[:4]}")
        assert isinstance(result.half_chain_entropy, float)


class TestEntropyVsCouplingScan:
    def test_scan_returns_keys(self):
        omega = OMEGA_N_16[:4]
        k_vals = np.array([0.1, 0.5, 1.0])
        results = entropy_vs_coupling_scan(omega, k_vals)
        assert "k_base" in results
        assert "half_chain_entropy" in results
        assert len(results["k_base"]) == 3

    def test_entropy_changes(self):
        omega = OMEGA_N_16[:4]
        k_vals = np.array([0.01, 3.0])
        results = entropy_vs_coupling_scan(omega, k_vals)
        assert results["half_chain_entropy"][0] != results["half_chain_entropy"][1]
