# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for MPS tensor network baseline."""

from __future__ import annotations

import pytest

from scpn_quantum_control.benchmarks.mps_baseline import (
    MPSBaselineResult,
    exact_memory,
    mps_baseline_comparison,
    mps_memory,
    quantum_advantage_n,
    required_bond_dimension,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


class TestRequiredBondDimension:
    def test_zero_entropy_chi_1(self):
        assert required_bond_dimension(0.0) == 1

    def test_one_bit_entropy_chi_2(self):
        assert required_bond_dimension(1.0) == 2

    def test_monotonic(self):
        c1 = required_bond_dimension(0.5)
        c2 = required_bond_dimension(2.0)
        assert c2 >= c1


class TestMemory:
    def test_mps_positive(self):
        assert mps_memory(10, 4) > 0

    def test_exact_exponential(self):
        m10 = exact_memory(10)
        m20 = exact_memory(20)
        assert m20 / m10 == pytest.approx(1024.0)


class TestQuantumAdvantageN:
    def test_positive(self):
        n = quantum_advantage_n(chi_max=256)
        assert n > 0

    def test_higher_chi_larger_n(self):
        n_low = quantum_advantage_n(chi_max=64)
        n_high = quantum_advantage_n(chi_max=4096)
        assert n_high > n_low


class TestMPSBaselineComparison:
    def test_returns_result(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = mps_baseline_comparison(K, omega)
        assert isinstance(result, MPSBaselineResult)

    def test_n_qubits(self):
        K = build_knm_paper27(L=6)
        omega = OMEGA_N_16[:6]
        result = mps_baseline_comparison(K, omega)
        assert result.n_qubits == 6

    def test_compression_positive(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = mps_baseline_comparison(K, omega)
        assert result.compression_ratio > 0

    def test_small_system_tractable(self):
        """4 qubits should always be MPS-tractable."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = mps_baseline_comparison(K, omega)
        assert result.mps_tractable

    def test_scpn_mps_comparison(self):
        """Record MPS baseline for SCPN defaults."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = mps_baseline_comparison(K, omega)
        print("\n  MPS baseline (4 osc):")
        print(f"  S(n/2) = {result.half_chain_entropy:.6f}")
        print(f"  χ required = {result.required_bond_dim}")
        print(f"  MPS memory = {result.mps_memory_bytes} bytes")
        print(f"  Exact memory = {result.exact_memory_bytes} bytes")
        print(f"  Compression = {result.compression_ratio:.1f}x")
        print(f"  Advantage threshold = {result.quantum_advantage_threshold} qubits")
        assert isinstance(result.required_bond_dim, int)
