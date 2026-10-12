# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Mps Baseline
"""Tests for MPS tensor network baseline."""

from __future__ import annotations

import numpy as np
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
    """Check the entropy lower bound independently of solver qualification."""

    def test_zero_entropy_chi_1(self) -> None:
        """A product partition has lower-bound bond one."""
        assert required_bond_dimension(0.0) == 1

    def test_one_bit_entropy_chi_2(self) -> None:
        """One bit of bipartite entropy requires at least bond two."""
        assert required_bond_dimension(1.0) == 2

    def test_monotonic(self) -> None:
        """Increasing entropy cannot reduce the necessary bond lower bound."""
        c1 = required_bond_dimension(0.5)
        c2 = required_bond_dimension(2.0)
        assert c2 >= c1


class TestMemory:
    """Verify declared tensor and dense component estimates."""

    def test_mps_positive(self) -> None:
        """A nonempty MPS component estimate needs positive storage."""
        assert mps_memory(10, 4) > 0

    def test_exact_exponential(self) -> None:
        """Ten extra qubits multiply dense amplitude storage by 1024."""
        m10 = exact_memory(10)
        m20 = exact_memory(20)
        assert m20 / m10 == pytest.approx(1024.0)


class TestQuantumAdvantageN:
    """Check the legacy cap heuristic without a hardware-advantage claim."""

    def test_positive(self) -> None:
        """The declared positive cap has a positive heuristic crossing size."""
        n = quantum_advantage_n(chi_max=256)
        assert n > 0

    def test_higher_chi_larger_n(self) -> None:
        """A larger cap delays the assumed volume-law crossing."""
        n_low = quantum_advantage_n(chi_max=64)
        n_high = quantum_advantage_n(chi_max=4096)
        assert n_high > n_low


class TestMPSBaselineComparison:
    """Exercise the actual model-to-entropy storage estimation path."""

    def test_returns_result(self) -> None:
        """Return a typed estimate from the original four-site model."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = mps_baseline_comparison(K, omega)
        assert isinstance(result, MPSBaselineResult)

    def test_n_qubits(self) -> None:
        """Keep the six-site model size in the resulting estimate."""
        K = build_knm_paper27(L=6)
        omega = OMEGA_N_16[:6]
        result = mps_baseline_comparison(K, omega)
        assert result.n_qubits == 6

    def test_compression_positive(self) -> None:
        """Return a positive ratio for the declared component estimates."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = mps_baseline_comparison(K, omega)
        assert result.compression_ratio > 0

    def test_small_system_tractable(self) -> None:
        """The four-site entropy lower bound fits the declared default cap."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = mps_baseline_comparison(K, omega)
        assert result.mps_tractable
        assert result.bond_dimension_basis == "half-chain entropy lower bound"
        assert "not established" in result.qualification_scope

    def test_scpn_mps_comparison(self) -> None:
        """Record MPS baseline for SCPN defaults."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = mps_baseline_comparison(K, omega)
        print("\n  MPS baseline (4 osc):")
        print(f"  S(n/2) = {result.half_chain_entropy:.6f}")
        print(f"  χ entropy lower bound = {result.required_bond_dim}")
        print(f"  MPS memory = {result.mps_memory_bytes} bytes")
        print(f"  Exact memory = {result.exact_memory_bytes} bytes")
        print(f"  Compression = {result.compression_ratio:.1f}x")
        print(f"  Assumed cap crossing = {result.quantum_advantage_threshold} qubits")
        assert isinstance(result.required_bond_dim, int)


def test_entropy_lower_bound_does_not_certify_exact_schmidt_rank() -> None:
    """A rank-four Schmidt distribution can have a bond lower bound of two."""
    probabilities = np.array([0.97, 0.01, 0.01, 0.01])
    partition_amplitudes = np.diag(np.sqrt(probabilities))
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    assert np.linalg.matrix_rank(partition_amplitudes) == 4
    assert required_bond_dimension(entropy) == 2


def test_zero_entropy_slope_retains_heuristic_sentinel() -> None:
    """Keep the documented legacy zero-slope sentinel without promotion."""
    assert quantum_advantage_n(chi_max=256, entropy_per_qubit=0.0) == 1000
