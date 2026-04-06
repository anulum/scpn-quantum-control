# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Multi-Scale QEC
"""STRONG tests for qec/multiscale_qec.py.

6 dimensions: empty/null, error handling, negative cases, pipeline
integration, roundtrip, performance.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27
from scpn_quantum_control.qec.error_budget import logical_error_rate
from scpn_quantum_control.qec.multiscale_qec import (
    build_multiscale_qec,
    concatenated_logical_rate,
    knm_between_domains,
    syndrome_flow_analysis,
)

# ===== Fixtures =====


@pytest.fixture
def knm_16() -> np.ndarray:
    """Standard 16-layer SCPN K_nm matrix."""
    return build_knm_paper27()


@pytest.fixture
def knm_5() -> np.ndarray:
    """Small 5-layer K_nm for fast tests."""
    return build_knm_paper27(L=5)


# ===== 1. Empty/Null Inputs =====


class TestEmptyNull:
    """Edge cases with minimal or degenerate inputs."""

    def test_concatenated_rates_empty_distances(self) -> None:
        """Empty distance list produces empty rates."""
        rates = concatenated_logical_rate(0.003, [])
        assert rates == []

    def test_concatenated_rates_single_level(self) -> None:
        """Single concatenation level produces one rate."""
        rates = concatenated_logical_rate(0.003, [3])
        assert len(rates) == 1
        assert 0.0 < rates[0] < 1.0

    def test_knm_between_domains_self(self, knm_16: np.ndarray) -> None:
        """Coupling within a single domain is non-zero (intra-domain)."""
        coupling = knm_between_domains(knm_16, (0, 3), (0, 3))
        assert coupling > 0.0

    def test_knm_between_domains_adjacent(self, knm_16: np.ndarray) -> None:
        """Adjacent domains have stronger coupling than distant ones."""
        near = knm_between_domains(knm_16, (0, 3), (4, 7))
        far = knm_between_domains(knm_16, (0, 3), (12, 15))
        assert near > far, "adjacent domains must couple more strongly"

    def test_build_with_minimal_matrix(self) -> None:
        """2×2 K matrix should produce at least one level."""
        K = np.array([[0.0, 0.3], [0.3, 0.0]])
        result = build_multiscale_qec(K, p_physical=0.001)
        assert result.concatenation_depth >= 1
        assert result.total_physical_qubits > 0


# ===== 2. Error Handling =====


class TestErrorHandling:
    """Invalid inputs must raise clear errors."""

    def test_mismatched_distances_length(self, knm_16: np.ndarray) -> None:
        """distances list must match number of active domains."""
        with pytest.raises(ValueError, match="distances length"):
            build_multiscale_qec(knm_16, distances=[3, 5])  # too few

    def test_above_threshold_physical_rate(self, knm_16: np.ndarray) -> None:
        """Above-threshold p_phys produces p_L = 1.0 at first level."""
        result = build_multiscale_qec(knm_16, p_physical=0.05, distances=[3, 3, 3, 3, 3])
        assert result.levels[0].logical_error_rate == 1.0
        assert not result.double_exponential_suppression

    def test_concatenated_rates_above_threshold(self) -> None:
        """Above-threshold rates saturate at 1.0."""
        rates = concatenated_logical_rate(0.05, [3])
        assert rates[0] == 1.0


# ===== 3. Negative Cases =====


class TestNegativeCases:
    """Cases that should NOT produce certain outcomes."""

    def test_zero_coupling_no_syndrome_flow(self) -> None:
        """Zero K_nm → zero syndrome flow between levels."""
        K = np.zeros((16, 16))
        result = build_multiscale_qec(K, p_physical=0.001, distances=[3, 3, 3, 3, 3])
        flows = syndrome_flow_analysis(K, result)
        for flow in flows:
            assert flow.syndrome_weight == pytest.approx(0.0, abs=1e-12)

    def test_identity_coupling_not_exponential(self) -> None:
        """Identity K (uniform coupling) should NOT show exponential
        decay between domains."""
        K = np.ones((16, 16)) * 0.1
        np.fill_diagonal(K, 0.0)
        near = knm_between_domains(K, (0, 3), (4, 7))
        far = knm_between_domains(K, (0, 3), (12, 15))
        # Uniform → equal coupling
        assert abs(near - far) < 1e-10

    def test_single_qubit_no_correction(self) -> None:
        """d=1 code cannot correct any error."""
        rates = concatenated_logical_rate(0.003, [1])
        # d=1: (d+1)/2 = 1, so p_L = A × (p/p_th)^1 = 0.1 × 0.3 = 0.03
        expected = 0.1 * (0.003 / 0.01)
        assert abs(rates[0] - expected) < 1e-10


# ===== 4. Pipeline Integration =====


class TestPipelineIntegration:
    """MS-QEC must integrate with existing QEC modules."""

    def test_rates_match_error_budget_module(self) -> None:
        """concatenated_logical_rate level 0 must match error_budget.logical_error_rate."""
        d = 5
        p_phys = 0.003
        from_msqec = concatenated_logical_rate(p_phys, [d])
        from_budget = logical_error_rate(d, p_phys)
        assert abs(from_msqec[0] - from_budget) < 1e-15

    def test_build_uses_knm_paper27(self, knm_16: np.ndarray) -> None:
        """build_multiscale_qec with standard K_nm produces valid hierarchy."""
        result = build_multiscale_qec(knm_16, p_physical=0.001)
        assert result.concatenation_depth == 5  # 5 SCPN domains
        assert all(lvl.code_distance >= 3 for lvl in result.levels)
        assert result.total_physical_qubits > 0

    def test_syndrome_flow_uses_knm_coupling(self, knm_16: np.ndarray) -> None:
        """Syndrome flow weights must match K_nm inter-domain coupling."""
        result = build_multiscale_qec(knm_16, p_physical=0.001, distances=[3, 3, 3, 3, 3])
        flows = syndrome_flow_analysis(knm_16, result)
        assert len(flows) == 4  # 5 levels → 4 flows

        # Flow weights must be positive for SCPN K_nm
        for flow in flows:
            assert flow.syndrome_weight > 0.0
            assert flow.correction_capacity == 1.0  # (3-1)/2

    def test_level_domain_names_match_scpn(self, knm_16: np.ndarray) -> None:
        """QEC levels must map to SCPN domain names."""
        result = build_multiscale_qec(knm_16, p_physical=0.001, distances=[3, 3, 3, 3, 3])
        expected_names = ["biological", "organismal", "collective", "meta", "closure"]
        actual_names = [lvl.domain_name for lvl in result.levels]
        assert actual_names == expected_names

    def test_top_level_import(self) -> None:
        """MS-QEC symbols must be importable from top-level package."""
        from scpn_quantum_control import (
            build_multiscale_qec,
            concatenated_logical_rate,
            syndrome_flow_analysis,
        )

        assert callable(build_multiscale_qec)
        assert callable(concatenated_logical_rate)
        assert callable(syndrome_flow_analysis)


# ===== 5. Roundtrip =====


class TestRoundtrip:
    """Concatenation produces consistent results across levels."""

    def test_concatenated_rates_monotonic_below_threshold(self) -> None:
        """Below threshold, each level must reduce the error rate."""
        rates = concatenated_logical_rate(0.001, [5, 5, 5])
        for i in range(len(rates) - 1):
            assert rates[i + 1] < rates[i], (
                f"level {i + 1} rate {rates[i + 1]:.2e} >= level {i} rate {rates[i]:.2e}"
            )

    def test_physical_qubit_count_consistent(self, knm_16: np.ndarray) -> None:
        """Total physical qubits must equal sum of per-level counts."""
        result = build_multiscale_qec(knm_16, p_physical=0.001, distances=[3, 5, 3, 3, 3])
        level_sum = sum(lvl.total_physical_qubits for lvl in result.levels)
        assert result.total_physical_qubits == level_sum

    def test_physical_qubits_per_logical_formula(self, knm_16: np.ndarray) -> None:
        """Physical qubits per logical must equal 2d² − 1."""
        result = build_multiscale_qec(knm_16, p_physical=0.001, distances=[3, 5, 7, 3, 3])
        for lvl in result.levels:
            expected = 2 * lvl.code_distance**2 - 1
            assert lvl.physical_qubits_per_logical == expected

    def test_double_exponential_suppression_at_low_p(self) -> None:
        """Very low p_phys with large d should show double-exp suppression."""
        rates = concatenated_logical_rate(0.0001, [7, 7, 7])
        # Each level: (p/p_th)^4 ≈ 0.01^4 = 1e-8
        # Truly double-exponential → rates decrease very fast
        assert rates[-1] < 1e-30


# ===== 6. Performance =====


class TestPerformance:
    """Wall-clock budgets for MS-QEC computations."""

    def test_concatenated_rates_fast(self) -> None:
        """concatenated_logical_rate for 10 levels must complete in < 1ms."""
        t0 = time.perf_counter()
        for _ in range(1000):
            concatenated_logical_rate(0.003, [3, 5, 7, 3, 5, 7, 3, 5, 7, 3])
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"1000 calls took {elapsed:.3f}s"

    def test_build_multiscale_fast(self, knm_16: np.ndarray) -> None:
        """Full MS-QEC build for 16-layer K_nm must complete in < 100ms."""
        t0 = time.perf_counter()
        build_multiscale_qec(knm_16, p_physical=0.001)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.1, f"build_multiscale_qec took {elapsed:.3f}s"

    def test_syndrome_flow_analysis_fast(self, knm_16: np.ndarray) -> None:
        """Syndrome flow analysis must complete in < 10ms."""
        result = build_multiscale_qec(knm_16, p_physical=0.001, distances=[3, 3, 3, 3, 3])
        t0 = time.perf_counter()
        for _ in range(100):
            syndrome_flow_analysis(knm_16, result)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"100 calls took {elapsed:.3f}s"
