# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for cross-domain validation."""

from __future__ import annotations

from scpn_quantum_control.applications.cross_domain import (
    CrossDomainResult,
    run_cross_domain_validation,
)


class TestCrossDomainValidation:
    def test_returns_result(self):
        result = run_cross_domain_validation()
        assert isinstance(result, CrossDomainResult)

    def test_five_systems(self):
        result = run_cross_domain_validation()
        assert len(result.system_names) == 5
        assert len(result.topology_correlations) == 5
        assert len(result.frequency_correlations) == 5

    def test_correlations_bounded(self):
        result = run_cross_domain_validation()
        for r in result.topology_correlations:
            assert -1 <= r <= 1

    def test_best_system_in_list(self):
        result = run_cross_domain_validation()
        assert result.best_system in result.system_names

    def test_mean_correlation_positive(self):
        result = run_cross_domain_validation()
        assert result.mean_correlation >= 0

    def test_n_above_threshold_bounded(self):
        result = run_cross_domain_validation()
        assert 0 <= result.n_above_threshold <= 5

    def test_gap1_summary(self):
        """Record the Gap 1 cross-domain validation results."""
        result = run_cross_domain_validation()
        print("\n  Cross-domain validation (Gap 1):")
        for name, rho, freq in zip(
            result.system_names,
            result.topology_correlations,
            result.frequency_correlations,
        ):
            print(f"    {name}: ρ={rho:.3f}, freq r={freq:.3f}")
        print(f"  Best: {result.best_system} (ρ={result.best_correlation:.3f})")
        print(f"  Mean |ρ|: {result.mean_correlation:.3f}")
        print(f"  Systems with |ρ| > 0.3: {result.n_above_threshold}/5")
        assert isinstance(result.best_correlation, float)
