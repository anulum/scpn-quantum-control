# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Finite Size Scaling
"""Tests for finite-size scaling K_c extraction."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.analysis.finite_size_scaling import (
    FSSResult,
    finite_size_scaling,
)


class TestFiniteSizeScaling:
    def test_returns_result(self):
        result = finite_size_scaling(
            system_sizes=[2, 3],
            k_range=np.linspace(0.5, 4.0, 10),
        )
        assert isinstance(result, FSSResult)
        assert len(result.k_c_values) == 2

    def test_k_c_positive(self):
        result = finite_size_scaling(
            system_sizes=[2, 3, 4],
            k_range=np.linspace(0.5, 5.0, 12),
        )
        for kc in result.k_c_values:
            assert kc > 0

    def test_gap_min_positive(self):
        result = finite_size_scaling(
            system_sizes=[2, 3],
            k_range=np.linspace(0.5, 4.0, 10),
        )
        for g in result.gap_min_values:
            assert g > 0

    def test_extrapolation_exists(self):
        result = finite_size_scaling(
            system_sizes=[2, 3, 4],
            k_range=np.linspace(0.5, 5.0, 12),
        )
        # At least one extrapolation should succeed
        assert result.k_c_extrapolated_bkt is not None or result.k_c_extrapolated_power is not None

    def test_extrapolated_values_finite(self):
        result = finite_size_scaling(
            system_sizes=[2, 3, 4],
            k_range=np.linspace(0.5, 5.0, 12),
        )
        if result.k_c_extrapolated_bkt is not None:
            assert np.isfinite(result.k_c_extrapolated_bkt)
        if result.k_c_extrapolated_power is not None:
            assert np.isfinite(result.k_c_extrapolated_power)

    def test_single_size(self):
        """Single system size → no extrapolation possible."""
        result = finite_size_scaling(
            system_sizes=[3],
            k_range=np.linspace(0.5, 4.0, 8),
        )
        assert len(result.k_c_values) == 1
