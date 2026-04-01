# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Confinement
"""Tests for confinement-deconfinement transition."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.gauge.confinement import (
    ConfinementResult,
    confinement_analysis,
    confinement_vs_coupling,
    extract_string_tension,
)


class TestExtractStringTension:
    def test_equal_wilson_zero_tension(self):
        sigma = extract_string_tension(0.5, 0.5)
        assert sigma is not None
        assert sigma == pytest.approx(0.0, abs=1e-10)

    def test_decaying_wilson_positive_tension(self):
        sigma = extract_string_tension(0.8, 0.5)
        assert sigma is not None
        assert sigma > 0

    def test_none_for_zero_wilson(self):
        sigma = extract_string_tension(0.0, 0.5)
        assert sigma is None

    def test_none_for_equal_areas(self):
        sigma = extract_string_tension(0.5, 0.3, area_small=1.0, area_large=1.0)
        assert sigma is None


class TestConfinementAnalysis:
    def test_returns_result(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = confinement_analysis(K, omega)
        assert isinstance(result, ConfinementResult)

    def test_n_qubits(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = confinement_analysis(K, omega)
        assert result.n_qubits == 4

    def test_wilson_averages_non_negative(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = confinement_analysis(K, omega)
        assert result.wilson_triangle_avg >= 0
        assert result.wilson_square_avg >= 0

    def test_confinement_ratio_type(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = confinement_analysis(K, omega)
        assert isinstance(result.confinement_ratio, float)

    def test_is_confined_bool(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = confinement_analysis(K, omega)
        assert isinstance(result.is_confined, bool)

    def test_scpn_confinement(self):
        """Record confinement at SCPN defaults."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = confinement_analysis(K, omega)
        print(f"\n  String tension: {result.string_tension}")
        print(f"  Confined: {result.is_confined}")
        print(f"  W_triangle: {result.wilson_triangle_avg:.4f}")
        print(f"  W_square: {result.wilson_square_avg:.4f}")
        print(f"  Ratio: {result.confinement_ratio:.4f}")
        assert isinstance(result.string_tension, (float, type(None)))


class TestConfinementVsCoupling:
    def test_scan_returns_keys(self):
        omega = OMEGA_N_16[:4]
        k_vals = np.array([0.1, 0.5, 1.0])
        results = confinement_vs_coupling(omega, k_vals)
        for key in ["k_base", "string_tension", "wilson_triangle", "wilson_square"]:
            assert key in results
            assert len(results[key]) == 3

    def test_string_tension_non_negative(self):
        omega = OMEGA_N_16[:4]
        k_vals = np.array([0.5, 1.0])
        results = confinement_vs_coupling(omega, k_vals)
        for st in results["string_tension"]:
            assert st >= 0
