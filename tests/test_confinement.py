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

import scpn_quantum_control.gauge.confinement as confinement
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.gauge.confinement import (
    ConfinementResult,
    _average_wilson_by_length,
    confinement_analysis,
    confinement_vs_coupling,
    extract_string_tension,
)


class TestExtractStringTension:
    """Extracting the string tension from Wilson loops."""

    def test_equal_wilson_zero_tension(self) -> None:
        """Equal Wilson loops give zero tension."""
        sigma = extract_string_tension(0.5, 0.5)
        assert sigma is not None
        assert sigma == pytest.approx(0.0, abs=1e-10)

    def test_decaying_wilson_positive_tension(self) -> None:
        """Decaying Wilson loops give a positive tension."""
        sigma = extract_string_tension(0.8, 0.5)
        assert sigma is not None
        assert sigma > 0

    def test_none_for_zero_wilson(self) -> None:
        """A zero Wilson loop gives no tension."""
        sigma = extract_string_tension(0.0, 0.5)
        assert sigma is None

    def test_none_for_equal_areas(self) -> None:
        """Equal areas give no tension."""
        sigma = extract_string_tension(0.5, 0.3, area_small=1.0, area_large=1.0)
        assert sigma is None

    def test_none_for_zero_ratio(self) -> None:
        """A zero ratio gives no tension."""
        sigma = extract_string_tension(np.inf, 0.5)
        assert sigma is None


class TestAverageWilsonByLength:
    """Averaging Wilson loops by length."""

    def test_zero_when_requested_length_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A length with no loops averages to zero."""

        class Loop:
            loop_length = 2
            magnitude = 0.75

        monkeypatch.setattr(
            confinement, "compute_wilson_loops", lambda *_args, **_kwargs: [Loop()]
        )
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        assert _average_wilson_by_length(K, omega, 4) == 0.0


class TestConfinementAnalysis:
    """The confinement analysis."""

    def test_returns_result(self) -> None:
        """The analysis returns a result record."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = confinement_analysis(K, omega)
        assert isinstance(result, ConfinementResult)

    def test_n_qubits(self) -> None:
        """The result records the qubit count."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = confinement_analysis(K, omega)
        assert result.n_qubits == 4

    def test_wilson_averages_non_negative(self) -> None:
        """The triangle and square averages are non-negative."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = confinement_analysis(K, omega)
        assert result.wilson_triangle_avg >= 0
        assert result.wilson_square_avg >= 0

    def test_confinement_ratio_type(self) -> None:
        """The confinement ratio is a float."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = confinement_analysis(K, omega)
        assert isinstance(result.confinement_ratio, float)

    def test_is_confined_bool(self) -> None:
        """The confinement flag is a boolean."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = confinement_analysis(K, omega)
        assert isinstance(result.is_confined, bool)

    def test_scpn_confinement(self) -> None:
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
    """Scanning confinement across couplings."""

    def test_scan_returns_keys(self) -> None:
        """Each series has one value per coupling."""
        omega = OMEGA_N_16[:4]
        k_vals = np.array([0.1, 0.5, 1.0])
        results = confinement_vs_coupling(omega, k_vals)
        for key in ["k_base", "string_tension", "wilson_triangle", "wilson_square"]:
            assert key in results
            assert len(results[key]) == 3

    def test_string_tension_non_negative(self) -> None:
        """Every string tension is non-negative."""
        omega = OMEGA_N_16[:4]
        k_vals = np.array([0.5, 1.0])
        results = confinement_vs_coupling(omega, k_vals)
        for st in results["string_tension"]:
            assert st >= 0

    def test_default_scan_uses_fifteen_points(self) -> None:
        """The default scan uses 15 couplings from 0.1 to 3.0."""
        omega = OMEGA_N_16[:4]
        results = confinement_vs_coupling(omega)
        assert len(results["k_base"]) == 15
        assert results["k_base"][0] == pytest.approx(0.1)
        assert results["k_base"][-1] == pytest.approx(3.0)
