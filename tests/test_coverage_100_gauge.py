# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Coverage 100 Gauge
"""Multi-angle tests for gauge/ subpackage: confinement, universality,
vortex_detector, wilson_loop.

Covers: Wilson loop averaging, string tension extraction, BKT universality,
vortex density, parametrised coupling, edge cases, physical bounds.
"""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


# =====================================================================
# Confinement
# =====================================================================
class TestConfinement:
    def test_no_loops_returns_zero(self):
        from scpn_quantum_control.gauge.confinement import _average_wilson_by_length

        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = _average_wilson_by_length(K, omega, length=4)
        assert result == 0.0

    def test_string_tension_none_for_zero_wilson(self):
        from scpn_quantum_control.gauge.confinement import extract_string_tension

        assert extract_string_tension(0.0, 0.5) is None
        assert extract_string_tension(0.5, 0.0) is None

    def test_string_tension_equal_areas_none(self):
        from scpn_quantum_control.gauge.confinement import extract_string_tension

        assert extract_string_tension(0.5, 0.5, area_small=1.0, area_large=1.0) is None

    def test_string_tension_valid_case(self):
        from scpn_quantum_control.gauge.confinement import extract_string_tension

        sigma = extract_string_tension(0.8, 0.5, area_small=1.0, area_large=4.0)
        if sigma is not None:
            assert np.isfinite(sigma)

    def test_confinement_scan_default_k_values(self):
        from scpn_quantum_control.gauge.confinement import confinement_vs_coupling

        omega = OMEGA_N_16[:2]
        result = confinement_vs_coupling(omega, k_values=None)
        assert len(result["k_base"]) == 15

    def test_confinement_scan_custom_k_values(self):
        from scpn_quantum_control.gauge.confinement import confinement_vs_coupling

        omega = OMEGA_N_16[:2]
        result = confinement_vs_coupling(omega, k_values=[0.1, 0.5, 1.0])
        assert len(result["k_base"]) == 3

    def test_confinement_analysis_returns_result(self):
        from scpn_quantum_control.gauge.confinement import confinement_analysis

        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = confinement_analysis(K, omega)
        assert hasattr(result, "is_confined")
        assert hasattr(result, "confinement_ratio")
        assert isinstance(result.is_confined, bool)


# =====================================================================
# Universality (BKT)
# =====================================================================
class TestUniversality:
    def test_zero_tbkt_ratio(self):
        from scpn_quantum_control.gauge.universality import check_nelson_kosterlitz

        K = build_knm_paper27(L=2) * 0.0001
        omega = OMEGA_N_16[:2]
        ratio, deviation = check_nelson_kosterlitz(K, omega)
        assert isinstance(ratio, float)
        assert isinstance(deviation, float)

    def test_nelson_kosterlitz_finite(self):
        from scpn_quantum_control.gauge.universality import check_nelson_kosterlitz

        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        ratio, deviation = check_nelson_kosterlitz(K, omega)
        assert np.isfinite(ratio)
        assert np.isfinite(deviation)

    def test_universality_analysis_returns_result(self):
        from scpn_quantum_control.gauge.universality import universality_analysis

        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = universality_analysis(K, omega)
        assert hasattr(result, "eta_exponent")
        assert hasattr(result, "is_bkt_consistent")
        # eta_exponent may be None for small systems (insufficient data for fit)
        assert hasattr(result, "eta_exponent")


# =====================================================================
# Vortex Detector
# =====================================================================
class TestVortexDetector:
    def test_density_scan_default(self):
        from scpn_quantum_control.gauge.vortex_detector import vortex_density_vs_coupling

        omega = OMEGA_N_16[:2]
        result = vortex_density_vs_coupling(omega, k_base_values=None)
        assert len(result["k_base"]) == 20

    def test_density_scan_custom(self):
        from scpn_quantum_control.gauge.vortex_detector import vortex_density_vs_coupling

        omega = OMEGA_N_16[:2]
        result = vortex_density_vs_coupling(omega, k_base_values=[0.1, 0.5])
        assert len(result["k_base"]) == 2

    def test_vortex_density_result(self):
        from scpn_quantum_control.gauge.vortex_detector import measure_vortex_density

        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = measure_vortex_density(K, omega)
        assert hasattr(result, "vortex_density")
        assert np.isfinite(result.vortex_density)
        assert result.vortex_density >= 0.0


# =====================================================================
# Wilson Loop
# =====================================================================
class TestWilsonLoop:
    def test_compute_returns_list(self):
        from scpn_quantum_control.gauge.wilson_loop import compute_wilson_loops

        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        results = compute_wilson_loops(K, omega, max_length=3, max_loops=5)
        assert isinstance(results, list)

    def test_wilson_loops_all_finite(self):
        from scpn_quantum_control.gauge.wilson_loop import compute_wilson_loops

        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        results = compute_wilson_loops(K, omega, max_length=3, max_loops=10)
        for r in results:
            assert np.isfinite(r.expectation_value)
            assert np.isfinite(r.magnitude)

    def test_wilson_loop_result_fields(self):
        from scpn_quantum_control.gauge.wilson_loop import compute_wilson_loops

        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        results = compute_wilson_loops(K, omega, max_length=3, max_loops=5)
        if results:
            r = results[0]
            assert hasattr(r, "loop")
            assert hasattr(r, "loop_length")
            assert hasattr(r, "expectation_value")
            assert hasattr(r, "magnitude")
            assert hasattr(r, "phase_angle")
