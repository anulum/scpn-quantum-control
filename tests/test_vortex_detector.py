# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Vortex Detector
"""Tests for vortex density measurement."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.gauge.vortex_detector import (
    VortexResult,
    _angle_diff,
    _find_plaquettes,
    measure_vortex_density,
    plaquette_vorticity,
    vortex_density_vs_coupling,
)


class TestAngleDiff:
    def test_zero_diff(self):
        assert _angle_diff(0.0, 0.0) == pytest.approx(0.0)

    def test_small_positive(self):
        d = _angle_diff(0.0, 0.1)
        assert d == pytest.approx(0.1, abs=1e-10)

    def test_wrapping(self):
        """3.0 - (-3.0) = 6.0, but wrapped should be ≈ 6.0 - 2π ≈ -0.28."""
        d = _angle_diff(-3.0, 3.0)
        assert abs(d) < np.pi

    def test_range(self):
        for _ in range(100):
            a, b = np.random.uniform(-np.pi, np.pi, 2)
            d = _angle_diff(a, b)
            assert -np.pi <= d <= np.pi


class TestFindPlaquettes:
    def test_complete_graph_has_plaquettes(self):
        K = build_knm_paper27(L=4)
        plaquettes = _find_plaquettes(K)
        assert len(plaquettes) > 0

    def test_plaquettes_are_triangles(self):
        K = build_knm_paper27(L=4)
        plaquettes = _find_plaquettes(K)
        for p in plaquettes:
            assert len(p) == 3

    def test_disconnected_no_plaquettes(self):
        K = np.zeros((4, 4))
        plaquettes = _find_plaquettes(K)
        assert len(plaquettes) == 0

    def test_n4_complete_has_4_triangles(self):
        """C(4,3) = 4 triangles on complete graph."""
        K = build_knm_paper27(L=4)
        plaquettes = _find_plaquettes(K)
        assert len(plaquettes) == 4


class TestPlaquetteVorticity:
    def test_aligned_phases_zero_vorticity(self):
        phases = np.array([0.0, 0.1, 0.2, 0.3])
        v = plaquette_vorticity(phases, [0, 1, 2])
        assert v == 0

    def test_winding_phases(self):
        """Phase winds by 2π around plaquette → vorticity = 1."""
        phases = np.array([0.0, 2 * np.pi / 3, -2 * np.pi / 3])
        v = plaquette_vorticity(phases, [0, 1, 2])
        assert v == 1 or v == -1  # direction depends on convention

    def test_integer_valued(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            phases = rng.uniform(-np.pi, np.pi, 4)
            v = plaquette_vorticity(phases, [0, 1, 2])
            assert isinstance(v, int)


class TestMeasureVortexDensity:
    def test_returns_result(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = measure_vortex_density(K, omega)
        assert isinstance(result, VortexResult)

    def test_density_bounded(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = measure_vortex_density(K, omega)
        assert 0 <= result.vortex_density <= 1.0

    def test_phases_shape(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = measure_vortex_density(K, omega)
        assert result.phases.shape == (4,)

    def test_net_charge_integer(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = measure_vortex_density(K, omega)
        assert isinstance(result.net_charge, int)

    def test_n_plaquettes_matches(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = measure_vortex_density(K, omega)
        assert result.n_plaquettes == 4  # C(4,3) = 4
        assert len(result.plaquette_vorticities) == 4

    def test_scpn_default_vortices(self):
        """Record vortex density at SCPN default parameters."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = measure_vortex_density(K, omega)
        print(f"\n  Vortex density (4 osc, default K): {result.vortex_density:.4f}")
        print(f"  Vortices: {result.n_vortices}, Antivortices: {result.n_antivortices}")
        print(f"  Net charge: {result.net_charge}")
        print(f"  Phases: {result.phases}")
        assert isinstance(result.vortex_density, float)


class TestVortexDensityVsCoupling:
    def test_scan_returns_keys(self):
        omega = OMEGA_N_16[:4]
        k_vals = np.array([0.1, 0.5, 1.0])
        results = vortex_density_vs_coupling(omega, k_vals)
        assert "k_base" in results
        assert "vortex_density" in results
        assert len(results["k_base"]) == 3

    def test_density_non_negative(self):
        omega = OMEGA_N_16[:4]
        k_vals = np.array([0.1, 1.0, 3.0])
        results = vortex_density_vs_coupling(omega, k_vals)
        for d in results["vortex_density"]:
            assert d >= 0
