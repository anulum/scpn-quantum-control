# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for quantum Kuramoto phase diagram."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.analysis.phase_diagram import (
    PhaseDiagramResult,
    compute_phase_diagram,
    critical_coupling_finite_graph,
    critical_coupling_mean_field,
    decoherence_temperature,
    effective_temperature,
    order_parameter_steady_state,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


class TestCriticalCoupling:
    def test_finite_graph_positive(self):
        omega = OMEGA_N_16
        K = build_knm_paper27(L=16)
        from scpn_quantum_control.analysis.bkt_analysis import fiedler_eigenvalue

        fiedler = fiedler_eigenvalue(K)
        k_c = critical_coupling_finite_graph(omega, fiedler)
        assert k_c > 0

    def test_finite_graph_scales_with_frequency_spread(self):
        fiedler = 1.0
        omega_narrow = np.array([1.0, 1.01, 1.02])
        omega_wide = np.array([0.5, 1.0, 1.5])
        k_c_narrow = critical_coupling_finite_graph(omega_narrow, fiedler)
        k_c_wide = critical_coupling_finite_graph(omega_wide, fiedler)
        assert k_c_wide > k_c_narrow

    def test_finite_graph_inversely_scales_with_fiedler(self):
        omega = np.array([0.5, 1.0, 1.5])
        k_c_low = critical_coupling_finite_graph(omega, fiedler=0.5)
        k_c_high = critical_coupling_finite_graph(omega, fiedler=2.0)
        assert k_c_low > k_c_high

    def test_mean_field_positive(self):
        k_c = critical_coupling_mean_field(OMEGA_N_16)
        assert k_c > 0

    def test_mean_field_identical_frequencies(self):
        omega = np.ones(10)
        k_c = critical_coupling_mean_field(omega)
        assert k_c == 0.0

    def test_mean_field_single_oscillator(self):
        k_c = critical_coupling_mean_field(np.array([1.0]))
        assert k_c == 0.0


class TestDecoherenceTemperature:
    def test_infinite_t1_t2_zero_temperature(self):
        t_dec = decoherence_temperature(np.inf, np.inf)
        assert t_dec == pytest.approx(0.0, abs=1e-10)

    def test_finite_t2_positive_temperature(self):
        t_dec = decoherence_temperature(t1=200.0, t2=100.0)
        assert t_dec > 0

    def test_shorter_t2_higher_temperature(self):
        t_long = decoherence_temperature(t1=200.0, t2=100.0)
        t_short = decoherence_temperature(t1=200.0, t2=10.0)
        assert t_short > t_long


class TestEffectiveTemperature:
    def test_classical_only(self):
        omega = np.array([0.5, 1.0, 1.5])
        t_eff = effective_temperature(omega)
        assert t_eff == pytest.approx(np.std(omega), abs=1e-10)

    def test_decoherence_increases_temperature(self):
        omega = OMEGA_N_16
        t_classical = effective_temperature(omega)
        t_quantum = effective_temperature(omega, t1=200.0, t2=100.0)
        assert t_quantum > t_classical


class TestOrderParameter:
    def test_below_critical_zero(self):
        R = order_parameter_steady_state(K_coupling=0.5, k_critical=1.0)
        assert R == 0.0

    def test_at_critical_zero(self):
        R = order_parameter_steady_state(K_coupling=1.0, k_critical=1.0)
        assert R == 0.0

    def test_above_critical_positive(self):
        R = order_parameter_steady_state(K_coupling=2.0, k_critical=1.0)
        assert R > 0

    def test_far_above_critical_near_one(self):
        R = order_parameter_steady_state(K_coupling=100.0, k_critical=1.0)
        assert R > 0.9

    def test_mean_field_formula(self):
        """R = sqrt(1 - K_c/K) for K > K_c."""
        K, K_c = 4.0, 1.0
        R = order_parameter_steady_state(K, K_c)
        expected = np.sqrt(1.0 - K_c / K)
        assert pytest.approx(expected, abs=1e-12) == R

    def test_bounded_zero_one(self):
        for k in np.linspace(0.1, 10.0, 20):
            R = order_parameter_steady_state(k, k_critical=1.0)
            assert 0 <= R <= 1.0


class TestComputePhaseDiagram:
    def test_returns_result(self):
        K = build_knm_paper27(L=8)
        omega = OMEGA_N_16[:8]
        result = compute_phase_diagram(K, omega, n_k=10, n_t=8)
        assert isinstance(result, PhaseDiagramResult)

    def test_shapes(self):
        n_k, n_t = 15, 12
        K = build_knm_paper27(L=8)
        omega = OMEGA_N_16[:8]
        result = compute_phase_diagram(K, omega, n_k=n_k, n_t=n_t)
        assert result.k_values.shape == (n_k,)
        assert result.t_eff_values.shape == (n_t,)
        assert result.order_parameter.shape == (n_k, n_t)
        assert result.k_critical_curve.shape == (n_t,)

    def test_order_parameter_bounded(self):
        K = build_knm_paper27(L=8)
        omega = OMEGA_N_16[:8]
        result = compute_phase_diagram(K, omega, n_k=10, n_t=8)
        assert np.all(result.order_parameter >= 0)
        assert np.all(result.order_parameter <= 1)

    def test_bkt_temperature_positive(self):
        K = build_knm_paper27(L=16)
        omega = OMEGA_N_16
        result = compute_phase_diagram(K, omega, n_k=10, n_t=8)
        assert result.bkt_temperature > 0

    def test_quantum_kc_exceeds_classical(self):
        """Decoherence increases critical coupling."""
        K = build_knm_paper27(L=16)
        omega = OMEGA_N_16
        result = compute_phase_diagram(K, omega, n_k=10, n_t=8)
        assert result.quantum_k_c >= result.classical_k_c

    def test_scpn_default_phase_diagram(self):
        """Record actual SCPN phase diagram values — this is the finding."""
        K = build_knm_paper27(L=16)
        omega = OMEGA_N_16
        result = compute_phase_diagram(K, omega, n_k=20, n_t=15)
        print(f"\n  Classical K_c = {result.classical_k_c:.4f}")
        print(f"  Quantum K_c (T2=100μs) = {result.quantum_k_c:.4f}")
        print(f"  T_BKT = {result.bkt_temperature:.6f}")
        print(f"  K_c ratio (quantum/classical) = {result.quantum_k_c / result.classical_k_c:.3f}")
        assert isinstance(result.classical_k_c, float)
