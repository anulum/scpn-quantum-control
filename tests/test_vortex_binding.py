# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Vortex Binding
"""Tests for vortex binding energy analysis."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.analysis.vortex_binding import (
    VortexBindingResult,
    compute_vortex_binding,
    kosterlitz_rg_step,
    vortex_free_energy,
    vortex_pair_energy,
    vortex_pair_entropy,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


class TestVortexPairEnergy:
    def test_positive(self):
        e = vortex_pair_energy(1.0, 4.0)
        assert e > 0

    def test_scales_with_j(self):
        e1 = vortex_pair_energy(0.5, 4.0)
        e2 = vortex_pair_energy(1.0, 4.0)
        assert e2 == pytest.approx(2 * e1)

    def test_logarithmic_in_size(self):
        e1 = vortex_pair_energy(1.0, 2.0)
        e2 = vortex_pair_energy(1.0, 4.0)
        # E ~ ln(L), so E(4)/E(2) = ln(4)/ln(2) = 2
        assert e2 / e1 == pytest.approx(2.0, rel=0.01)

    def test_zero_at_cutoff(self):
        e = vortex_pair_energy(1.0, 1.0)
        assert e == 0.0


class TestVortexPairEntropy:
    def test_positive(self):
        s = vortex_pair_entropy(4.0)
        assert s > 0

    def test_zero_at_cutoff(self):
        s = vortex_pair_entropy(1.0)
        assert s == 0.0


class TestVortexFreeEnergy:
    def test_zero_at_bkt(self):
        """At T_BKT = pi*J, F should be approximately zero."""
        j = 1.0
        t_bkt = np.pi * j
        f = vortex_free_energy(j, t_bkt, 4.0)
        # F = (2*pi*J - 2*T)*ln(L) = (2*pi*J - 2*pi*J)*ln(L) = 0
        assert f == pytest.approx(0.0, abs=1e-10)

    def test_positive_below_bkt(self):
        """F > 0 below T_BKT (pairs bound)."""
        j = 1.0
        f = vortex_free_energy(j, 0.5, 4.0)
        assert f > 0


class TestKosterlitzRGStep:
    def test_returns_two_values(self):
        k_new, y_new = kosterlitz_rg_step(1.0, 0.1)
        assert isinstance(k_new, float)
        assert isinstance(y_new, float)

    def test_fugacity_shrinks_below_threshold(self):
        """Below BKT (K > 2/pi), fugacity decreases."""
        k_inv = 0.5  # K = 2 > 2/pi, below BKT
        y = 0.01
        _, y_new = kosterlitz_rg_step(k_inv, y, dl=0.1)
        assert y_new < y


class TestComputeVortexBinding:
    def test_returns_result(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = compute_vortex_binding(K, omega)
        assert isinstance(result, VortexBindingResult)

    def test_n_oscillators(self):
        K = build_knm_paper27(L=8)
        omega = OMEGA_N_16[:8]
        result = compute_vortex_binding(K, omega)
        assert result.n_oscillators == 8

    def test_rg_fixed_point(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = compute_vortex_binding(K, omega)
        assert result.rg_fixed_point_k == pytest.approx(2.0 / np.pi)

    def test_free_energy_positive_at_estimated_bkt(self):
        """F > 0 at our estimated T_BKT (mean-field approximation)."""
        K = build_knm_paper27(L=16)
        omega = OMEGA_N_16
        result = compute_vortex_binding(K, omega)
        # Our T_BKT = pi/2 * J < pi * J (exact), so F > 0 (still bound)
        assert result.free_energy_at_bkt >= 0

    def test_scpn_binding(self):
        """Record vortex binding at SCPN defaults."""
        K = build_knm_paper27(L=16)
        omega = OMEGA_N_16
        result = compute_vortex_binding(K, omega)
        print("\n  Vortex binding (16 osc):")
        print(f"  J_eff = {result.j_eff:.6f}")
        print(f"  E_pair = {result.e_pair:.6f}")
        print(f"  T_BKT = {result.t_bkt:.6f}")
        print(f"  E/T ratio = {result.binding_ratio:.4f}")
        print(f"  F(T_BKT) = {result.free_energy_at_bkt:.2e}")
        print(f"  Bound: {result.is_bound}")
        assert isinstance(result.j_eff, float)
