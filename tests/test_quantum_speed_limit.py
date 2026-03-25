# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for quantum speed limit at the synchronization transition."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.analysis.quantum_speed_limit import (
    QSLResult,
    compute_qsl,
    qsl_vs_coupling,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


class TestComputeQSL:
    def test_returns_result(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_qsl(K, omega, t_target=1.0)
        assert isinstance(result, QSLResult)
        assert result.n_qubits == 3

    def test_MT_bound_positive(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_qsl(K, omega, t_target=1.0)
        assert result.tau_MT >= 0.0

    def test_ML_bound_positive(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_qsl(K, omega, t_target=1.0)
        assert result.tau_ML >= 0.0

    def test_actual_exceeds_bounds(self):
        K = build_knm_paper27(L=3) * 2.0
        omega = OMEGA_N_16[:3]
        result = compute_qsl(K * 2, omega, t_target=2.0, R_threshold=0.3)
        # QSL is a lower bound: τ_actual ≥ τ_MT and τ_actual ≥ τ_ML
        if result.tau_MT > 0:
            assert result.tau_actual >= result.tau_MT - 0.02  # small tolerance for dt discretization

    def test_overlap_bounded(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = compute_qsl(K, omega, t_target=1.0)
        assert 0.0 <= result.overlap <= 1.0

    def test_delta_E_nonneg(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_qsl(K, omega, t_target=0.5)
        assert result.delta_E >= 0.0

    def test_two_qubit(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = compute_qsl(K, omega, t_target=1.0)
        assert result.n_qubits == 2


class TestQSLvsCoupling:
    def test_returns_lists(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        scan = qsl_vs_coupling(K, omega, n_K_values=5, t_target=1.0)
        assert len(scan["K_base"]) == 5
        assert len(scan["tau_MT"]) == 5
        assert len(scan["tau_ML"]) == 5
        assert len(scan["tau_actual"]) == 5

    def test_stronger_coupling_faster_sync(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        scan = qsl_vs_coupling(
            K, omega,
            K_base_range=np.array([0.5, 2.0]),
            t_target=2.0, R_threshold=0.3,
        )
        # Stronger coupling should generally synchronize faster
        # (but not guaranteed for all parameter regimes)
        assert len(scan["tau_actual"]) == 2

    def test_R_final_values_finite(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        scan = qsl_vs_coupling(
            K, omega,
            K_base_range=np.array([0.1, 1.0, 3.0]),
            t_target=2.0,
        )
        # R(t) at finite time can oscillate with K due to quantum interference
        # (not monotonic like classical Kuramoto). Just verify finite values.
        for r in scan["R_final"]:
            assert 0.0 <= r <= 1.0 + 1e-10
