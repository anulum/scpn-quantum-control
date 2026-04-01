# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Qsvt Evolution
"""Tests for QSVT Hamiltonian simulation resource estimation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.phase.qsvt_evolution import (
    QSVTResourceEstimate,
    hamiltonian_1norm,
    hamiltonian_spectral_norm,
    qsp_phase_angles,
    qsvt_query_count,
    qsvt_resource_estimate,
    trotter1_step_count,
    trotter2_step_count,
)


class TestHamiltonian1Norm:
    def test_positive(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        alpha = hamiltonian_1norm(K, omega)
        assert alpha > 0

    def test_scales_with_coupling(self):
        omega = OMEGA_N_16[:4]
        a_weak = hamiltonian_1norm(build_knm_paper27(L=4, K_base=0.1), omega)
        a_strong = hamiltonian_1norm(build_knm_paper27(L=4, K_base=1.0), omega)
        assert a_strong > a_weak

    def test_scales_with_n(self):
        a4 = hamiltonian_1norm(build_knm_paper27(L=4), OMEGA_N_16[:4])
        a8 = hamiltonian_1norm(build_knm_paper27(L=8), OMEGA_N_16[:8])
        assert a8 > a4


class TestHamiltonianSpectralNorm:
    def test_positive(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        norm = hamiltonian_spectral_norm(K, omega)
        assert norm > 0

    def test_leq_1norm(self):
        """Spectral norm ≤ 1-norm always."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        spec = hamiltonian_spectral_norm(K, omega)
        alpha = hamiltonian_1norm(K, omega)
        assert spec <= alpha + 1e-10


class TestQueryCounts:
    def test_qsvt_positive(self):
        q = qsvt_query_count(1.0, 1.0, 0.01)
        assert q >= 1

    def test_trotter1_positive(self):
        r = trotter1_step_count(1.0, 1.0, 0.01)
        assert r >= 1

    def test_trotter2_positive(self):
        r = trotter2_step_count(1.0, 1.0, 0.01)
        assert r >= 1

    def test_qsvt_fewer_than_trotter1(self):
        """QSVT should need fewer queries than first-order Trotter."""
        alpha, t, eps = 5.0, 2.0, 0.001
        q = qsvt_query_count(alpha, t, eps)
        r = trotter1_step_count(alpha, t, eps)
        assert q < r

    def test_scales_sublinearly_or_linearly(self):
        """QSVT: O(αt + log(1/ε)). At large t, nearly linear."""
        q10 = qsvt_query_count(1.0, 10.0, 0.01)
        q100 = qsvt_query_count(1.0, 100.0, 0.01)
        ratio = q100 / q10
        assert 5 < ratio < 15  # approximately 10x at large t


class TestQSVTResourceEstimate:
    def test_returns_result(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = qsvt_resource_estimate(K, omega)
        assert isinstance(result, QSVTResourceEstimate)

    def test_speedup_positive(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = qsvt_resource_estimate(K, omega, t=2.0, epsilon=0.001)
        assert result.speedup_vs_trotter1 > 1.0

    def test_n_qubits(self):
        K = build_knm_paper27(L=8)
        omega = OMEGA_N_16[:8]
        result = qsvt_resource_estimate(K, omega)
        assert result.n_qubits == 8

    def test_ancilla_positive(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = qsvt_resource_estimate(K, omega)
        assert result.n_ancilla_qsvt >= 2

    def test_scpn_8_resource(self):
        """Record QSVT resource estimate for 8 oscillators."""
        K = build_knm_paper27(L=8)
        omega = OMEGA_N_16[:8]
        result = qsvt_resource_estimate(K, omega, t=1.0, epsilon=0.01)
        print("\n  QSVT 8-osc resource estimate:")
        print(f"  α (1-norm) = {result.alpha:.4f}")
        print(f"  ||H|| (spectral) = {result.spectral_norm:.4f}")
        print(f"  QSVT queries: {result.qsvt_queries}")
        print(f"  Trotter-1 steps: {result.trotter1_steps}")
        print(f"  Trotter-2 steps: {result.trotter2_steps}")
        print(f"  Speedup vs T1: {result.speedup_vs_trotter1:.1f}x")
        print(f"  Speedup vs T2: {result.speedup_vs_trotter2:.1f}x")
        print(f"  Ancilla qubits: {result.n_ancilla_qsvt}")
        assert result.qsvt_queries > 0


class TestQSPPhaseAngles:
    def test_length(self):
        phases = qsp_phase_angles(10)
        assert len(phases) == 11

    def test_symmetric(self):
        phases = qsp_phase_angles(8)
        assert phases[0] == pytest.approx(phases[-1])
