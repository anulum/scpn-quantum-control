# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for L16 Director
"""Tests for quantum L16 director."""

from __future__ import annotations

import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.l16.quantum_director import (
    L16Result,
    compute_l16_lyapunov,
    energy_variance,
    fidelity_susceptibility,
    loschmidt_echo,
)


class TestLoschmidtEcho:
    def test_at_t0_is_one(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        le = loschmidt_echo(K, omega, t=0.0)
        assert le == pytest.approx(1.0, abs=1e-10)

    def test_bounded(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        le = loschmidt_echo(K, omega, t=1.0)
        assert 0 <= le <= 1.0 + 1e-10

    def test_ground_state_is_eigenstate(self):
        """Ground state echo should stay near 1 (eigenstate doesn't evolve)."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        le = loschmidt_echo(K, omega, t=0.5)
        assert le > 0.99


class TestEnergyVariance:
    def test_ground_state_near_zero(self):
        """Exact ground state has ΔE² ≈ 0."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        var = energy_variance(K, omega)
        assert var < 1e-10

    def test_non_negative(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        var = energy_variance(K, omega)
        assert var >= -1e-10


class TestFidelitySusceptibility:
    def test_non_negative(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        fs = fidelity_susceptibility(K, omega)
        assert fs >= 0

    def test_finite(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        fs = fidelity_susceptibility(K, omega)
        assert fs < 1e6


class TestComputeL16Lyapunov:
    def test_returns_result(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_l16_lyapunov(K, omega)
        assert isinstance(result, L16Result)

    def test_stability_score_bounded(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_l16_lyapunov(K, omega)
        assert 0 <= result.stability_score <= 1.0

    def test_action_valid(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_l16_lyapunov(K, omega)
        assert result.action in ("continue", "adjust", "halt")

    def test_r_global_bounded(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_l16_lyapunov(K, omega)
        assert 0 <= result.order_parameter <= 1.0

    def test_scpn_l16(self):
        """Record L16 assessment at SCPN defaults."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = compute_l16_lyapunov(K, omega)
        print("\n  L16 (4 osc):")
        print(f"  Loschmidt echo: {result.loschmidt_echo:.6f}")
        print(f"  Energy variance: {result.energy_variance:.2e}")
        print(f"  Fidelity susceptibility: {result.fidelity_susceptibility:.4f}")
        print(f"  R_global: {result.order_parameter:.4f}")
        print(f"  Stability: {result.stability_score:.4f} → {result.action}")
        assert isinstance(result.action, str)

    def test_adjust_action_threshold(self, monkeypatch):
        """Moderate composite score selects an adjustment action."""
        from scpn_quantum_control.l16 import quantum_director as qd

        monkeypatch.setattr(qd, "loschmidt_echo", lambda K, omega, t=0.5: 0.5)
        monkeypatch.setattr(qd, "energy_variance", lambda K, omega: 0.4)
        monkeypatch.setattr(qd, "fidelity_susceptibility", lambda K, omega: 1.0)
        monkeypatch.setattr(
            qd,
            "classical_exact_diag",
            lambda n, K, omega: {"ground_state": [1.0, 0.0]},
        )
        monkeypatch.setattr(qd, "quantum_to_ssgf_state", lambda sv, n: {"R_global": 0.5})

        K = build_knm_paper27(L=1)
        omega = OMEGA_N_16[:1]
        result = compute_l16_lyapunov(K, omega)

        assert result.action == "adjust"
        assert 0.4 < result.stability_score <= 0.7

    def test_halt_action_threshold(self, monkeypatch):
        """Low composite score selects a halt action."""
        from scpn_quantum_control.l16 import quantum_director as qd

        monkeypatch.setattr(qd, "loschmidt_echo", lambda K, omega, t=0.5: 0.1)
        monkeypatch.setattr(qd, "energy_variance", lambda K, omega: 2.0)
        monkeypatch.setattr(qd, "fidelity_susceptibility", lambda K, omega: 9.0)
        monkeypatch.setattr(
            qd,
            "classical_exact_diag",
            lambda n, K, omega: {"ground_state": [1.0, 0.0]},
        )
        monkeypatch.setattr(qd, "quantum_to_ssgf_state", lambda sv, n: {"R_global": 0.0})

        K = build_knm_paper27(L=1)
        omega = OMEGA_N_16[:1]
        result = compute_l16_lyapunov(K, omega)

        assert result.action == "halt"
        assert result.stability_score <= 0.4
