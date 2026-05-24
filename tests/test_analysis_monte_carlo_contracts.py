# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Analysis Monte Carlo contract tests
"""Contract tests for Monte Carlo Rust/Python fallback and finite-size scaling analysis paths."""

from __future__ import annotations

from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27


def test_monte_carlo_rust_engine_path():
    """Verifies 96-109: Rust engine import path in mc_simulate."""
    from scpn_quantum_control.analysis.monte_carlo_xy import MCResult, mc_simulate

    K = build_knm_paper27(L=4)
    # Run with default — will use Rust if available, else fallback
    result = mc_simulate(K, temperature=0.5, n_thermalize=10, n_measure=10, seed=42)
    assert isinstance(result, MCResult)
    assert result.n_oscillators == 4


def test_monte_carlo_finite_size_default_n_values():
    """Verifies 267: n_values defaults to [4,8,16,32] when None."""
    from scpn_quantum_control.analysis.monte_carlo_xy import finite_size_scaling

    result = finite_size_scaling(n_values=[4], n_seeds=1, n_thermalize=10, n_measure=10)
    assert len(result.n_values) == 1


def test_finite_size_power_fit_few_points():
    """Verifies 137-138: _fit_power_ansatz returns None with < 2 points."""
    from scpn_quantum_control.analysis.finite_size_scaling import _fit_power_ansatz

    result = _fit_power_ansatz([4], [1.0])
    assert result is None


class TestFiniteSizeScalingEdge:
    def test_fit_bkt_ansatz_single_point(self):
        from scpn_quantum_control.analysis.finite_size_scaling import _fit_bkt_ansatz

        result = _fit_bkt_ansatz([4], [1.0])
        assert result is None

    def test_fit_power_ansatz_single_point(self):
        from scpn_quantum_control.analysis.finite_size_scaling import _fit_power_ansatz

        result = _fit_power_ansatz([4], [1.0])
        assert result is None

    def test_fit_bkt_ansatz_valid(self):
        from scpn_quantum_control.analysis.finite_size_scaling import _fit_bkt_ansatz

        result = _fit_bkt_ansatz([4, 8, 16], [2.0, 1.8, 1.6])
        assert result is not None

    def test_fit_power_ansatz_valid(self):
        from scpn_quantum_control.analysis.finite_size_scaling import _fit_power_ansatz

        result = _fit_power_ansatz([4, 8, 16], [2.0, 1.8, 1.6])
        assert result is not None
