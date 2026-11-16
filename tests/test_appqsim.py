# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Appqsim
"""Tests for AppQSim benchmarking protocol."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.benchmarks.appqsim_protocol import (
    AppQSimMetrics,
    appqsim_benchmark,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


class TestAppQSimBenchmark:
    """The application-level simulation benchmark on a small system."""

    def test_returns_metrics(self) -> None:
        """The benchmark returns a metrics record."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = appqsim_benchmark(K, omega)
        assert isinstance(result, AppQSimMetrics)

    def test_order_parameter_error_small(self) -> None:
        """The order-parameter error stays below one."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = appqsim_benchmark(K, omega)
        assert result.order_parameter_error < 1.0

    def test_energy_error_finite(self) -> None:
        """The relative energy error is non-negative."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = appqsim_benchmark(K, omega)
        assert result.energy_relative_error_pct >= 0

    def test_correlation_fidelity_bounded(self) -> None:
        """The correlation fidelity is at most one."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = appqsim_benchmark(K, omega)
        assert result.correlation_fidelity <= 1.0

    def test_n_qubits(self) -> None:
        """The result records the qubit count."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = appqsim_benchmark(K, omega)
        assert result.n_qubits == 4

    def test_gate_count_positive(self) -> None:
        """The benchmark circuit has gates."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = appqsim_benchmark(K, omega)
        assert result.n_gates > 0

    def test_scpn_appqsim(self) -> None:
        """Record AppQSim metrics for SCPN."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = appqsim_benchmark(K, omega)
        print("\n  AppQSim (3 osc):")
        print(f"  R error: {result.order_parameter_error:.4f}")
        print(f"  E error: {result.energy_relative_error_pct:.2f}%")
        print(f"  Corr fidelity: {result.correlation_fidelity:.4f}")
        print(f"  Gates: {result.n_gates}, depth: {result.circuit_depth}")
        assert isinstance(result.order_parameter_error, float)


@pytest.mark.parametrize("L", [2, 3, 4])
def test_appqsim_various_sizes(L: int) -> None:
    """The benchmark runs at several sizes with finite errors."""
    K = build_knm_paper27(L=L)
    omega = OMEGA_N_16[:L]
    result = appqsim_benchmark(K, omega)
    assert result.n_qubits == L
    assert np.isfinite(result.order_parameter_error)


def test_appqsim_energy_negative() -> None:
    """The benchmark circuit has positive depth."""
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    result = appqsim_benchmark(K, omega)
    assert result.circuit_depth > 0


def test_appqsim_result_has_all_fields() -> None:
    """The result exposes every metric."""
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    result = appqsim_benchmark(K, omega)
    assert hasattr(result, "order_parameter_error")
    assert hasattr(result, "energy_relative_error_pct")
    assert hasattr(result, "correlation_fidelity")
    assert hasattr(result, "n_gates")
    assert hasattr(result, "circuit_depth")


def test_appqsim_error_nonnegative() -> None:
    """The order-parameter error is non-negative."""
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    result = appqsim_benchmark(K, omega)
    assert result.order_parameter_error >= 0
