# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Appqsim
"""Tests for AppQSim benchmarking protocol."""

from __future__ import annotations

from scpn_quantum_control.benchmarks.appqsim_protocol import (
    AppQSimMetrics,
    appqsim_benchmark,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


class TestAppQSimBenchmark:
    def test_returns_metrics(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = appqsim_benchmark(K, omega)
        assert isinstance(result, AppQSimMetrics)

    def test_order_parameter_error_small(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = appqsim_benchmark(K, omega)
        assert result.order_parameter_error < 1.0

    def test_energy_error_finite(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = appqsim_benchmark(K, omega)
        assert result.energy_relative_error_pct >= 0

    def test_correlation_fidelity_bounded(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = appqsim_benchmark(K, omega)
        assert result.correlation_fidelity <= 1.0

    def test_n_qubits(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = appqsim_benchmark(K, omega)
        assert result.n_qubits == 4

    def test_gate_count_positive(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = appqsim_benchmark(K, omega)
        assert result.n_gates > 0

    def test_scpn_appqsim(self):
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
