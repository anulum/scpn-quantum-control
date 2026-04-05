# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for AppQSim Protocol
"""Tests for application-oriented quantum simulation benchmarking.

Covers:
    - AppQSimMetrics dataclass
    - _exact_order_parameter correctness
    - _exact_correlators output shape
    - appqsim_benchmark with provided statevector
    - appqsim_benchmark with VQE fallback (circuit_sv=None)
    - Metric bounds and physical invariants
    - Small system (3 qubits)
"""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.benchmarks.appqsim_protocol import (
    AppQSimMetrics,
    _exact_correlators,
    _exact_order_parameter,
    appqsim_benchmark,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


def _system(n: int = 3):
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    return K, omega


class TestExactOrderParameter:
    def test_bounded(self):
        K, omega = _system(3)
        r = _exact_order_parameter(K, omega)
        assert 0 <= r <= 1.0

    def test_finite(self):
        K, omega = _system(4)
        r = _exact_order_parameter(K, omega)
        assert np.isfinite(r)


class TestExactCorrelators:
    def test_shape(self):
        K, omega = _system(3)
        C = _exact_correlators(K, omega)
        assert C.shape == (3, 3)

    def test_symmetric(self):
        K, omega = _system(3)
        C = _exact_correlators(K, omega)
        np.testing.assert_allclose(C, C.T, atol=1e-10)


class TestAppQSimBenchmark:
    def test_with_exact_sv(self):
        """Provide exact ground state as circuit_sv → near-zero errors."""
        from qiskit.quantum_info import Statevector

        from scpn_quantum_control.hardware.classical import classical_exact_diag

        K, omega = _system(3)
        exact = classical_exact_diag(3, K=K, omega=omega)
        psi = np.ascontiguousarray(exact["ground_state"])
        sv = Statevector(psi)

        result = appqsim_benchmark(K, omega, circuit_sv=sv, n_gates=10, circuit_depth=5)
        assert isinstance(result, AppQSimMetrics)
        assert result.order_parameter_error < 1e-8
        assert result.energy_relative_error_pct < 1e-6
        assert result.n_qubits == 3
        assert result.n_gates == 10
        assert result.circuit_depth == 5

    def test_correlation_fidelity_exact(self):
        """Exact SV should give correlation fidelity ≈ 1."""
        from qiskit.quantum_info import Statevector

        from scpn_quantum_control.hardware.classical import classical_exact_diag

        K, omega = _system(3)
        exact = classical_exact_diag(3, K=K, omega=omega)
        sv = Statevector(np.ascontiguousarray(exact["ground_state"]))
        result = appqsim_benchmark(K, omega, circuit_sv=sv)
        assert result.correlation_fidelity > 0.99

    def test_vqe_fallback(self):
        """Without circuit_sv, should fall back to VQE."""
        K, omega = _system(3)
        result = appqsim_benchmark(K, omega)
        assert isinstance(result, AppQSimMetrics)
        assert result.n_qubits == 3
        assert result.n_gates > 0
        assert result.circuit_depth > 0

    def test_metrics_positive(self):
        """Errors and metrics should be non-negative."""
        K, omega = _system(3)
        result = appqsim_benchmark(K, omega)
        assert result.order_parameter_error >= 0
        assert result.energy_relative_error_pct >= 0

    def test_4_qubit(self):
        K, omega = _system(4)
        result = appqsim_benchmark(K, omega)
        assert result.n_qubits == 4
