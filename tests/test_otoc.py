# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for OTOC measurement protocol."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.analysis.otoc import (
    OTOCResult,
    _pauli_matrix,
    compute_otoc,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


class TestPauliMatrix:
    def test_x_hermitian(self):
        X = _pauli_matrix("X", 0, 2)
        np.testing.assert_allclose(X, X.conj().T, atol=1e-12)

    def test_z_diagonal(self):
        Z = _pauli_matrix("Z", 0, 2)
        assert Z[0, 0] == pytest.approx(1.0)
        assert Z[3, 3] == pytest.approx(-1.0)

    def test_shape(self):
        P = _pauli_matrix("Y", 1, 3)
        assert P.shape == (8, 8)

    def test_unitary(self):
        X = _pauli_matrix("X", 0, 2)
        np.testing.assert_allclose(X @ X, np.eye(4), atol=1e-12)


class TestComputeOTOC:
    def test_returns_result(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_otoc(K, omega, times=np.linspace(0, 1, 5))
        assert isinstance(result, OTOCResult)

    def test_otoc_at_t0_is_one(self):
        """F(0) = <W†VWV†> with W,V Pauli → should be ±1."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        times = np.array([0.0, 0.1, 0.2])
        result = compute_otoc(K, omega, times=times)
        assert abs(result.otoc_values[0]) == pytest.approx(1.0, abs=0.01)

    def test_otoc_bounded(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_otoc(K, omega, times=np.linspace(0, 1, 10))
        for val in result.otoc_values:
            assert abs(val) <= 1.0 + 1e-10

    def test_times_shape(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        t = np.linspace(0, 2, 15)
        result = compute_otoc(K, omega, times=t)
        assert result.otoc_values.shape == (15,)

    def test_n_qubits(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = compute_otoc(K, omega, times=np.array([0.0, 0.5]))
        assert result.n_qubits == 4

    def test_operator_labels(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_otoc(K, omega, times=np.array([0.0]), w_qubit=0, v_qubit=2)
        assert "Z_0" in result.operator_w
        assert "X_2" in result.operator_v

    def test_custom_paulis(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_otoc(K, omega, times=np.array([0.0, 0.5]), w_pauli="X", v_pauli="Y")
        assert "X" in result.operator_w
        assert "Y" in result.operator_v

    def test_scrambling_time_type(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_otoc(K, omega, times=np.linspace(0, 5, 20))
        # May or may not find scrambling time depending on dynamics
        assert result.scrambling_time is None or isinstance(result.scrambling_time, float)

    def test_scpn_otoc_measurement(self):
        """Record OTOC at SCPN default parameters."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = compute_otoc(K, omega, times=np.linspace(0, 2, 10))
        print(f"\n  OTOC (4 osc): F(0)={result.otoc_values[0]:.4f}")
        print(f"  F(t_max)={result.otoc_values[-1]:.4f}")
        print(f"  Lyapunov estimate: {result.lyapunov_estimate}")
        print(f"  Scrambling time: {result.scrambling_time}")
        assert isinstance(result.otoc_values[0], float)
