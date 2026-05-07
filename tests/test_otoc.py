# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Otoc
"""Tests for OTOC measurement protocol."""

from __future__ import annotations

import builtins
import importlib

import numpy as np
import pytest

from scpn_quantum_control.analysis.otoc import (
    OTOC,
    OTOCResult,
    _estimate_scrambling_time,
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


class TestOTOCClassInterface:
    def test_compute_delegates_to_function(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        times = np.array([0.0, 0.2])

        otoc = OTOC(K, omega, w_qubit=0, v_qubit=1, w_pauli="X", v_pauli="Z")
        result = otoc.compute(times=times)

        assert isinstance(result, OTOCResult)
        np.testing.assert_allclose(result.times, times)
        assert result.operator_w == "X_0"
        assert result.operator_v == "Z_1"

    def test_compute_requires_k_and_omega(self):
        with pytest.raises(ValueError, match="requires K and omega"):
            OTOC().compute(times=np.array([0.0]))


# ---------------------------------------------------------------------------
# Coverage: Python fallback, edge cases
# ---------------------------------------------------------------------------


class TestOTOCCoverage:
    """Cover missing lines: Python fallback, Lyapunov/scrambling None paths."""

    def test_default_times(self):
        """Cover line 101: times=None defaults to linspace(0, 2, 30)."""
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = compute_otoc(K, omega, times=None)
        assert len(result.times) == 30

    def test_python_fallback(self):
        """Cover lines 133-143: Python OTOC when Rust unavailable."""
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        times = np.linspace(0, 0.5, 5)

        real_import = builtins.__import__
        otoc_module = importlib.import_module("scpn_quantum_control.analysis.otoc")
        real_knm_to_dense = otoc_module.knm_to_dense_matrix
        H_mat = real_knm_to_dense(K, omega)

        def import_without_engine(name, *args, **kwargs):
            if name == "scpn_quantum_engine":
                raise ImportError("forced Python OTOC path")
            return real_import(name, *args, **kwargs)

        try:
            otoc_module.knm_to_dense_matrix = lambda *_args, **_kwargs: H_mat
            builtins.__import__ = import_without_engine
            result = compute_otoc(K, omega, times=times)
        finally:
            builtins.__import__ = real_import
            otoc_module.knm_to_dense_matrix = real_knm_to_dense

        assert len(result.otoc_values) == 5
        assert np.isfinite(result.otoc_values).all()

    def test_lyapunov_none_zero_f0(self):
        """Cover line 167: F(0) ≈ 0 → Lyapunov returns None."""
        from scpn_quantum_control.analysis.otoc import _estimate_lyapunov

        times = np.linspace(0, 1, 10)
        otoc = np.zeros(10)
        result = _estimate_lyapunov(times, otoc)
        assert result is None

    def test_lyapunov_none_too_few_decay_points(self):
        """Cover line 178: fewer than 3 positive decay points."""
        from scpn_quantum_control.analysis.otoc import _estimate_lyapunov

        times = np.linspace(0, 1, 10)
        otoc = np.ones(10) * 0.99
        result = _estimate_lyapunov(times, otoc)
        assert result is None

    def test_scrambling_none_zero_f0(self):
        """Cover line 191: F(0) ≈ 0 → scrambling returns None."""
        from scpn_quantum_control.analysis.otoc import _estimate_scrambling_time

        times = np.linspace(0, 1, 10)
        otoc = np.zeros(10)
        result = _estimate_scrambling_time(times, otoc)
        assert result is None

    def test_scrambling_none_no_threshold(self):
        """Cover line 197: OTOC never drops below 1/e → None."""
        from scpn_quantum_control.analysis.otoc import _estimate_scrambling_time

        times = np.linspace(0, 1, 10)
        otoc = np.ones(10)
        result = _estimate_scrambling_time(times, otoc)
        assert result is None

    def test_scrambling_time_first_threshold_crossing(self):
        times = np.array([0.0, 0.25, 0.5, 0.75])
        otoc = np.array([1.0, 0.8, 0.2, 0.1])

        assert _estimate_scrambling_time(times, otoc) == 0.5
