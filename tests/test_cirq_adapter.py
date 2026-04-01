# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Cirq Adapter
"""Tests for Cirq backend adapter — elite multi-angle coverage."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from scpn_quantum_control.hardware.cirq_adapter import (
    CirqResult,
    CirqRunner,
    is_cirq_available,
)

# ---------------------------------------------------------------------------
# Availability gate
# ---------------------------------------------------------------------------


class TestCirqAvailability:
    """Tests that work regardless of cirq installation status."""

    def test_is_cirq_available_returns_bool(self):
        assert isinstance(is_cirq_available(), bool)

    def test_import_does_not_crash(self):
        from scpn_quantum_control.hardware import cirq_adapter  # noqa: F401

        assert True

    def test_cirq_result_dataclass(self):
        r = CirqResult(energy=-1.5, n_qubits=4, device_name="simulator")
        assert r.energy == -1.5
        assert r.n_qubits == 4
        assert r.device_name == "simulator"

    def test_cirq_result_equality(self):
        r1 = CirqResult(energy=0.0, n_qubits=2, device_name="sim")
        r2 = CirqResult(energy=0.0, n_qubits=2, device_name="sim")
        assert r1 == r2

    def test_cirq_result_different_energy(self):
        r1 = CirqResult(energy=0.0, n_qubits=2, device_name="sim")
        r2 = CirqResult(energy=1.0, n_qubits=2, device_name="sim")
        assert r1 != r2


# ---------------------------------------------------------------------------
# CirqRunner when cirq is NOT available
# ---------------------------------------------------------------------------


class TestCirqRunnerWithoutCirq:
    """Verify correct ImportError when cirq is absent."""

    def test_runner_raises_import_error(self):
        with patch("scpn_quantum_control.hardware.cirq_adapter._CIRQ_AVAILABLE", False):
            K = np.eye(2)
            omega = np.ones(2)
            with pytest.raises(ImportError, match="Cirq not installed"):
                CirqRunner(K, omega)


# ---------------------------------------------------------------------------
# CirqRunner when cirq IS available
# ---------------------------------------------------------------------------

_SKIP_NO_CIRQ = pytest.mark.skipif(not is_cirq_available(), reason="Cirq not installed")


@_SKIP_NO_CIRQ
class TestCirqRunnerInit:
    """Construction and attribute checks."""

    def test_init_stores_params(self):
        K = np.array([[0, 0.5], [0.5, 0]])
        omega = np.array([1.0, -1.0])
        runner = CirqRunner(K, omega)
        assert runner.n == 2
        assert runner.device_name == "simulator"
        np.testing.assert_array_equal(runner.K, K)
        np.testing.assert_array_equal(runner.omega, omega)

    def test_custom_device_name(self):
        runner = CirqRunner(np.eye(2), np.ones(2), device="sycamore")
        assert runner.device_name == "sycamore"

    def test_qubit_count_matches_K(self):
        for n in (2, 3, 4):
            K = np.eye(n) * 0.3
            omega = np.ones(n)
            runner = CirqRunner(K, omega)
            assert len(runner.qubits) == n


@_SKIP_NO_CIRQ
class TestCirqTrotterCircuit:
    """Trotter step circuit construction."""

    def test_trotter_step_returns_circuit(self):
        import cirq

        K = np.array([[0, 0.5], [0.5, 0]])
        omega = np.array([1.0, -1.0])
        runner = CirqRunner(K, omega)
        step = runner._build_trotter_step(dt=0.1)
        assert isinstance(step, cirq.Circuit)
        assert len(step) > 0

    def test_zero_coupling_no_xx_yy_gates(self):
        """Zero K should produce only Z rotations."""
        K = np.zeros((2, 2))
        omega = np.array([1.0, 2.0])
        runner = CirqRunner(K, omega)
        step = runner._build_trotter_step(dt=0.1)
        # Should only have single-qubit ops (rz)
        for moment in step:
            for op in moment:
                assert len(op.qubits) == 1

    def test_zero_omega_no_z_gates(self):
        """Zero omega should produce only two-qubit coupling gates."""
        K = np.array([[0, 1.0], [1.0, 0]])
        omega = np.zeros(2)
        runner = CirqRunner(K, omega)
        step = runner._build_trotter_step(dt=0.1)
        for moment in step:
            for op in moment:
                assert len(op.qubits) == 2


@_SKIP_NO_CIRQ
class TestCirqRunTrotter:
    """Full Trotter evolution and energy."""

    def test_run_trotter_returns_cirq_result(self):
        K = np.array([[0, 0.3], [0.3, 0]])
        omega = np.array([1.0, -1.0])
        runner = CirqRunner(K, omega)
        result = runner.run_trotter(t=0.5, reps=3)
        assert isinstance(result, CirqResult)
        assert result.n_qubits == 2
        assert result.device_name == "simulator"
        assert isinstance(result.energy, float)

    def test_energy_is_real_finite(self):
        K = np.array([[0, 0.5], [0.5, 0]])
        omega = np.array([0.5, -0.5])
        runner = CirqRunner(K, omega)
        result = runner.run_trotter(t=1.0, reps=5)
        assert np.isfinite(result.energy)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_run_trotter_various_sizes(self, n):
        rng = np.random.default_rng(42)
        K = rng.uniform(0, 0.5, (n, n))
        K = (K + K.T) / 2
        np.fill_diagonal(K, 0)
        omega = rng.uniform(-1, 1, n)
        runner = CirqRunner(K, omega)
        result = runner.run_trotter(t=0.5, reps=2)
        assert result.n_qubits == n
        assert np.isfinite(result.energy)

    def test_zero_evolution_time(self):
        """t=0 → trivial circuit → energy from |00...0>."""
        K = np.array([[0, 0.3], [0.3, 0]])
        omega = np.array([1.0, 1.0])
        runner = CirqRunner(K, omega)
        result = runner.run_trotter(t=0.0, reps=1)
        assert np.isfinite(result.energy)

    def test_single_rep(self):
        K = np.array([[0, 0.2], [0.2, 0]])
        omega = np.array([0.5, -0.5])
        runner = CirqRunner(K, omega)
        result = runner.run_trotter(t=1.0, reps=1)
        assert isinstance(result.energy, float)


@_SKIP_NO_CIRQ
class TestCirqComputeEnergy:
    """Direct tests for _compute_energy."""

    def test_all_zero_state_energy(self):
        """State |00> = [1,0,0,0]: Z expectation = +1 for all qubits."""
        K = np.array([[0, 0.3], [0.3, 0]])
        omega = np.array([2.0, 3.0])
        runner = CirqRunner(K, omega)
        sv = np.array([1, 0, 0, 0], dtype=complex)
        e = runner._compute_energy(sv)
        # E = -omega_0/2 * (+1) - omega_1/2 * (+1) = -2.5
        np.testing.assert_allclose(e, -2.5)

    def test_energy_normalised_state(self):
        """Uniform superposition should give zero Z expectation."""
        K = np.zeros((2, 2))
        omega = np.array([1.0, 1.0])
        runner = CirqRunner(K, omega)
        sv = np.ones(4, dtype=complex) / 2.0  # |++>
        e = runner._compute_energy(sv)
        np.testing.assert_allclose(e, 0.0, atol=1e-14)
