# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Variational Metric
"""Tests for the exact variational-dynamics linear system (analytic QGT).

Covers the π-shift state-derivative identity against a closed-form single-qubit
derivative and against central finite differences, the McLachlan metric and force
assemblies against direct contractions, and the ansatz validation contract.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector

from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27, knm_to_ansatz
from scpn_quantum_control.phase.variational_metric import (
    analytic_state_derivatives,
    assert_single_parameter_rotations,
    imaginary_time_force,
    mclachlan_metric,
    real_time_force,
)


def _state_evaluator(ansatz: QuantumCircuit):
    def state_of(values):
        return np.asarray(
            Statevector.from_instruction(ansatz.assign_parameters(values)).data,
            dtype=np.complex128,
        )

    return state_of


def _knm_ansatz_state():
    ansatz = knm_to_ansatz(build_knm_paper27(L=3), reps=2)
    rng = np.random.default_rng(7)
    theta = rng.normal(0.0, 0.6, size=ansatz.num_parameters)
    return ansatz, theta, _state_evaluator(ansatz)


class TestPiShiftDerivative:
    def test_single_qubit_ry_matches_closed_form(self):
        """RY(θ)|0> = [cos(θ/2), sin(θ/2)]; its exact derivative is known."""
        theta_sym = Parameter("t")
        qc = QuantumCircuit(1)
        qc.ry(theta_sym, 0)
        state_of = _state_evaluator(qc)

        theta = np.array([0.737], dtype=np.float64)
        dpsi = analytic_state_derivatives(state_of, theta)

        half = theta[0] / 2.0
        expected = np.array([-0.5 * np.sin(half), 0.5 * np.cos(half)], dtype=np.complex128)
        assert dpsi.shape == (1, 2)
        np.testing.assert_allclose(dpsi[0], expected, atol=1e-12)

    def test_matches_central_finite_difference(self):
        """The exact identity agrees with a converged central finite difference."""
        ansatz, theta, state_of = _knm_ansatz_state()
        dpsi = analytic_state_derivatives(state_of, theta)

        eps = 1e-6
        for k in range(theta.size):
            tp = theta.copy()
            tp[k] += eps
            tm = theta.copy()
            tm[k] -= eps
            fd = (state_of(tp) - state_of(tm)) / (2.0 * eps)
            np.testing.assert_allclose(dpsi[k], fd, atol=1e-8)

    def test_shape_and_dtype(self):
        ansatz, theta, state_of = _knm_ansatz_state()
        dpsi = analytic_state_derivatives(state_of, theta)
        assert dpsi.shape == (theta.size, 2**3)
        assert dpsi.dtype == np.complex128


class TestMetricAndForces:
    def test_metric_equals_real_gram_matrix(self):
        ansatz, theta, state_of = _knm_ansatz_state()
        dpsi = analytic_state_derivatives(state_of, theta)
        metric = mclachlan_metric(dpsi)

        n = theta.size
        direct = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                direct[i, j] = float(np.real(np.vdot(dpsi[i], dpsi[j])))
        np.testing.assert_allclose(metric, direct, atol=1e-12)
        np.testing.assert_allclose(metric, metric.T, atol=1e-12)

    def test_metric_is_positive_semidefinite(self):
        ansatz, theta, state_of = _knm_ansatz_state()
        dpsi = analytic_state_derivatives(state_of, theta)
        eigenvalues = np.linalg.eigvalsh(mclachlan_metric(dpsi))
        assert eigenvalues.min() > -1e-10

    def test_real_time_force_matches_definition(self):
        ansatz, theta, state_of = _knm_ansatz_state()
        dpsi = analytic_state_derivatives(state_of, theta)
        dim = dpsi.shape[1]
        rng = np.random.default_rng(3)
        h_psi = (rng.normal(size=dim) + 1j * rng.normal(size=dim)).astype(np.complex128)

        force = real_time_force(dpsi, h_psi)
        direct = np.array([-np.imag(np.vdot(dpsi[i], h_psi)) for i in range(theta.size)])
        np.testing.assert_allclose(force, direct, atol=1e-12)

    def test_imaginary_time_force_matches_definition(self):
        ansatz, theta, state_of = _knm_ansatz_state()
        dpsi = analytic_state_derivatives(state_of, theta)
        dim = dpsi.shape[1]
        rng = np.random.default_rng(4)
        h_psi = (rng.normal(size=dim) + 1j * rng.normal(size=dim)).astype(np.complex128)

        force = imaginary_time_force(dpsi, h_psi)
        direct = np.array([-np.real(np.vdot(dpsi[i], h_psi)) for i in range(theta.size)])
        np.testing.assert_allclose(force, direct, atol=1e-12)


class TestAnsatzValidation:
    def test_accepts_physics_informed_ansatz(self):
        ansatz = knm_to_ansatz(build_knm_paper27(L=3), reps=2)
        assert_single_parameter_rotations(ansatz)  # must not raise

    def test_rejects_non_pauli_rotation(self):
        theta = Parameter("t")
        qc = QuantumCircuit(1)
        qc.p(theta, 0)  # phase gate: generator (I-Z)/2, not a Pauli
        with pytest.raises(ValueError, match="not a Pauli rotation"):
            assert_single_parameter_rotations(qc)

    def test_rejects_multiple_parameters_in_one_gate(self):
        a = Parameter("a")
        b = Parameter("b")
        qc = QuantumCircuit(1)
        qc.ry(a + b, 0)  # one ry gate, two free symbols
        with pytest.raises(ValueError, match="free parameters"):
            assert_single_parameter_rotations(qc)

    def test_rejects_reused_parameter(self):
        a = Parameter("a")
        qc = QuantumCircuit(2)
        qc.ry(a, 0)
        qc.rz(a, 1)  # same parameter in two gates
        with pytest.raises(ValueError, match="reused across multiple gates"):
            assert_single_parameter_rotations(qc)

    def test_skips_non_circuit_doubles(self):
        class FakeAnsatz:
            num_parameters = 1

        assert_single_parameter_rotations(FakeAnsatz())  # must not raise

    def test_ignores_unparametrised_gates(self):
        theta = Parameter("t")
        qc = QuantumCircuit(2)
        qc.h(0)  # no free parameter
        qc.ry(theta, 1)
        qc.cz(0, 1)  # no free parameter
        assert_single_parameter_rotations(qc)  # must not raise
