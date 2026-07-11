# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase-QNode Circuit Differentiation Integration Tests
"""Integration tests for Phase-QNode gradients and information metrics."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.phase.qnode_circuit import (
    DenseHermitianObservable,
    PauliCovarianceObservable,
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeClassicalFisherResult,
    PhaseQNodeMetricTensorResult,
    PhaseQNodeSupportError,
    SparsePauliHamiltonian,
    execute_phase_qnode_circuit,
    parameter_shift_phase_qnode_gradient,
    phase_qnode_computational_basis_fisher_information,
    phase_qnode_computational_basis_fisher_support_report,
    phase_qnode_natural_gradient_metric,
    phase_qnode_quantum_fisher_information,
)


def test_phase_qnode_parameter_shift_matches_finite_difference_for_registered_generators() -> None:
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            ("ry", (0,), 0),
            ("cnot", (0, 1)),
            ("rzz", (0, 1), 1),
            ("rx", (1,), 2),
        ),
        observable=SparsePauliHamiltonian((PauliTerm(1.0, ((0, "z"), (1, "x"))),)),
    )
    params = np.array([0.31, -0.27, 0.43], dtype=float)

    gradient = parameter_shift_phase_qnode_gradient(circuit, params)
    finite_difference = np.zeros_like(params)
    eps = 1e-6
    for index in range(params.size):
        plus = params.copy()
        minus = params.copy()
        plus[index] += eps
        minus[index] -= eps
        finite_difference[index] = (
            execute_phase_qnode_circuit(circuit, plus).value
            - execute_phase_qnode_circuit(circuit, minus).value
        ) / (2.0 * eps)

    np.testing.assert_allclose(gradient.gradient, finite_difference, atol=1e-6)
    assert gradient.support_report.differentiable_parameters == (0, 1, 2)
    assert gradient.parameter_shift_evaluations == 6
    assert gradient.evaluation_plan is not None
    assert gradient.evaluation_plan.planned_shifted_evaluations == 6


def test_phase_qnode_covariance_gradient_uses_product_rule() -> None:
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(("ry", (0,), 0), ("cnot", (0, 1))),
        observable=PauliCovarianceObservable(
            PauliTerm(1.0, ((0, "z"),)),
            PauliTerm(1.0, ((1, "z"),)),
        ),
    )
    params = np.array([0.37], dtype=float)

    gradient = parameter_shift_phase_qnode_gradient(circuit, params)

    np.testing.assert_allclose(gradient.value, np.sin(params[0]) ** 2, atol=1e-12)
    np.testing.assert_allclose(gradient.gradient, [np.sin(2.0 * params[0])], atol=1e-12)
    assert gradient.parameter_shift_evaluations == 2


def test_phase_qnode_dense_hermitian_gradient_matches_finite_difference() -> None:
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable=DenseHermitianObservable(
            np.array([[0.7, 0.2], [0.2, -0.3]], dtype=np.complex128)
        ),
    )
    params = np.array([0.41], dtype=float)

    gradient = parameter_shift_phase_qnode_gradient(circuit, params)
    eps = 1e-6
    plus = params + eps
    minus = params - eps
    finite_difference = (
        execute_phase_qnode_circuit(circuit, plus).value
        - execute_phase_qnode_circuit(circuit, minus).value
    ) / (2.0 * eps)

    np.testing.assert_allclose(gradient.gradient, [finite_difference], atol=1e-6)


def test_phase_qnode_quantum_fisher_information_matches_ry_reference() -> None:
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable="pauli_z",
    )

    result = phase_qnode_quantum_fisher_information(circuit, np.array([0.31], dtype=float))

    assert isinstance(result, PhaseQNodeMetricTensorResult)
    np.testing.assert_allclose(result.fubini_study_metric, [[0.25]], atol=1e-12)
    np.testing.assert_allclose(result.quantum_fisher_information, [[1.0]], atol=1e-12)
    assert result.support_report.differentiable_parameters == (0,)
    assert result.claim_boundary.startswith("pure-state local Phase-QNode")


def test_phase_qnode_quantum_fisher_information_is_gauge_invariant_psd() -> None:
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            ("h", (0,)),
            ("cnot", (0, 1)),
            ("ry", (0,), 0),
            ("rz", (1,), 1),
            ("rxx", (0, 1), 2),
        ),
        observable=SparsePauliHamiltonian((PauliTerm(1.0, ((0, "z"),)),)),
    )

    result = phase_qnode_quantum_fisher_information(
        circuit,
        np.array([0.17, -0.23, 0.41], dtype=float),
    )

    np.testing.assert_allclose(result.fubini_study_metric, result.fubini_study_metric.T)
    np.testing.assert_allclose(result.quantum_fisher_information, 4.0 * result.fubini_study_metric)
    assert np.min(np.linalg.eigvalsh(result.fubini_study_metric)) >= -1e-12
    assert result.derivative_norms.shape == (3,)
    assert np.all(result.derivative_norms > 0.0)


def test_phase_qnode_computational_basis_fisher_matches_ry_reference() -> None:
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable="pauli_z",
    )

    result = phase_qnode_computational_basis_fisher_information(
        circuit,
        np.array([0.31], dtype=float),
    )

    assert isinstance(result, PhaseQNodeClassicalFisherResult)
    np.testing.assert_allclose(
        result.probabilities, [np.cos(0.31 / 2.0) ** 2, np.sin(0.31 / 2.0) ** 2]
    )
    np.testing.assert_allclose(result.classical_fisher_information, [[1.0]], atol=1e-12)
    assert result.measurement == "computational_basis"
    assert "finite-shot" in result.claim_boundary


def test_phase_qnode_computational_basis_fisher_is_bounded_by_qfi() -> None:
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            ("h", (0,)),
            ("cnot", (0, 1)),
            ("ry", (0,), 0),
            ("rz", (1,), 1),
            ("rxx", (0, 1), 2),
        ),
        observable=SparsePauliHamiltonian((PauliTerm(1.0, ((0, "z"),)),)),
    )
    params = np.array([0.17, -0.23, 0.41], dtype=float)

    classical = phase_qnode_computational_basis_fisher_information(circuit, params)
    quantum = phase_qnode_quantum_fisher_information(circuit, params)

    np.testing.assert_allclose(
        classical.classical_fisher_information,
        classical.classical_fisher_information.T,
        atol=1e-12,
    )
    gap = quantum.quantum_fisher_information - classical.classical_fisher_information
    assert np.min(np.linalg.eigvalsh(gap)) >= -1e-10


def test_phase_qnode_computational_basis_fisher_reports_finite_shot_uncertainty() -> None:
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable="pauli_z",
    )

    result = phase_qnode_computational_basis_fisher_information(
        circuit,
        np.array([0.31], dtype=float),
        shot_count=4096,
        confidence_z=2.0,
    )

    assert result.shot_count == 4096
    assert result.count_record is None
    assert result.sampling_model == "multinomial_delta_method_expected_counts"
    finite_shot = result.finite_shot_classical_fisher_information
    assert finite_shot is not None
    np.testing.assert_allclose(
        finite_shot,
        result.classical_fisher_information,
        atol=1e-12,
    )
    standard_error = result.fisher_standard_error
    confidence_radius = result.fisher_confidence_radius
    assert standard_error is not None
    assert confidence_radius is not None
    assert standard_error.shape == (1, 1)
    assert standard_error[0, 0] > 0.0
    np.testing.assert_allclose(
        confidence_radius,
        2.0 * standard_error,
        atol=1e-12,
    )
    payload = result.to_dict()
    assert payload["shot_count"] == 4096
    assert payload["count_record"] is None
    assert payload["sampling_model"] == "multinomial_delta_method_expected_counts"
    assert payload["fisher_standard_error"] == standard_error.tolist()
    assert "finite-shot" in result.claim_boundary


def test_phase_qnode_computational_basis_fisher_replays_raw_count_record() -> None:
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable="pauli_z",
    )
    params = np.array([0.31], dtype=float)
    counts = np.array([3900, 196], dtype=np.int64)

    result = phase_qnode_computational_basis_fisher_information(
        circuit,
        params,
        observed_counts=counts,
        confidence_z=1.5,
    )

    empirical_probabilities = counts / counts.sum()
    exact = phase_qnode_computational_basis_fisher_information(circuit, params)
    expected = (
        exact.probability_derivatives
        @ (exact.probability_derivatives / empirical_probabilities[np.newaxis, :]).T
    )

    assert result.shot_count == int(counts.sum())
    assert result.count_record == (3900, 196)
    assert result.sampling_model == "multinomial_delta_method_raw_count_replay"
    replay_probabilities = result.empirical_probabilities
    finite_shot = result.finite_shot_classical_fisher_information
    standard_error = result.fisher_standard_error
    confidence_radius = result.fisher_confidence_radius
    assert replay_probabilities is not None
    assert finite_shot is not None
    assert standard_error is not None
    assert confidence_radius is not None
    np.testing.assert_allclose(replay_probabilities, empirical_probabilities)
    np.testing.assert_allclose(finite_shot, expected)
    np.testing.assert_allclose(
        confidence_radius,
        1.5 * standard_error,
        atol=1e-12,
    )


def test_phase_qnode_computational_basis_fisher_validates_finite_shot_inputs() -> None:
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable="pauli_z",
    )
    params = np.array([0.31], dtype=float)

    with pytest.raises(ValueError, match="shot_count must be a positive integer"):
        phase_qnode_computational_basis_fisher_information(circuit, params, shot_count=0)
    with pytest.raises(ValueError, match="observed_counts must have shape"):
        phase_qnode_computational_basis_fisher_information(
            circuit,
            params,
            observed_counts=np.array([1, 2, 3]),
        )
    with pytest.raises(ValueError, match="observed_counts must be integer counts"):
        phase_qnode_computational_basis_fisher_information(
            circuit,
            params,
            observed_counts=np.array([1.5, 2.5]),
        )
    with pytest.raises(ValueError, match="observed_counts sum must equal shot_count"):
        phase_qnode_computational_basis_fisher_information(
            circuit,
            params,
            shot_count=12,
            observed_counts=np.array([5, 6]),
        )
    with pytest.raises(ValueError, match="strictly positive"):
        phase_qnode_computational_basis_fisher_information(
            circuit,
            params,
            observed_counts=np.array([4096, 0]),
        )


def test_phase_qnode_computational_basis_fisher_fails_closed_at_singular_probability() -> None:
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable="pauli_z",
    )

    report = phase_qnode_computational_basis_fisher_support_report(
        circuit,
        np.array([0.0], dtype=float),
    )

    assert not report.supported
    assert "zero-probability" in report.failure_reason
    with pytest.raises(PhaseQNodeSupportError, match="zero-probability") as exc_info:
        phase_qnode_computational_basis_fisher_information(circuit, np.array([0.0], dtype=float))
    assert exc_info.value.report == report


def test_phase_qnode_natural_gradient_metric_provider_returns_fubini_study_metric() -> None:
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("rx", (0,), 0),),
        observable="pauli_z",
    )
    metric = phase_qnode_natural_gradient_metric(circuit)

    np.testing.assert_allclose(metric(np.array([0.4], dtype=float)), [[0.25]], atol=1e-12)


def test_phase_qnode_quantum_fisher_information_fails_closed_for_unsupported_routes() -> None:
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("u3", (0,), 0),),
        observable="pauli_z",
    )

    with pytest.raises(PhaseQNodeSupportError) as exc_info:
        phase_qnode_quantum_fisher_information(circuit, np.array([0.2], dtype=float))
    assert "unsupported gates" in exc_info.value.report.failure_reason
