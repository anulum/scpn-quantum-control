# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase-QNode Circuit Support Integration Tests
"""Integration tests for Phase-QNode support and planning surfaces."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.phase.qnode_circuit import (
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeDensityCircuit,
    PhaseQNodeGradientEvaluationPlan,
    PhaseQNodeNoiseChannel,
    PhaseQNodeSupportError,
    execute_phase_qnode_circuit,
    execute_phase_qnode_density_matrix,
    parameter_shift_phase_qnode_gradient,
    phase_qnode_computational_basis_fisher_information,
    phase_qnode_computational_basis_fisher_support_report,
    phase_qnode_density_support_report,
    phase_qnode_gradient_support_report,
    phase_qnode_metric_support_report,
    phase_qnode_quantum_fisher_information,
    phase_qnode_support_report,
    plan_phase_qnode_parameter_shift_evaluations,
)


def test_phase_qnode_gate_aware_plan_reuses_tied_commuting_parameter_shifts() -> None:
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(
            ("rz", (0,), 0),
            ("rz", (0,), 0),
            ("ry", (0,), 1),
        ),
        observable=PauliTerm(1.0, ((0, "x"),)),
    )
    params = np.array([0.31, -0.27], dtype=float)

    plan = plan_phase_qnode_parameter_shift_evaluations(circuit, params)
    gradient = parameter_shift_phase_qnode_gradient(circuit, params)
    payload = plan.to_dict()

    assert isinstance(plan, PhaseQNodeGradientEvaluationPlan)
    assert plan.supported
    assert plan.method == "registered_phase_qnode_gate_aware_parameter_shift"
    assert plan.differentiable_parameters == (0, 1)
    assert plan.operation_level_naive_evaluations == 6
    assert plan.planned_shifted_evaluations == 4
    assert plan.saved_shifted_evaluations == 2
    assert plan.groups[0].parameter_index == 0
    assert plan.groups[0].commuting
    assert plan.groups[0].operation_indices == (0, 1)
    assert plan.groups[0].generator_keys == ("z", "z")
    assert plan.groups[0].frequency_gaps == (2.0,)
    assert payload["generic_scalar_objective_evaluations"] == 4
    assert gradient.parameter_shift_evaluations == plan.planned_shifted_evaluations
    assert gradient.evaluation_plan == plan
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


def test_phase_qnode_gate_aware_plan_does_not_claim_noncommuting_reuse() -> None:
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("rx", (0,), 0), ("ry", (0,), 0)),
        observable=PauliTerm(1.0, ((0, "z"),)),
    )
    params = np.array([0.41], dtype=float)

    plan = plan_phase_qnode_parameter_shift_evaluations(circuit, params)
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

    assert plan.supported
    assert plan.groups[0].commuting is False
    assert plan.groups[0].frequency_gaps == (1.0, 2.0)
    assert plan.planned_shifted_evaluations == 4
    assert "non-commuting" in plan.groups[0].reason
    np.testing.assert_allclose(gradient.gradient, finite_difference, atol=1e-6)


def test_phase_qnode_gate_aware_plan_fails_closed_for_density_route() -> None:
    circuit = PhaseQNodeDensityCircuit(
        n_qubits=1,
        operations=(("rx", (0,), 0), PhaseQNodeNoiseChannel("phase_flip", (0,), 0.1)),
        observable=PauliTerm(1.0, ((0, "z"),)),
    )

    plan = plan_phase_qnode_parameter_shift_evaluations(circuit, np.array([0.2]))

    assert not plan.supported
    assert plan.planned_shifted_evaluations == 0
    assert "pure-state" in plan.fallback_reason


def test_phase_qnode_route_support_reports_block_density_noise_for_pure_metrics() -> None:
    circuit = PhaseQNodeDensityCircuit(
        n_qubits=1,
        operations=(
            ("ry", (0,), 0),
            PhaseQNodeNoiseChannel("amplitude_damping", (0,), 0.2),
        ),
        observable="pauli_z",
    )
    params = np.array([0.31], dtype=float)

    gradient_report = phase_qnode_gradient_support_report(circuit, params)
    metric_report = phase_qnode_metric_support_report(circuit, params)
    fisher_report = phase_qnode_computational_basis_fisher_support_report(circuit, params)

    for report in (gradient_report, metric_report, fisher_report):
        assert not report.supported
        assert "PhaseQNodeCircuit" in report.failure_reason
        assert "amplitude_damping" in report.failure_reason
        assert report.differentiable_parameters == (0,)
        assert report.to_dict()["alternatives"]
    with pytest.raises(PhaseQNodeSupportError) as gradient_error:
        parameter_shift_phase_qnode_gradient(circuit, params)
    with pytest.raises(PhaseQNodeSupportError) as metric_error:
        phase_qnode_quantum_fisher_information(circuit, params)
    with pytest.raises(PhaseQNodeSupportError) as fisher_error:
        phase_qnode_computational_basis_fisher_information(circuit, params)
    assert gradient_error.value.report == gradient_report
    assert metric_error.value.report == metric_report
    assert fisher_error.value.report == fisher_report


def test_phase_qnode_unsupported_routes_fail_with_structured_support_report() -> None:
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("u3", (0,), 0),),
        observable="pauli_z",
    )

    report = phase_qnode_support_report(circuit, np.array([0.2], dtype=float))

    assert not report.supported
    assert report.unsupported_gates == ("u3",)
    assert "u3" in report.failure_reason
    with pytest.raises(PhaseQNodeSupportError) as exc_info:
        execute_phase_qnode_circuit(circuit, np.array([0.2], dtype=float))
    assert exc_info.value.report == report


def test_phase_qnode_controlled_gate_support_reports_validate_arity() -> None:
    circuit = PhaseQNodeCircuit(
        n_qubits=3,
        operations=(("ccnot", (0, 1)),),
        observable=PauliTerm(1.0, ((0, "z"),)),
    )

    report = phase_qnode_support_report(circuit, np.array([], dtype=float))

    assert not report.supported
    assert "gate arity mismatches" in report.failure_reason
    with pytest.raises(PhaseQNodeSupportError):
        execute_phase_qnode_circuit(circuit, np.array([], dtype=float))


def test_phase_qnode_density_support_reports_fail_closed() -> None:
    circuit = PhaseQNodeDensityCircuit(
        n_qubits=2,
        operations=(("ry", (0,)), ("bit_flip", (0, 1), 0.2)),
        observable="pauli_z",
    )

    report = phase_qnode_density_support_report(circuit, np.array([], dtype=float))

    assert not report.supported
    assert "missing parameter" in report.failure_reason
    assert "noise channel arity" in report.failure_reason
    with pytest.raises(PhaseQNodeSupportError):
        execute_phase_qnode_density_matrix(circuit, np.array([], dtype=float))
    with pytest.raises(ValueError, match="between 0 and 1"):
        PhaseQNodeNoiseChannel("bit_flip", (0,), 1.01)
