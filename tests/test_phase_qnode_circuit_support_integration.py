# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase-QNode Circuit Support Integration Tests
"""Integration tests for Phase-QNode support and planning surfaces."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control.phase.qnode_circuit import (
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeDensityCircuit,
    PhaseQNodeGradientEvaluationPlan,
    PhaseQNodeNoiseChannel,
    PhaseQNodeSupportError,
    SparsePauliHamiltonian,
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
    """Tied commuting copies of one generator share one shifted pair and match finite differences."""
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
    """Non-commuting generators on one angle get the two-frequency rule and no reuse claim."""
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
    """A noisy density circuit gets an unsupported plan with no shifted evaluations."""
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
    """Gradient, metric and Fisher routes block density noise and raise their own reports."""
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
    """An unregistered gate is reported, and execution raises that same report."""
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
    """A three-qubit gate written with two qubits is an arity mismatch and does not execute."""
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
    """Density reports name missing parameters and noise arity, and noise probabilities are bounded."""
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


_MIXED_OBSERVABLE = SparsePauliHamiltonian(
    (
        PauliTerm(1.0, ((1, "y"),)),
        PauliTerm(0.6, ((0, "x"), (1, "x"))),
        PauliTerm(0.5, ((2, "y"),)),
        PauliTerm(0.7, ((2, "x"),)),
    )
)


@pytest.mark.parametrize(
    "shared",
    [
        pytest.param((("crx", (0, 1), 0), ("rx", (2,), 0)), id="crx-and-rx"),
        pytest.param((("crx", (0, 1), 0), ("crx", (2, 1), 0)), id="crx-on-two-pairs"),
        pytest.param((("crz", (0, 1), 0), ("rz", (2,), 0)), id="crz-and-rz"),
    ],
)
def test_shared_parameter_with_a_controlled_rotation_uses_half_integer_gaps(
    shared: tuple[tuple[str, tuple[int, ...], int], ...],
) -> None:
    """A shared angle on distinct gates, one of them controlled, needs every half-integer gap.

    Regression: the planner used gaps ``1..n`` for every non-identical group, so a
    controlled rotation's ``1/2`` components were missed and the gradient was wrong.
    """
    preparation = (("ry", (0,), 1), ("ry", (1,), 2), ("ry", (2,), 3), ("h", (0,)))
    circuit = PhaseQNodeCircuit(
        n_qubits=3, operations=(*preparation, *shared), observable=_MIXED_OBSERVABLE
    )
    params = np.array([0.53, 0.8, -0.4, 1.1], dtype=float)

    plan = plan_phase_qnode_parameter_shift_evaluations(circuit, params)
    gradient = parameter_shift_phase_qnode_gradient(circuit, params)

    shared_group = next(group for group in plan.groups if group.parameter_index == 0)
    assert shared_group.frequency_gaps == (0.5, 1.0, 1.5, 2.0)
    assert shared_group.shifted_evaluations == 8
    step = 1e-6
    plus = params.copy()
    minus = params.copy()
    plus[0] += step
    minus[0] -= step
    finite_difference = (
        execute_phase_qnode_circuit(circuit, plus).value
        - execute_phase_qnode_circuit(circuit, minus).value
    ) / (2.0 * step)
    np.testing.assert_allclose(gradient.gradient[0], finite_difference, atol=1e-6)


def test_identical_controlled_rotations_keep_their_collapsed_gaps() -> None:
    """Tied copies of one controlled rotation still collapse to the scaled pair of gaps."""
    circuit = PhaseQNodeCircuit(
        n_qubits=3,
        operations=(("h", (0,)), ("crx", (0, 1), 0), ("crx", (0, 1), 0)),
        observable=_MIXED_OBSERVABLE,
    )

    plan = plan_phase_qnode_parameter_shift_evaluations(circuit, [0.53])

    assert plan.groups[0].frequency_gaps == (1.0, 2.0)


def test_pure_support_report_names_parametric_gates_without_a_parameter_index() -> None:
    """A parametric gate written without its parameter index is reported by position."""
    circuit = PhaseQNodeCircuit(n_qubits=1, operations=(("rx", (0,)),), observable="pauli_z")

    report = phase_qnode_support_report(circuit, [0.2])

    assert not report.supported
    assert (
        report.failure_reason == "parametric gates missing parameter indices at operations: (0,)"
    )


@pytest.mark.parametrize(
    ("n_qubits", "operations", "observable", "reason"),
    [
        pytest.param(1, (("u3", (0,), 0),), "pauli_z", "unsupported unitary gates: u3", id="gate"),
        pytest.param(
            1,
            (PhaseQNodeNoiseChannel("erasure", (0,), 0.1),),
            "pauli_z",
            "unsupported noise channels: erasure",
            id="noise",
        ),
        pytest.param(
            1, (("rx", (0,), 0),), "pauli_q", "unsupported observables: pauli_q", id="observable"
        ),
        pytest.param(
            1,
            (("rx", (0,), 3),),
            "pauli_z",
            "parameter indices outside supplied vector: (3,)",
            id="parameter-range",
        ),
        pytest.param(
            1,
            (("rx", (0,)),),
            "pauli_z",
            "parametric gates missing parameter indices at operations: (0,)",
            id="missing-parameter",
        ),
        pytest.param(
            2,
            (("rx", (0, 1), 0),),
            "pauli_z",
            "gate arity mismatches at operations: ('0:rx/2',)",
            id="gate-arity",
        ),
        pytest.param(
            2,
            (PhaseQNodeNoiseChannel("phase_flip", (0, 1), 0.1),),
            "pauli_z",
            "noise channel arity mismatches at operations: ('0:phase_flip/2',)",
            id="noise-arity",
        ),
    ],
)
def test_density_support_report_names_each_unsupported_part(
    n_qubits: int, operations: tuple[object, ...], observable: str, reason: str
) -> None:
    """The density report names unsupported gates, noise, observables, parameters and arities."""
    circuit = PhaseQNodeDensityCircuit(
        n_qubits=n_qubits, operations=cast(Any, operations), observable=observable
    )

    report = phase_qnode_density_support_report(circuit, [0.2])

    assert not report.supported
    assert report.failure_reason == reason


def test_blocked_density_routes_carry_the_density_refusal_without_a_noise_clause() -> None:
    """A noiseless density circuit with an unsupported gate gets both refusals and no noise note."""
    circuit = PhaseQNodeDensityCircuit(
        n_qubits=1, operations=(("u3", (0,), 0),), observable="pauli_z"
    )

    report = phase_qnode_metric_support_report(circuit, [0.2])

    assert not report.supported
    assert report.failure_reason == (
        "pure-state Fubini-Study and QFI metrics require the pure-state PhaseQNodeCircuit "
        "route; unsupported unitary gates: u3"
    )
    assert report.unsupported_gates == ("u3",)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        pytest.param(np.array([True]), "finite real numeric values", id="bool"),
        pytest.param(np.array([1j]), "finite real numeric values", id="complex"),
        pytest.param(np.array(["a"]), "finite real numeric values", id="text"),
        pytest.param(np.ones((1, 1)), "one-dimensional array", id="matrix"),
        pytest.param(np.array([np.nan]), "finite real numeric values", id="nan"),
    ],
)
def test_support_reports_refuse_parameters_that_are_not_a_finite_real_vector(
    parameters: np.ndarray, message: str
) -> None:
    """Parameters must be a one-dimensional vector of finite real numbers."""
    circuit = PhaseQNodeCircuit(n_qubits=1, operations=(("rx", (0,), 0),), observable="pauli_z")

    with pytest.raises(ValueError, match=f"^parameters must (contain|be a) {message}$"):
        phase_qnode_support_report(circuit, parameters)


@pytest.mark.parametrize(
    ("gate", "qubits", "key"),
    [
        ("rx", (0,), "x"),
        ("ry", (0,), "y"),
        ("rz", (0,), "z"),
        ("phase", (0,), "z"),
        ("crx", (0, 1), "x"),
        ("cry", (0, 1), "y"),
        ("crz", (0, 1), "z"),
        ("rxx", (0, 1), "xx"),
        ("ryy", (0, 1), "yy"),
        ("rzz", (0, 1), "zz"),
    ],
)
def test_planner_names_the_pauli_generator_of_every_parametric_gate(
    gate: str, qubits: tuple[int, ...], key: str
) -> None:
    """Each registered parametric gate is planned with its Pauli generator."""
    circuit = PhaseQNodeCircuit(
        n_qubits=2, operations=(("h", (0,)), (gate, qubits, 0)), observable="pauli_z"
    )

    plan = plan_phase_qnode_parameter_shift_evaluations(circuit, [0.4])

    assert plan.groups[0].generator_keys == (key,)


@pytest.mark.parametrize(
    ("shared", "commuting"),
    [
        pytest.param((("rxx", (0, 1), 0), ("rx", (1,), 0)), True, id="xx-with-x"),
        pytest.param((("rzz", (0, 1), 0), ("rx", (1,), 0)), False, id="zz-with-x"),
        pytest.param((("rxx", (0, 1), 0), ("ryy", (0, 1), 0)), True, id="xx-with-yy"),
    ],
)
def test_planner_decides_commutation_of_overlapping_two_qubit_generators(
    shared: tuple[tuple[str, tuple[int, ...], int], ...], commuting: bool
) -> None:
    """Overlapping Pauli strings commute when they differ on an even number of qubits."""
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(("ry", (0,), 1), ("ry", (1,), 2), *shared),
        observable=SparsePauliHamiltonian(
            (PauliTerm(1.0, ((0, "x"), (1, "z"))), PauliTerm(0.5, ((1, "y"),)))
        ),
    )
    params = np.array([0.53, 0.8, -0.4], dtype=float)

    plan = plan_phase_qnode_parameter_shift_evaluations(circuit, params)
    gradient = parameter_shift_phase_qnode_gradient(circuit, params)

    assert plan.groups[0].commuting is commuting
    step = 1e-6
    plus = params.copy()
    minus = params.copy()
    plus[0] += step
    minus[0] -= step
    finite_difference = (
        execute_phase_qnode_circuit(circuit, plus).value
        - execute_phase_qnode_circuit(circuit, minus).value
    ) / (2.0 * step)
    np.testing.assert_allclose(gradient.gradient[0], finite_difference, atol=1e-6)
