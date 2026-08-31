# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase Objectives
"""Tests for phase/objectives.py composed differentiable objectives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from scpn_quantum_control.phase import (
    ComposedObjectiveTrainingCertificate,
    ComposedPhaseObjective,
    ObjectiveTerm,
    build_phase_control_objective,
    periodic_regularization_term,
    phase_energy_term,
    phase_fidelity_target_term,
    phase_symmetry_penalty_term,
    smooth_box_safety_penalty_term,
    train_composed_phase_objective,
    validate_composed_objective_training,
)

FloatArray = NDArray[np.float64]


def _finite_difference_gradient(
    objective: ComposedPhaseObjective, params: FloatArray
) -> FloatArray:
    """Evaluate a centred finite-difference gradient through the public objective."""
    step = 1e-6
    grad = np.zeros_like(params)
    for index in range(params.size):
        plus = params.copy()
        minus = params.copy()
        plus[index] += step
        minus[index] -= step
        grad[index] = (objective(plus) - objective(minus)) / (2.0 * step)
    return grad


def test_composed_phase_objective_reports_terms_and_exact_gradient() -> None:
    """Compose every term family and match its gradient to finite differences."""
    objective = build_phase_control_objective(
        3,
        energy_weight=0.7,
        fidelity_target=np.array([0.1, -0.2, 0.3], dtype=float),
        fidelity_weight=0.4,
        regularization_center=np.zeros(3, dtype=float),
        regularization_weight=0.1,
        symmetry_pairs=((0, 1), (1, 2)),
        symmetry_weight=0.2,
        safety_bounds=(-0.6, 0.6),
        safety_weight=0.3,
    )
    params = np.array([0.5, -0.4, 0.8], dtype=float)
    evaluation = objective.evaluate(params)
    finite_difference = _finite_difference_gradient(objective, params)

    assert objective.term_names == (
        "phase_energy",
        "phase_fidelity_target",
        "periodic_regularization",
        "phase_symmetry_penalty",
        "smooth_box_safety_penalty",
    )
    assert not objective.parameter_shift_compatible
    assert evaluation.value > 0.0
    assert len(evaluation.terms) == 5
    np.testing.assert_allclose(evaluation.gradient, finite_difference, rtol=1e-5, atol=1e-6)
    payload = evaluation.to_dict()
    assert payload["parameter_shift_compatible"] is False
    terms = payload["terms"]
    assert isinstance(terms, list)
    safety = terms[-1]
    assert isinstance(safety, dict)
    assert safety["gradient_mode"] == "analytic"
    assert objective.to_dict()["name"] == "phase_control_objective"


def test_parameter_shift_compatible_objective_fails_closed_when_safety_added() -> None:
    """Keep analytic safety penalties outside parameter-shift-only objectives."""
    compatible = ComposedPhaseObjective(
        terms=(phase_energy_term(2),),
    )
    incompatible = ComposedPhaseObjective(
        terms=(
            phase_energy_term(2),
            smooth_box_safety_penalty_term(-1.0, 1.0, width=2),
        ),
    )

    compatible.require_parameter_shift_compatible()
    with pytest.raises(ValueError, match="non-parameter-shift"):
        incompatible.require_parameter_shift_compatible()


def test_composed_phase_objective_training_decreases_and_certifies() -> None:
    """Train a mixed objective and serialise its accepted-step certificate."""
    objective = build_phase_control_objective(
        2,
        energy_weight=1.0,
        fidelity_target=np.zeros(2, dtype=float),
        fidelity_weight=0.2,
        safety_bounds=(-1.0, 1.0),
        safety_weight=0.1,
    )
    result = train_composed_phase_objective(
        objective,
        np.array([0.8, -0.7], dtype=float),
        learning_rate=0.4,
        max_steps=40,
        gradient_tolerance=1e-7,
    )
    certificate = validate_composed_objective_training(
        result,
        min_decrease=0.1,
    )

    assert result.accepted_steps > 0
    assert result.rejected_steps == 0
    assert result.best_value < result.initial_value
    assert certificate.monotone_accepted_values
    assert certificate.min_decrease_satisfied
    assert not certificate.parameter_shift_compatible
    assert "term-gradient" in certificate.claim_boundary
    result_payload = result.to_dict()
    certificate_payload = certificate.to_dict()
    assert result_payload["accepted_steps"] == result.accepted_steps
    assert certificate_payload["value_decrease"] == certificate.value_decrease
    assert len(result.accepted_value_history) == result.accepted_steps + 1


def test_composed_phase_objective_rejects_invalid_boundaries() -> None:
    """Reject invalid dimensions, safety bounds, and incomplete target gates."""
    with pytest.raises(ValueError, match="width"):
        phase_energy_term(0)

    with pytest.raises(ValueError, match="lower bounds"):
        smooth_box_safety_penalty_term(1.0, -1.0, width=2)

    with pytest.raises(ValueError, match="at least one term"):
        ComposedPhaseObjective(terms=())

    objective = build_phase_control_objective(1)
    result = train_composed_phase_objective(objective, np.array([0.2], dtype=float), max_steps=2)
    with pytest.raises(ValueError, match="target_value_tolerance"):
        validate_composed_objective_training(result, target_value_tolerance=1e-6)


def test_objective_term_rejects_inconsistent_metadata() -> None:
    """Reject term metadata that could misrepresent its gradient semantics."""

    def value(params: FloatArray) -> float:
        return float(np.sum(params))

    def gradient(params: FloatArray) -> FloatArray:
        return np.ones_like(params)

    term = ObjectiveTerm(
        name="linear",
        kind="diagnostic",
        weight=1.0,
        value_fn=value,
        gradient_fn=gradient,
        gradient_mode="analytic",
        parameter_shift_compatible=False,
        description="linear diagnostic term",
    )
    with pytest.raises(ValueError, match="name"):
        replace(term, name="")
    with pytest.raises(ValueError, match="kind"):
        replace(term, kind="")
    with pytest.raises(ValueError, match="non-negative"):
        replace(term, weight=-0.1)
    with pytest.raises(ValueError, match="finite"):
        replace(term, weight=float("nan"))
    with pytest.raises(ValueError, match="gradient_mode"):
        replace(term, gradient_mode="automatic")
    with pytest.raises(ValueError, match="analytic-only"):
        replace(term, parameter_shift_compatible=True)


def test_objective_term_rejects_non_finite_values_and_invalid_gradients() -> None:
    """Validate callback results before admitting them into objective evidence."""

    def non_finite_value(params: FloatArray) -> float:
        del params
        return float("inf")

    def wrong_gradient(params: FloatArray) -> FloatArray:
        return np.ones(params.size + 1, dtype=np.float64)

    non_finite = ObjectiveTerm(
        name="non_finite",
        kind="diagnostic",
        weight=1.0,
        value_fn=non_finite_value,
        gradient_fn=lambda params: np.ones_like(params),
        gradient_mode="analytic",
        parameter_shift_compatible=False,
        description="non-finite callback probe",
    )
    wrong_shape = ObjectiveTerm(
        name="wrong_shape",
        kind="diagnostic",
        weight=1.0,
        value_fn=lambda params: float(np.sum(params)),
        gradient_fn=wrong_gradient,
        gradient_mode="analytic",
        parameter_shift_compatible=False,
        description="gradient-shape callback probe",
    )

    with pytest.raises(ValueError, match="term value must be finite"):
        non_finite.evaluate([0.0])
    with pytest.raises(ValueError, match=r"shape \(1,\)"):
        wrong_shape.gradient([0.0])


def test_composed_objective_rejects_invalid_identity_and_duplicate_terms() -> None:
    """Refuse unnamed objectives and duplicate term identities."""
    term = phase_energy_term(1)
    with pytest.raises(ValueError, match="objective name"):
        ComposedPhaseObjective(terms=(term,), name="")
    with pytest.raises(ValueError, match="unique"):
        ComposedPhaseObjective(terms=(term, term))


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ([], "non-empty"),
        ([[0.0]], "one-dimensional"),
        ([0.0, 1.0], "shape"),
        ([float("inf")], "finite"),
    ],
)
def test_phase_term_evaluation_rejects_invalid_parameter_vectors(
    params: ArrayLike, message: str
) -> None:
    """Reject empty, non-vector, wrong-width, and non-finite parameter inputs."""
    term = phase_energy_term(1)
    with pytest.raises(ValueError, match=message):
        term.evaluate(params)


def test_term_builders_accept_vector_weights_and_preserve_periodic_gradients() -> None:
    """Exercise vector-weight, target, regularisation, symmetry, and offset contracts."""
    params = np.array([0.2, -0.4], dtype=np.float64)
    terms = (
        phase_energy_term(2, weights=np.array([0.5, 1.5], dtype=np.float64)),
        phase_fidelity_target_term(np.array([0.1, -0.2], dtype=np.float64)),
        periodic_regularization_term(np.zeros(2, dtype=np.float64)),
        phase_symmetry_penalty_term(2, ((0, 1),), offsets=np.array([0.3])),
    )
    objective = ComposedPhaseObjective(terms=terms)
    np.testing.assert_allclose(
        objective.evaluate(params).gradient,
        _finite_difference_gradient(objective, params),
        rtol=1e-5,
        atol=1e-6,
    )
    assert objective.parameter_shift_compatible


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: phase_energy_term(2, weights=[1.0]), "shape"),
        (lambda: phase_symmetry_penalty_term(0, ((0, 1),)), "width"),
        (lambda: phase_symmetry_penalty_term(2, ()), "at least one"),
        (lambda: phase_symmetry_penalty_term(2, ((0, 0),)), "distinct"),
        (lambda: phase_symmetry_penalty_term(2, ((0, 2),)), "out of bounds"),
        (lambda: phase_symmetry_penalty_term(2, ((0, 1),), offsets=[0.0, 1.0]), "shape"),
        (lambda: smooth_box_safety_penalty_term(-1.0, 1.0, width=0), "width"),
        (
            lambda: smooth_box_safety_penalty_term(-1.0, 1.0, width=1, sharpness=0.0),
            "positive",
        ),
    ],
)
def test_term_builders_reject_invalid_dimensions_and_symmetry_contracts(
    factory: Callable[[], ObjectiveTerm], message: str
) -> None:
    """Fail closed on malformed term dimensions, pairs, offsets, and sharpness."""
    with pytest.raises(ValueError, match=message):
        factory()


def test_phase_objective_builder_rejects_empty_or_invalid_compositions() -> None:
    """Reject non-positive width and configurations that select no objective term."""
    with pytest.raises(ValueError, match="width"):
        build_phase_control_objective(0)
    with pytest.raises(ValueError, match="at least one term"):
        build_phase_control_objective(
            2,
            energy_weight=0.0,
            fidelity_target=np.zeros(2),
            fidelity_weight=0.0,
            regularization_center=np.zeros(2),
            regularization_weight=0.0,
            symmetry_pairs=((0, 1),),
            symmetry_weight=0.0,
            safety_bounds=(-1.0, 1.0),
            safety_weight=0.0,
        )


def test_training_rejects_invalid_optimiser_controls() -> None:
    """Reject non-positive or out-of-range line-search controls."""
    objective = ComposedPhaseObjective(terms=(phase_energy_term(1),))
    params = np.array([0.2], dtype=np.float64)
    with pytest.raises(ValueError, match="learning_rate"):
        train_composed_phase_objective(objective, params, learning_rate=0.0)
    with pytest.raises(ValueError, match="max_steps"):
        train_composed_phase_objective(objective, params, max_steps=0)
    with pytest.raises(ValueError, match="max_steps"):
        train_composed_phase_objective(objective, params, max_steps=True)
    with pytest.raises(ValueError, match="gradient_tolerance"):
        train_composed_phase_objective(objective, params, gradient_tolerance=-1.0)
    with pytest.raises(ValueError, match="sufficient_decrease"):
        train_composed_phase_objective(objective, params, sufficient_decrease=0.0)
    with pytest.raises(ValueError, match="smaller than one"):
        train_composed_phase_objective(objective, params, backtracking_factor=1.0)
    with pytest.raises(ValueError, match="max_backtracks"):
        train_composed_phase_objective(objective, params, max_backtracks=0)


def test_training_reports_initial_convergence_and_line_search_failure() -> None:
    """Distinguish an initially stationary objective from a rejected descent step."""
    stationary = ComposedPhaseObjective(terms=(phase_energy_term(1),))
    converged = train_composed_phase_objective(stationary, np.zeros(1, dtype=np.float64))
    assert converged.converged
    assert converged.reason == "gradient_tolerance"
    assert not converged.steps

    constant = ObjectiveTerm(
        name="constant_with_gradient",
        kind="line_search_probe",
        weight=1.0,
        value_fn=lambda params: float(params.size),
        gradient_fn=lambda params: np.ones_like(params),
        gradient_mode="analytic",
        parameter_shift_compatible=False,
        description="constant value with a deliberately non-zero gradient callback",
    )
    rejected = train_composed_phase_objective(
        ComposedPhaseObjective(terms=(constant,)),
        np.array([0.0], dtype=np.float64),
        max_steps=2,
        max_backtracks=2,
    )
    assert rejected.reason == "line_search_failed"
    assert rejected.rejected_steps == 1
    assert rejected.steps[0].accepted is False
    assert rejected.steps[0].to_dict()["backtracks"] == 3


def test_training_detects_convergence_after_the_final_allowed_step() -> None:
    """Mark convergence when the last admitted step reaches zero gradient."""

    def quadratic_value(params: FloatArray) -> float:
        return float(0.5 * np.dot(params, params))

    quadratic = ObjectiveTerm(
        name="quadratic",
        kind="convergence_probe",
        weight=1.0,
        value_fn=quadratic_value,
        gradient_fn=lambda params: params.copy(),
        gradient_mode="analytic",
        parameter_shift_compatible=False,
        description="quadratic objective with exact analytic gradient",
    )
    result = train_composed_phase_objective(
        ComposedPhaseObjective(terms=(quadratic,)),
        np.array([1.0], dtype=np.float64),
        learning_rate=1.0,
        max_steps=1,
    )
    assert result.converged
    assert result.reason == "gradient_tolerance"
    assert result.accepted_steps == 1
    assert result.best_value == 0.0


def test_training_validation_exercises_optional_gates_and_refusals() -> None:
    """Evaluate gradient, target, tolerance, and decrease gates independently."""
    objective = ComposedPhaseObjective(terms=(phase_energy_term(1),))
    result = train_composed_phase_objective(
        objective,
        np.array([0.3], dtype=np.float64),
        learning_rate=0.5,
        max_steps=4,
    )
    certificate = validate_composed_objective_training(
        result,
        gradient_tolerance=1.0,
        target_value=result.best_value,
        target_value_tolerance=0.0,
        min_decrease=0.0,
    )
    assert isinstance(certificate, ComposedObjectiveTrainingCertificate)
    assert certificate.within_gradient_tolerance is True
    assert certificate.within_target_value_tolerance is True
    assert certificate.min_decrease_satisfied is True

    target_without_tolerance = validate_composed_objective_training(
        result,
        target_value=result.best_value,
    )
    assert target_without_tolerance.within_target_value_tolerance is True

    with pytest.raises(ValueError, match="gradient_tolerance"):
        validate_composed_objective_training(result, gradient_tolerance=-1.0)
    with pytest.raises(ValueError, match="target_value_tolerance"):
        validate_composed_objective_training(
            result,
            target_value=0.0,
            target_value_tolerance=-1.0,
        )
    with pytest.raises(ValueError, match="min_decrease"):
        validate_composed_objective_training(result, min_decrease=-1.0)
