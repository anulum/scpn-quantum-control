# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase Objectives
"""Tests for phase/objectives.py composed differentiable objectives."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.phase import (
    ComposedPhaseObjective,
    build_phase_control_objective,
    phase_energy_term,
    smooth_box_safety_penalty_term,
    train_composed_phase_objective,
    validate_composed_objective_training,
)


def _finite_difference_gradient(
    objective: ComposedPhaseObjective, params: np.ndarray
) -> np.ndarray:
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
    assert payload["terms"][-1]["gradient_mode"] == "analytic"


def test_parameter_shift_compatible_objective_fails_closed_when_safety_added() -> None:
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


def test_composed_phase_objective_rejects_invalid_boundaries() -> None:
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
