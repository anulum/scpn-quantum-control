# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Synchronisation Objectives
"""Tests for differentiable synchronisation phase losses."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control import phase
from scpn_quantum_control.phase import ComposedPhaseObjective
from scpn_quantum_control.phase.synchronisation_objectives import (
    SYNCHRONISATION_OBJECTIVE_CLAIM_BOUNDARY,
    build_synchronisation_objective,
    cluster_synchronisation_target_term,
    kuramoto_order_parameter,
    kuramoto_order_parameter_gradient,
    kuramoto_order_parameter_target_term,
    phase_locking_target_term,
)

FloatArray = NDArray[np.float64]


def _finite_difference_gradient(
    objective: Callable[[FloatArray], float],
    params: FloatArray,
    *,
    step: float = 1.0e-6,
) -> FloatArray:
    gradient = np.zeros_like(params, dtype=np.float64)
    for index in range(params.size):
        plus = params.copy()
        minus = params.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (objective(plus) - objective(minus)) / (2.0 * step)
    return gradient


def test_kuramoto_order_parameter_loss_matches_finite_difference() -> None:
    params = np.array([0.15, 0.45, -0.2, 0.35], dtype=np.float64)
    term = kuramoto_order_parameter_target_term(4, target=0.82, term_weight=1.7)

    value = term.evaluate(params)
    gradient = term.gradient(params)
    finite_difference = _finite_difference_gradient(
        lambda vector: term.evaluate(vector).weighted_value,
        params,
    )

    assert phase.kuramoto_order_parameter is kuramoto_order_parameter
    assert phase.SYNCHRONISATION_OBJECTIVE_CLAIM_BOUNDARY == (
        SYNCHRONISATION_OBJECTIVE_CLAIM_BOUNDARY
    )
    assert value.kind == "synchronisation_order_parameter"
    assert value.gradient_mode == "analytic"
    assert value.parameter_shift_compatible is False
    assert kuramoto_order_parameter(params) == pytest.approx(abs(np.mean(np.exp(1j * params))))
    np.testing.assert_allclose(
        kuramoto_order_parameter_gradient(params),
        _finite_difference_gradient(kuramoto_order_parameter, params),
        atol=1.0e-6,
    )
    np.testing.assert_allclose(gradient, finite_difference, atol=1.0e-6)


def test_phase_locking_loss_matches_finite_difference_and_parameter_shift_mode() -> None:
    params = np.array([0.2, 0.5, -0.4, 0.1], dtype=np.float64)
    term = phase_locking_target_term(
        4,
        [(0, 1), (2, 3)],
        offsets=[0.1, -0.2],
        term_weight=0.5,
    )

    evaluation = term.evaluate(params)
    gradient = term.gradient(params)
    finite_difference = _finite_difference_gradient(
        lambda vector: term.evaluate(vector).weighted_value,
        params,
    )

    assert evaluation.name == "phase_locking_target"
    assert evaluation.kind == "phase_locking"
    assert evaluation.gradient_mode == "parameter_shift"
    assert evaluation.parameter_shift_compatible is True
    np.testing.assert_allclose(gradient, finite_difference, atol=1.0e-6)


def test_cluster_synchronisation_loss_matches_finite_difference() -> None:
    params = np.array([0.1, 0.35, -0.2, 1.4, 1.65, 1.9], dtype=np.float64)
    term = cluster_synchronisation_target_term(
        6,
        [(0, 1, 2), (3, 4, 5)],
        targets=[0.9, 0.85],
        term_weight=2.0,
    )

    evaluation = term.evaluate(params)
    gradient = term.gradient(params)
    finite_difference = _finite_difference_gradient(
        lambda vector: term.evaluate(vector).weighted_value,
        params,
    )

    assert evaluation.kind == "cluster_synchronisation"
    assert evaluation.gradient_mode == "analytic"
    assert evaluation.parameter_shift_compatible is False
    np.testing.assert_allclose(gradient, finite_difference, atol=1.0e-6)


def test_build_synchronisation_objective_composes_all_loss_families() -> None:
    params = np.array([0.1, 0.35, -0.2, 1.4, 1.65, 1.9], dtype=np.float64)
    objective = build_synchronisation_objective(
        6,
        order_parameter_target=0.75,
        order_parameter_weight=0.6,
        phase_locking_pairs=[(0, 1), (3, 4)],
        phase_locking_offsets=0.0,
        phase_locking_weight=0.4,
        clusters=[(0, 1, 2), (3, 4, 5)],
        cluster_targets=0.9,
        cluster_weight=0.8,
    )
    evaluation = objective.evaluate(params)
    finite_difference = _finite_difference_gradient(objective, params)
    payload = objective.to_dict()

    assert isinstance(objective, ComposedPhaseObjective)
    assert objective.term_names == (
        "kuramoto_order_parameter_target",
        "phase_locking_target",
        "cluster_synchronisation_target",
    )
    assert objective.parameter_shift_compatible is False
    assert payload["claim_boundary"] == SYNCHRONISATION_OBJECTIVE_CLAIM_BOUNDARY
    assert "hardware" in evaluation.claim_boundary
    np.testing.assert_allclose(evaluation.gradient, finite_difference, atol=1.0e-6)


def test_order_parameter_zero_value_and_singular_gradient_boundary() -> None:
    incoherent = np.array([0.0, np.pi], dtype=np.float64)

    assert kuramoto_order_parameter(incoherent) < 1.0e-12
    with pytest.raises(ValueError, match="singular"):
        kuramoto_order_parameter_gradient(incoherent)
    with pytest.raises(ValueError, match="singular"):
        kuramoto_order_parameter_target_term(2).gradient(incoherent)
    with pytest.raises(ValueError, match="singular"):
        cluster_synchronisation_target_term(2, [(0, 1)]).gradient(incoherent)


def test_synchronisation_objectives_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="width"):
        kuramoto_order_parameter_target_term(0)
    with pytest.raises(ValueError, match="width"):
        kuramoto_order_parameter_target_term(True)
    with pytest.raises(ValueError, match="target"):
        kuramoto_order_parameter_target_term(2, target=1.5)
    with pytest.raises(ValueError, match="target"):
        kuramoto_order_parameter_target_term(2, target=float("nan"))
    with pytest.raises(ValueError, match="min_order_parameter"):
        kuramoto_order_parameter_target_term(2, min_order_parameter=-1.0)
    with pytest.raises(ValueError, match="phase vector"):
        kuramoto_order_parameter([])
    with pytest.raises(ValueError, match="shape"):
        kuramoto_order_parameter_target_term(2).evaluate(np.array([0.1], dtype=np.float64))
    with pytest.raises(ValueError, match="finite phases"):
        kuramoto_order_parameter([0.1, float("nan")])

    with pytest.raises(ValueError, match="pairs"):
        phase_locking_target_term(2, [])
    with pytest.raises(ValueError, match="integers"):
        phase_locking_target_term(2, [(True, 1)])
    with pytest.raises(ValueError, match="distinct"):
        phase_locking_target_term(2, [(0, 0)])
    with pytest.raises(ValueError, match="out of bounds"):
        phase_locking_target_term(2, [(0, 2)])
    with pytest.raises(ValueError, match="shape"):
        phase_locking_target_term(2, [(0, 1)], offsets=[0.0, 0.1])

    with pytest.raises(ValueError, match="clusters"):
        cluster_synchronisation_target_term(2, [])
    with pytest.raises(ValueError, match="at least two"):
        cluster_synchronisation_target_term(2, [(0,)])
    with pytest.raises(ValueError, match="unique"):
        cluster_synchronisation_target_term(3, [(0, 0)])
    with pytest.raises(ValueError, match="integers"):
        cluster_synchronisation_target_term(3, [(0, True)])
    with pytest.raises(ValueError, match="out of bounds"):
        cluster_synchronisation_target_term(3, [(0, 3)])
    with pytest.raises(ValueError, match="disjoint"):
        cluster_synchronisation_target_term(4, [(0, 1), (1, 2)])
    with pytest.raises(ValueError, match="targets"):
        cluster_synchronisation_target_term(2, [(0, 1)], targets=1.5)
    with pytest.raises(ValueError, match="targets"):
        cluster_synchronisation_target_term(4, [(0, 1), (2, 3)], targets=[0.8])

    with pytest.raises(ValueError, match="at least one term"):
        build_synchronisation_objective(
            2,
            order_parameter_weight=0.0,
            phase_locking_weight=0.0,
            cluster_weight=0.0,
        )
