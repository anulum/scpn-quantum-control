# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Hierarchical objective tests
"""Finite-difference and proposal tests through real objective surfaces."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.chimera_control.objectives import (
    PhaseControlProposal,
    build_chimera_control_objective,
    propose_phase_control_step,
)
from scpn_quantum_control.chimera_control.schema import (
    ChimeraControlSpecification,
    HierarchyTarget,
    two_population_hierarchy,
)
from scpn_quantum_control.phase.objectives import ComposedPhaseObjective, ObjectiveTerm


def _specification() -> ChimeraControlSpecification:
    hierarchy = two_population_hierarchy(3)
    return ChimeraControlSpecification(
        hierarchy,
        (
            HierarchyTarget("population", (1.0, 0.45), weight=1.0),
            HierarchyTarget("ensemble", (0.68,), weight=0.4),
        ),
    )


def test_composed_hierarchy_objective_has_analytic_finite_difference_gradient() -> None:
    """Match the composed analytic hierarchy gradient to finite differences."""
    objective = build_chimera_control_objective(_specification())
    phases = np.array([0.1, 0.2, -0.1, -1.0, 0.7, 2.1])
    evaluation = objective.evaluate(phases)
    epsilon = 1.0e-6
    finite = np.empty_like(phases)
    for index in range(phases.size):
        plus = phases.copy()
        minus = phases.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        finite[index] = (objective(plus) - objective(minus)) / (2.0 * epsilon)

    assert objective.term_names == (
        "chimera_population_target",
        "chimera_ensemble_target",
    )
    assert not evaluation.parameter_shift_compatible
    np.testing.assert_allclose(evaluation.gradient, finite, rtol=2.0e-6, atol=2.0e-8)


def test_objective_skips_zero_weight_and_rejects_invalid_threshold_or_empty_weight() -> None:
    """Skip zero-weight rows while rejecting unusable objective specifications."""
    hierarchy = two_population_hierarchy(2)
    mixed = ChimeraControlSpecification(
        hierarchy,
        (
            HierarchyTarget("population", (1.0, 0.5), weight=1.0),
            HierarchyTarget("ensemble", (0.7,), weight=0.0),
        ),
    )
    assert build_chimera_control_objective(mixed).term_names == ("chimera_population_target",)
    zero = ChimeraControlSpecification(
        hierarchy,
        (HierarchyTarget("population", (1.0, 0.5), weight=0.0),),
    )
    with pytest.raises(ValueError, match="positive weight"):
        build_chimera_control_objective(zero)
    for threshold in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="min_order_parameter"):
            build_chimera_control_objective(mixed, min_order_parameter=threshold)


def test_backtracking_proposal_strictly_reduces_real_objective_without_mutation() -> None:
    """Propose an immutable phase step that strictly decreases the objective."""
    objective = build_chimera_control_objective(_specification())
    phases = np.array([0.1, 0.2, -0.1, -1.0, 0.7, 2.1])
    original = phases.copy()
    proposal = propose_phase_control_step(objective, phases, initial_step_size=8.0)

    np.testing.assert_array_equal(phases, original)
    assert proposal.accepted
    assert proposal.proposed_value < proposal.original_value
    assert 0.0 < proposal.step_size <= 8.0
    assert not proposal.phase_delta.flags.writeable
    assert not proposal.proposed_phases.flags.writeable


def test_zero_gradient_and_no_decrease_return_unchanged_unapplied_proposals() -> None:
    """Return an unchanged rejected proposal when no strict decrease exists."""
    constant = ObjectiveTerm(
        name="constant",
        kind="test_contract",
        weight=1.0,
        value_fn=lambda values: 1.0,
        gradient_fn=lambda values: np.zeros_like(values),
        gradient_mode="analytic",
        parameter_shift_compatible=False,
        description="constant public objective contract",
    )
    objective = ComposedPhaseObjective((constant,))
    zero = propose_phase_control_step(objective, np.array([0.0, 1.0]))
    assert not zero.accepted
    assert zero.step_size == 0.0
    assert zero.backtracks == 0

    inconsistent = ObjectiveTerm(
        name="inconsistent",
        kind="test_contract",
        weight=1.0,
        value_fn=lambda values: 1.0,
        gradient_fn=lambda values: np.ones_like(values),
        gradient_mode="analytic",
        parameter_shift_compatible=False,
        description="fixed-value objective with a deliberately inconsistent gradient",
    )
    rejected = propose_phase_control_step(
        ComposedPhaseObjective((inconsistent,)),
        np.array([0.0, 1.0]),
        max_backtracks=2,
    )
    assert not rejected.accepted
    assert rejected.backtracks == 2
    np.testing.assert_array_equal(rejected.proposed_phases, [0.0, 1.0])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"initial_step_size": 0.0}, "initial_step_size"),
        ({"initial_step_size": np.nan}, "initial_step_size"),
        ({"max_backtracks": 0}, "max_backtracks"),
        ({"max_backtracks": True}, "max_backtracks"),
    ],
)
def test_phase_proposal_rejects_invalid_search_arguments(
    kwargs: dict[str, object], message: str
) -> None:
    """Reject invalid backtracking arguments and malformed phase vectors."""
    objective = build_chimera_control_objective(_specification())
    with pytest.raises(ValueError, match=message):
        propose_phase_control_step(objective, np.zeros(6), **kwargs)
    with pytest.raises(ValueError, match="phases"):
        propose_phase_control_step(objective, np.zeros((2, 3)))
    with pytest.raises(ValueError, match="phases"):
        propose_phase_control_step(objective, np.array([np.nan]))


def test_phase_control_proposal_contract_rejects_invalid_custody() -> None:
    """Reject proposal records with contradictory scalar or array custody."""
    valid = dict(
        original_value=1.0,
        proposed_value=0.5,
        step_size=0.1,
        backtracks=1,
        accepted=True,
        phase_delta=np.array([0.1, -0.1]),
        proposed_phases=np.array([0.2, 0.3]),
    )
    for key in ("original_value", "proposed_value", "step_size"):
        values = valid | {key: -1.0}
        with pytest.raises(ValueError, match=key):
            PhaseControlProposal(**values)
    with pytest.raises(ValueError, match="backtracks"):
        PhaseControlProposal(**(valid | {"backtracks": True}))
    with pytest.raises(ValueError, match="backtracks"):
        PhaseControlProposal(**(valid | {"backtracks": 1.5}))
    with pytest.raises(ValueError, match="equal non-empty vectors"):
        PhaseControlProposal(**(valid | {"phase_delta": np.zeros((1, 2))}))
    with pytest.raises(ValueError, match="finite"):
        PhaseControlProposal(**(valid | {"proposed_phases": np.array([np.nan, 0.0])}))
    with pytest.raises(ValueError, match="claim_boundary"):
        PhaseControlProposal(**(valid | {"claim_boundary": " "}))
    with pytest.raises(ValueError, match="accepted proposals"):
        PhaseControlProposal(**(valid | {"step_size": 0.0}))
    with pytest.raises(ValueError, match="rejected proposals"):
        PhaseControlProposal(**(valid | {"accepted": False}))
