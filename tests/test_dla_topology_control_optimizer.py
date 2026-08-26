# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — topology-control projected-optimizer tests
"""Strict-decrease, projection, rejection, and custody tests."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from scpn_quantum_control.dla_topology_control.objectives import (
    ParityProtectedQuadraticObjective,
)
from scpn_quantum_control.dla_topology_control.optimizer import (
    ParityProjectedOptimisationTrace,
    ProjectedGradientConfig,
    ProjectedGradientStep,
    optimise_parity_protected_state,
)
from scpn_quantum_control.dla_topology_control.parity import ParitySectorProjector
from scpn_quantum_control.dla_topology_control.schema import ParitySector


def _objective() -> ParityProtectedQuadraticObjective:
    projector = ParitySectorProjector(3, ParitySector.EVEN)
    target = projector.project(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.5j, -0.2, 0.0]))
    target = target / np.linalg.norm(target)
    return ParityProtectedQuadraticObjective(projector, target, leakage_weight=2.0)


def test_projected_gradient_strictly_decreases_and_removes_leakage() -> None:
    """Project every accepted proposal inside the selected parity sector."""
    rng = np.random.default_rng(54)
    objective = _objective()
    initial = objective.target_state + 0.4 * (rng.normal(size=8) + 1j * rng.normal(size=8))
    trace = optimise_parity_protected_state(
        initial,
        objective,
        ProjectedGradientConfig(max_steps=24, initial_step_size=0.5),
    )
    assert trace.accepted_steps == len(trace.steps) == 24
    assert all(step.accepted for step in trace.steps)
    assert all(step.proposed_value < step.original_value for step in trace.steps)
    assert objective(trace.final_state) < objective(initial)
    assert objective.evaluate(trace.final_state).leakage_mass == pytest.approx(0.0)
    np.testing.assert_array_equal(
        objective.projector.project(trace.final_state), trace.final_state
    )
    assert not trace.initial_state.flags.writeable
    assert not trace.final_state.flags.writeable
    assert len(trace.content_digest) == 64


def test_zero_gradient_returns_empty_immutable_trace() -> None:
    """Stop without inventing a step when the initial state is already optimal."""
    objective = _objective()
    trace = optimise_parity_protected_state(objective.target_state, objective)
    assert trace.steps == ()
    assert trace.accepted_steps == 0
    np.testing.assert_array_equal(trace.initial_state, trace.final_state)


def test_oversized_step_can_fail_closed_without_state_change() -> None:
    """Record a rejected zero-size proposal when no backtracking is allowed."""
    objective = _objective()
    initial = objective.target_state * 2.0
    trace = optimise_parity_protected_state(
        initial,
        objective,
        ProjectedGradientConfig(
            max_steps=2,
            initial_step_size=10.0,
            max_backtracks=0,
            minimum_step_size=1.0,
        ),
    )
    assert len(trace.steps) == 1
    step = trace.steps[0]
    assert not step.accepted
    assert step.step_size == 0.0
    assert step.original_value == step.proposed_value
    np.testing.assert_array_equal(trace.final_state, initial)


def test_backtracking_recovers_from_an_oversized_initial_step() -> None:
    """Contract step size until a strict projected decrease is found."""
    objective = _objective()
    initial = objective.target_state * 2.0
    trace = optimise_parity_protected_state(
        initial,
        objective,
        ProjectedGradientConfig(max_steps=1, initial_step_size=10.0),
    )
    assert trace.steps[0].accepted
    assert trace.steps[0].backtracks > 0
    assert trace.steps[0].step_size < 10.0


def test_backtracking_stops_below_minimum_step_size() -> None:
    """Reject immediately when contraction crosses the configured step floor."""
    objective = _objective()
    initial = objective.target_state * 2.0
    trace = optimise_parity_protected_state(
        initial,
        objective,
        ProjectedGradientConfig(
            max_steps=1,
            initial_step_size=10.0,
            contraction=0.1,
            max_backtracks=12,
            minimum_step_size=2.0,
        ),
    )
    assert len(trace.steps) == 1
    assert not trace.steps[0].accepted
    assert trace.steps[0].backtracks == 0


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"max_steps": 0}, "max_steps"),
        ({"max_steps": True}, "max_steps"),
        ({"max_backtracks": -1}, "max_backtracks"),
        ({"max_backtracks": True}, "max_backtracks"),
        ({"initial_step_size": 0.0}, "initial_step_size"),
        ({"initial_step_size": np.nan}, "initial_step_size"),
        ({"contraction": 0.0}, "contraction"),
        ({"contraction": 1.0}, "contraction"),
        ({"gradient_tolerance": -1.0}, "gradient_tolerance"),
        ({"minimum_step_size": 0.0}, "minimum_step_size"),
        ({"initial_step_size": 0.1, "minimum_step_size": 0.2}, "must not exceed"),
    ],
)
def test_optimizer_config_rejects_invalid_values(changes: dict[str, object], message: str) -> None:
    """Reject invalid step, backtracking, and convergence configuration."""
    with pytest.raises(ValueError, match=message):
        ProjectedGradientConfig(**changes)


def test_optimizer_rejects_wrong_objective_config_and_initial_state() -> None:
    """Require exact owner contracts and a finite correctly sized initial state."""
    objective = _objective()
    with pytest.raises(ValueError, match="objective"):
        optimise_parity_protected_state(
            np.zeros(8), cast(ParityProtectedQuadraticObjective, object())
        )
    with pytest.raises(ValueError, match="config"):
        optimise_parity_protected_state(
            np.zeros(8), objective, cast(ProjectedGradientConfig, object())
        )
    with pytest.raises(ValueError, match="shape"):
        optimise_parity_protected_state(np.zeros(7), objective)


def test_step_contract_rejects_invalid_indices_scalars_and_acceptance() -> None:
    """Reject malformed accepted/rejected step records."""
    valid: dict[str, object] = {
        "index": 0,
        "accepted": True,
        "backtracks": 0,
        "step_size": 0.5,
        "original_value": 1.0,
        "proposed_value": 0.5,
        "leakage_before": 0.2,
        "leakage_after": 0.0,
        "gradient_norm": 1.0,
        "state": np.ones(4, dtype=np.complex128),
    }
    with pytest.raises(ValueError, match="index"):
        ProjectedGradientStep(**(valid | {"index": True}))
    with pytest.raises(ValueError, match="backtracks"):
        ProjectedGradientStep(**(valid | {"backtracks": -1}))
    for key in (
        "step_size",
        "original_value",
        "proposed_value",
        "leakage_before",
        "leakage_after",
        "gradient_norm",
    ):
        with pytest.raises(ValueError, match=key):
            ProjectedGradientStep(**(valid | {key: -1.0}))
    with pytest.raises(ValueError, match="accepted steps"):
        ProjectedGradientStep(**(valid | {"proposed_value": 1.0}))
    with pytest.raises(ValueError, match="rejected steps"):
        ProjectedGradientStep(**(valid | {"accepted": False}))
    with pytest.raises(ValueError, match="state"):
        ProjectedGradientStep(**(valid | {"state": np.ones((2, 2))}))


def test_trace_contract_rejects_misaligned_arrays_digest_and_boundary() -> None:
    """Reject inconsistent trace arrays, steps, digest, and claim boundary."""
    objective = _objective()
    trace = optimise_parity_protected_state(
        objective.target_state * 2.0,
        objective,
        ProjectedGradientConfig(max_steps=1),
    )
    values: dict[str, object] = {
        "initial_state": trace.initial_state,
        "final_state": trace.final_state,
        "steps": trace.steps,
        "content_digest": trace.content_digest,
    }
    with pytest.raises(ValueError, match="initial_state"):
        ParityProjectedOptimisationTrace(**(values | {"initial_state": np.ones((2, 2))}))
    with pytest.raises(ValueError, match="final_state"):
        ParityProjectedOptimisationTrace(**(values | {"final_state": np.ones(7)}))
    mismatched_step = ProjectedGradientStep(
        index=0,
        accepted=False,
        backtracks=0,
        step_size=0.0,
        original_value=1.0,
        proposed_value=1.0,
        leakage_before=0.0,
        leakage_after=0.0,
        gradient_norm=0.0,
        state=np.ones(4, dtype=np.complex128),
    )
    with pytest.raises(ValueError, match="every step"):
        ParityProjectedOptimisationTrace(**(values | {"steps": (mismatched_step,)}))
    with pytest.raises(ValueError, match="content_digest"):
        ParityProjectedOptimisationTrace(**(values | {"content_digest": "bad"}))
    with pytest.raises(ValueError, match="claim_boundary"):
        ParityProjectedOptimisationTrace(**(values | {"claim_boundary": " "}))
