# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — adaptive FIM feedback tests
"""Tests for uncertainty-aware adaptive-FIM next-experiment proposals."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

import pytest

from scpn_quantum_control.analysis.adaptive_fim_feedback import (
    ADAPTIVE_FIM_SCHEMA,
    AdaptiveFIMConfig,
    AdaptiveFIMObserverRecord,
    AdaptiveFIMPlan,
    AdaptiveFIMStep,
    FIMWitness,
    adaptive_count_aware_schedule,
    adaptive_lambda_schedule,
    observer_record_from_step,
    plan_adaptive_fim_schedule,
    propose_count_aware_lambda,
    propose_next_lambda,
    wilson_score_interval,
)


def _witness(
    leakage_events: int = 80,
    retention_events: int = 800,
    shots: int = 1024,
) -> FIMWitness:
    return FIMWitness.from_counts(
        leakage_events=leakage_events,
        retention_events=retention_events,
        shots=shots,
        depth=2,
        source="synthetic",
        artifact_id="test-witness",
    )


def test_wilson_interval_matches_known_closed_form() -> None:
    interval = wilson_score_interval(5, 10, z=1.959963984540054)

    assert interval.estimate == 0.5
    assert interval.lower == pytest.approx(0.2365930905)
    assert interval.upper == pytest.approx(0.7634069095)
    assert interval.to_dict()["shots"] == 10


@pytest.mark.parametrize(
    ("events", "shots", "z", "message"),
    [
        (-1, 10, 1.96, "events"),
        (True, 10, 1.96, "events"),
        (1, 0, 1.96, "shots"),
        (1, True, 1.96, "shots"),
        (11, 10, 1.96, "exceed"),
        (1, 10, 0.0, "positive"),
        (1, 10, float("nan"), "finite"),
    ],
)
def test_wilson_interval_rejects_invalid_inputs(
    events: int, shots: int, z: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        wilson_score_interval(events, shots, z=z)


def test_count_witness_round_trip_and_disjoint_contract() -> None:
    witness = _witness()

    assert witness.count_bound is True
    assert witness.leakage == pytest.approx(80 / 1024)
    assert witness.retention == pytest.approx(800 / 1024)
    assert witness.to_dict()["count_bound"] is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"leakage": 1.1, "retention": 0.5}, "leakage"),
        ({"leakage": 0.1, "retention": 0.5, "depth": -1}, "depth"),
        ({"leakage": 0.1, "retention": 0.5, "source": "bad"}, "source"),
        ({"leakage": 0.1, "retention": 0.5, "artifact_id": ""}, "artifact_id"),
        ({"leakage": 0.1, "retention": 0.5, "shots": 0}, "shots"),
        (
            {"leakage": 0.1, "retention": 0.5, "shots": 10, "leakage_events": 1},
            "both event counts",
        ),
        (
            {
                "leakage": 0.1,
                "retention": 0.5,
                "shots": 10,
                "leakage_events": -1,
                "retention_events": 5,
            },
            "leakage_events",
        ),
        (
            {
                "leakage": 0.1,
                "retention": 0.5,
                "shots": 10,
                "leakage_events": 11,
                "retention_events": 0,
            },
            "exceed",
        ),
        (
            {
                "leakage": 0.6,
                "retention": 0.5,
                "shots": 10,
                "leakage_events": 6,
                "retention_events": 5,
            },
            "disjoint",
        ),
        (
            {
                "leakage": 0.2,
                "retention": 0.5,
                "shots": 10,
                "leakage_events": 1,
                "retention_events": 5,
            },
            "leakage must match",
        ),
        (
            {
                "leakage": 0.1,
                "retention": 0.4,
                "shots": 10,
                "leakage_events": 1,
                "retention_events": 5,
            },
            "retention must match",
        ),
    ],
)
def test_witness_validation(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FIMWitness(**kwargs)


@pytest.mark.parametrize("shots", [0, True])
def test_witness_factory_rejects_invalid_shots(shots: int) -> None:
    with pytest.raises(ValueError, match="shots"):
        FIMWitness.from_counts(leakage_events=0, retention_events=0, shots=shots)


def test_count_aware_leakage_decrease_uses_conservative_bound_and_delta_cap() -> None:
    config = AdaptiveFIMConfig(
        target_leakage=0.05,
        step_gain=100.0,
        max_delta_per_batch=0.25,
        min_shots=256,
    )
    step = propose_count_aware_lambda(4.0, _witness(120, 700), config)

    assert step.decision == "decrease"
    assert step.lambda_out == pytest.approx(3.75)
    assert step.interval is not None
    assert step.interval.lower > config.target_leakage
    assert step.count_qualified is True
    assert step.to_dict()["schema"] == ADAPTIVE_FIM_SCHEMA


def test_count_aware_boundary_and_missing_counts_hold() -> None:
    config = AdaptiveFIMConfig(target_leakage=0.05, min_shots=256)
    boundary = propose_count_aware_lambda(2.0, _witness(25, 430, 512), config)
    missing = propose_count_aware_lambda(
        2.0,
        FIMWitness(leakage=0.2, retention=0.7),
        config,
    )
    underpowered = propose_count_aware_lambda(2.0, _witness(3, 25, 32), config)

    assert boundary.decision == "hold"
    assert boundary.interval is not None
    assert boundary.error_signal <= 0.0
    assert missing.decision == "hold" and missing.interval is None
    assert underpowered.decision == "hold" and underpowered.interval is None


def test_retention_recovery_uses_upper_bound_and_never_increases() -> None:
    config = AdaptiveFIMConfig(
        mode="retention_recovery",
        target_retention=0.95,
        step_gain=2.0,
        min_shots=256,
    )
    decrease = propose_count_aware_lambda(3.0, _witness(50, 700), config)
    hold = propose_count_aware_lambda(3.0, _witness(10, 1000), config)

    assert decrease.decision == "decrease"
    assert decrease.interval is not None and decrease.interval.upper < 0.95
    assert hold.decision == "hold"
    assert hold.lambda_out == 3.0


def test_zero_gain_and_lambda_floor_hold() -> None:
    witness = _witness(120, 700)
    zero_gain = propose_count_aware_lambda(
        2.0,
        witness,
        AdaptiveFIMConfig(target_leakage=0.05, step_gain=0.0),
    )
    floor = propose_count_aware_lambda(
        0.0,
        witness,
        AdaptiveFIMConfig(target_leakage=0.05, step_gain=2.0),
    )

    assert zero_gain.decision == "hold"
    assert floor.decision == "hold"
    assert floor.clipped is True


def test_count_schedule_threads_only_qualified_decreases() -> None:
    config = AdaptiveFIMConfig(target_leakage=0.05, step_gain=4.0, min_shots=256)
    steps = adaptive_count_aware_schedule(
        2.0,
        (_witness(80, 800), _witness(25, 430, 512), _witness(90, 780)),
        config,
    )

    assert [step.index for step in steps] == [0, 1, 2]
    assert [step.decision for step in steps] == ["decrease", "hold", "decrease"]
    assert steps[1].lambda_in == steps[0].lambda_out
    assert steps[2].lambda_in == steps[1].lambda_out


def test_hardware_safe_budget_precedes_schedule_and_observer_generation() -> None:
    witnesses = (_witness(), _witness())
    allowed = plan_adaptive_fim_schedule(
        4.0,
        witnesses,
        policy_id="ci_dry_run_only",
        shots_per_arm=128,
        config=AdaptiveFIMConfig(target_leakage=0.05),
    )
    refused = plan_adaptive_fim_schedule(
        4.0,
        witnesses,
        policy_id="ci_dry_run_only",
        shots_per_arm=4096,
    )

    assert allowed.allowed is True
    assert allowed.budget.evaluations == 4
    assert allowed.budget.estimated_total_shots == 512
    assert len(allowed.steps) == len(allowed.observers) == 2
    assert allowed.to_dict()["closed_loop_validated"] is False
    assert refused.allowed is False
    assert refused.steps == () and refused.observers == ()
    assert refused.to_dict()["hardware_execution"] is False


def test_hardware_request_refuses_before_schedule() -> None:
    plan = plan_adaptive_fim_schedule(
        4.0,
        (_witness(),),
        policy_id="default_no_submit",
        shots_per_arm=128,
        request_hardware=True,
    )

    assert plan.allowed is False
    assert any("hardware execution" in blocker for blocker in plan.blockers)
    assert plan.steps == ()


def test_plan_and_observer_contract_reject_invalid_states() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        plan_adaptive_fim_schedule(
            1.0,
            (),
            policy_id="ci_dry_run_only",
            shots_per_arm=64,
        )
    step = propose_count_aware_lambda(1.0, _witness(), AdaptiveFIMConfig(target_leakage=0.05))
    observer = observer_record_from_step(step, policy_id="ci_dry_run_only")
    with pytest.raises(ValueError, match="policy_id"):
        observer_record_from_step(step, policy_id="")
    with pytest.raises(ValueError, match="claim boundary"):
        replace(observer, claim_boundary="broader observer claims")
    with pytest.raises(ValueError, match="hardware execution"):
        AdaptiveFIMObserverRecord(
            "id",
            "hold",
            1.0,
            1.0,
            None,
            None,
            1024,
            "policy",
            "synthetic",
            hardware_execution=True,
        )
    with pytest.raises(ValueError, match="both present"):
        AdaptiveFIMObserverRecord(
            "id",
            "hold",
            1.0,
            1.0,
            0.0,
            None,
            1024,
            "policy",
            "synthetic",
        )


def test_step_and_plan_invariants_fail_closed() -> None:
    witness = _witness()
    step = propose_count_aware_lambda(1.0, witness, AdaptiveFIMConfig(target_leakage=0.05))
    with pytest.raises(ValueError, match="unknown adaptive FIM feedback schema"):
        replace(step, schema="adaptive_fim_feedback.v2")
    with pytest.raises(ValueError, match="claim boundary"):
        replace(step, claim_boundary="broader proposal claims")
    with pytest.raises(ValueError, match="non-negative"):
        replace(step, index=-1)
    with pytest.raises(ValueError, match="lambda values"):
        replace(step, lambda_in=-1.0, lambda_out=-2.0)
    with pytest.raises(ValueError, match="reduce"):
        replace(step, lambda_out=step.lambda_in)
    with pytest.raises(ValueError, match="interval presence"):
        replace(step, count_qualified=False)
    with pytest.raises(ValueError, match="rationale"):
        replace(step, rationale="")

    plan = plan_adaptive_fim_schedule(
        1.0,
        (witness,),
        policy_id="ci_dry_run_only",
        shots_per_arm=64,
        config=AdaptiveFIMConfig(target_leakage=0.05),
    )
    with pytest.raises(ValueError, match="unknown adaptive FIM feedback schema"):
        replace(plan, schema="adaptive_fim_feedback.v2")
    with pytest.raises(ValueError, match="claim boundary"):
        replace(plan, claim_boundary="broader plan claims")
    with pytest.raises(ValueError, match="allowed must match"):
        replace(plan, outcome="refused")
    with pytest.raises(ValueError, match="cannot list blockers"):
        replace(plan, blockers=("bad",))
    with pytest.raises(ValueError, match="observer"):
        replace(plan, observers=())
    with pytest.raises(ValueError, match="reason"):
        replace(plan, reason="")


def test_legacy_point_estimate_route_is_preserved_but_unqualified() -> None:
    config = AdaptiveFIMConfig(step_gain=2.0, target_leakage=0.05)
    step = propose_next_lambda(4.0, FIMWitness(leakage=0.15, retention=0.8), config)
    schedule = adaptive_lambda_schedule(
        2.0,
        [
            FIMWitness(leakage=0.2, retention=0.9),
            FIMWitness(leakage=0.1, retention=0.95),
        ],
        AdaptiveFIMConfig(step_gain=1.0),
    )

    assert step.lambda_out == pytest.approx(3.8)
    assert step.count_qualified is False
    assert "legacy point estimate" in step.rationale
    assert [item.index for item in schedule] == [0, 1]
    assert schedule[1].lambda_in == pytest.approx(1.8)

    retention = propose_next_lambda(
        2.0,
        FIMWitness(leakage=0.1, retention=0.7),
        AdaptiveFIMConfig(mode="retention_recovery", target_retention=0.9),
    )
    assert retention.error_signal == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"lambda_min": -1.0}, "lambda_min"),
        ({"lambda_min": 2.0, "lambda_max": 1.0}, "lambda_max"),
        ({"step_gain": -1.0}, "step_gain"),
        ({"max_delta_per_batch": 0.0}, "max_delta"),
        ({"target_leakage": 1.1}, "target_leakage"),
        ({"target_retention": -0.1}, "target_retention"),
        ({"deadband": 1.1}, "deadband"),
        ({"confidence_z": 0.0}, "confidence_z"),
        ({"min_shots": 0}, "min_shots"),
        ({"mode": "bad"}, "mode"),
    ],
)
def test_config_validation(changes: dict[str, object], message: str) -> None:
    config_factory = cast(Callable[..., AdaptiveFIMConfig], AdaptiveFIMConfig)
    with pytest.raises(ValueError, match=message):
        config_factory(**changes)


def test_invalid_current_lambda_is_rejected() -> None:
    witness = _witness()
    with pytest.raises(ValueError, match="non-negative"):
        propose_count_aware_lambda(-1.0, witness)
    with pytest.raises(ValueError, match="finite"):
        adaptive_count_aware_schedule(float("nan"), (witness,))


def test_direct_invalid_step_and_refused_plan_contracts() -> None:
    witness = _witness()
    with pytest.raises(ValueError, match="hold decisions"):
        AdaptiveFIMStep(
            0,
            1.0,
            0.5,
            witness,
            0.0,
            False,
            "hold",
            False,
            None,
            "bad hold",
        )
    budget_plan = plan_adaptive_fim_schedule(
        1.0,
        (witness,),
        policy_id="ci_dry_run_only",
        shots_per_arm=4096,
    ).budget
    with pytest.raises(ValueError, match="require blockers"):
        AdaptiveFIMPlan("refused", False, "bad", (), budget_plan, (), ())


def test_observer_rejects_blank_identity() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        AdaptiveFIMObserverRecord(
            "",
            "hold",
            1.0,
            1.0,
            None,
            None,
            1024,
            "policy",
            "synthetic",
        )
