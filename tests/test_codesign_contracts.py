# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — co-design contract tests
"""Validation and serialisation tests for public co-design contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import cast

import pytest

from scpn_quantum_control.codesign.contracts import (
    BackendCapabilities,
    CoDesignMode,
    ControllerProposal,
    GradientPlanRecord,
    LatencyDecision,
    LoopStepInput,
    LoopStepOutput,
    ObserverInputs,
    PlasmaObjectiveTemplate,
    QuantumEvaluation,
    SafetyAction,
    SafetyDecision,
    StateEstimate,
    plasma_objective_templates,
)


def _evaluation() -> QuantumEvaluation:
    return QuantumEvaluation(
        objective_value=0.2,
        gradient=(0.1, -0.1),
        order_parameter=0.8,
        evaluated_at_ms=1.0,
        backend_status="local_simulator",
        capabilities=BackendCapabilities("statevector", False, False, 10.0),
        gradient_plan=GradientPlanRecord(
            backend="statevector",
            requested_method="auto",
            selected_method="parameter_shift",
            supported=True,
            evaluations=4,
            requires_hardware_approval=False,
            reasons=("local",),
            claim_boundary="planner only",
        ),
        objective_terms=("sync",),
    )


def test_contract_round_trip_mappings_are_json_ready() -> None:
    """Serialise a complete immutable loop step without custom encoders."""
    step_input = LoopStepInput(
        step=0,
        observed_at_ms=0.0,
        apply_at_ms=2.0,
        parameters=(0.0, 0.5),
        measurement=0.6,
        target_order_parameter=0.9,
        mode=CoDesignMode.HYBRID_REPLAY,
    )
    estimate = StateEstimate(0.6, 0.1, 1)
    evaluation = _evaluation()
    proposal = ControllerProposal((0.1, 0.4), (0.1, -0.1), 1.2)
    latency = LatencyDecision(False, False, 1.0, SafetyAction.ALLOW, "current")
    safety = SafetyDecision(SafetyAction.ALLOW, "within envelope", proposal.parameters)
    output = LoopStepOutput(
        step=0,
        estimate=estimate,
        evaluation=evaluation,
        proposal=proposal,
        latency=latency,
        safety=safety,
        observers=ObserverInputs(active_sensing_id="candidate-1"),
        mode=step_input.mode,
    )

    assert step_input.to_dict()["mode"] == "hybrid_replay"
    assert estimate.to_dict()["sample_count"] == 1
    assert evaluation.to_dict()["gradient_source"] == "exact_phase_objective"
    assert proposal.to_dict()["gain_scale"] == 1.2
    assert latency.to_dict()["action"] == "allow"
    assert safety.to_dict()["applied_parameters"] == [0.1, 0.4]
    assert output.to_dict()["evaluation"] == evaluation.to_dict()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"step": -1}, "step"),
        ({"step": True}, "step"),
        ({"observed_at_ms": float("nan")}, "observed_at_ms"),
        ({"apply_at_ms": -1.0}, "precede"),
        ({"parameters": ()}, "parameters"),
        ({"parameters": (float("inf"),)}, "parameters"),
        ({"measurement": -0.1}, "measurement"),
        ({"target_order_parameter": 1.1}, "target_order_parameter"),
    ],
)
def test_loop_step_input_rejects_invalid_values(kwargs: dict[str, object], message: str) -> None:
    """Reject malformed logical time, phase, and measurement inputs."""
    values: dict[str, object] = {
        "step": 0,
        "observed_at_ms": 0.0,
        "apply_at_ms": 1.0,
        "parameters": (0.0,),
        "measurement": 0.5,
        "target_order_parameter": 0.8,
    }
    values.update(kwargs)
    unchecked_constructor = cast(Callable[..., LoopStepInput], LoopStepInput)
    with pytest.raises(ValueError, match=message):
        unchecked_constructor(**values)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: BackendCapabilities("", False, False, 1.0), "backend_id"),
        (lambda: BackendCapabilities("sim", False, False, 0.0), "max_latency"),
        (lambda: StateEstimate(1.1, 0.0, 1), "order_parameter"),
        (lambda: StateEstimate(0.5, float("nan"), 1), "innovation"),
        (lambda: StateEstimate(0.5, 0.0, 0), "sample_count"),
        (lambda: ControllerProposal((), (), 1.0), "parameters"),
        (lambda: ControllerProposal((0.0,), (0.0, 1.0), 1.0), "dimensions"),
        (lambda: ControllerProposal((0.0,), (0.0,), 0.0), "gain_scale"),
        (
            lambda: SafetyDecision(SafetyAction.HOLD, "", (0.0,)),
            "reason",
        ),
        (
            lambda: ObserverInputs(identity_action="clamp"),
            "identity_action",
        ),
        (
            lambda: ObserverInputs(l16_action="clamp"),
            "l16_action",
        ),
        (
            lambda: ObserverInputs(l16_reason="orphaned"),
            "requires an l16_action",
        ),
        (
            lambda: ObserverInputs(l16_action="continue", l16_reason=" "),
            "l16_reason",
        ),
        (
            lambda: ObserverInputs(geometry_gradient_norm=-1.0),
            "geometry_gradient_norm",
        ),
    ],
)
def test_contracts_reject_invalid_state(factory: Callable[[], object], message: str) -> None:
    """Reject invalid values across the public record constructors."""
    with pytest.raises(ValueError, match=message):
        factory()


def test_evaluation_validation_and_optional_serialisation() -> None:
    """Validate evaluation fields and optional open-system metadata."""
    evaluation = replace(_evaluation(), open_system_backend="lindblad_density")

    assert evaluation.to_dict()["open_system_backend"] == "lindblad_density"
    with pytest.raises(ValueError, match="objective_value"):
        replace(evaluation, objective_value=float("nan"))
    with pytest.raises(ValueError, match="gradient"):
        replace(evaluation, gradient=())
    with pytest.raises(ValueError, match="order_parameter"):
        replace(evaluation, order_parameter=2.0)
    with pytest.raises(ValueError, match="evaluated_at_ms"):
        replace(evaluation, evaluated_at_ms=float("inf"))
    with pytest.raises(ValueError, match="backend_status"):
        replace(evaluation, backend_status="")
    with pytest.raises(ValueError, match="objective_terms"):
        replace(evaluation, objective_terms=())


def test_l16_observer_serialisation_preserves_legacy_payloads() -> None:
    """Expose bounded-director fields only when the L16 observer is present."""
    legacy = ObserverInputs(active_sensing_id="candidate-1").to_dict()
    l16 = ObserverInputs(
        geometry_gradient_norm=0.2,
        l16_action="adjust",
        l16_reason="conservative hold",
    ).to_dict()

    assert "l16_action" not in legacy
    assert "l16_reason" not in legacy
    assert l16["l16_action"] == "adjust"
    assert l16["l16_reason"] == "conservative hold"
    assert l16["geometry_gradient_norm"] == 0.2


def test_plasma_templates_are_permanently_non_operational() -> None:
    """Keep plasma-relevance templates partner-gated and non-operational."""
    rows = plasma_objective_templates()

    assert {row.template_id for row in rows} == {
        "phase_coherence_recovery",
        "observer_staleness_hold",
    }
    assert all(row.non_operational and row.partner_validation_required for row in rows)
    with pytest.raises(ValueError, match="fields"):
        replace(rows[0], objective="")
    with pytest.raises(ValueError, match="non-operational"):
        PlasmaObjectiveTemplate("x", "estimate", "proxy", "objective", non_operational=False)
