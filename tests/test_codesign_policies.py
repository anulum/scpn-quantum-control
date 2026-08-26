# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — co-design policy tests
"""Multi-angle latency and safety-envelope tests for co-design."""

from __future__ import annotations

from dataclasses import replace

import pytest

from scpn_quantum_control.codesign.contracts import (
    BackendCapabilities,
    ControllerProposal,
    GradientPlanRecord,
    ObserverInputs,
    QuantumEvaluation,
    SafetyAction,
    StaleGradientAction,
)
from scpn_quantum_control.codesign.policies import LatencyPolicy, SafetyEnvelope


def _evaluation(gradient: tuple[float, ...] = (0.1, -0.1)) -> QuantumEvaluation:
    return QuantumEvaluation(
        objective_value=0.1,
        gradient=gradient,
        order_parameter=0.8,
        evaluated_at_ms=10.0,
        backend_status="local_simulator",
        capabilities=BackendCapabilities("sim", False, False, 20.0),
        gradient_plan=GradientPlanRecord(
            "sim", "auto", "parameter_shift", True, 4, False, ("local",), "planner"
        ),
        objective_terms=("sync",),
    )


def _proposal(
    parameters: tuple[float, ...] = (0.1, -0.1),
    update: tuple[float, ...] = (0.1, -0.1),
) -> ControllerProposal:
    return ControllerProposal(parameters, update, 1.0)


def _envelope() -> SafetyEnvelope:
    return SafetyEnvelope(max_abs_parameter=1.0, max_update_norm=0.25, max_gradient_norm=1.0)


def test_latency_policy_distinguishes_current_stale_missing_and_future() -> None:
    """Classify every logical gradient-age state explicitly."""
    policy = LatencyPolicy(
        max_age_ms=5.0,
        on_stale=StaleGradientAction.HOLD,
        on_missing=StaleGradientAction.ABORT,
    )

    current = policy.assess(_evaluation(), apply_at_ms=14.0)
    stale = policy.assess(_evaluation(), apply_at_ms=16.0)
    missing = policy.assess(None, apply_at_ms=16.0)
    future = policy.assess(_evaluation(), apply_at_ms=9.0)

    assert current.action is SafetyAction.ALLOW and current.to_dict()["age_ms"] == 4.0
    assert stale.stale is True and stale.action is SafetyAction.HOLD
    assert missing.missing is True and missing.action is SafetyAction.ABORT
    assert future.action is SafetyAction.ABORT and future.age_ms == -1.0


def test_latency_policy_supports_abort_on_stale_and_hold_on_missing() -> None:
    """Honour caller-selected stale and missing actions."""
    policy = LatencyPolicy(
        max_age_ms=1.0,
        on_stale=StaleGradientAction.ABORT,
        on_missing=StaleGradientAction.HOLD,
    )

    assert policy.assess(_evaluation(), apply_at_ms=12.0).action is SafetyAction.ABORT
    assert policy.assess(None, apply_at_ms=12.0).action is SafetyAction.HOLD
    with pytest.raises(ValueError, match="max_age_ms"):
        LatencyPolicy(max_age_ms=0.0)
    with pytest.raises(ValueError, match="apply_at_ms"):
        policy.assess(_evaluation(), apply_at_ms=float("nan"))


def test_safety_envelope_allows_and_clamps_updates() -> None:
    """Allow bounded proposals and clamp update or parameter excess."""
    latency = LatencyPolicy(5.0).assess(_evaluation(), apply_at_ms=11.0)
    allowed = _envelope().decide((0.0, 0.0), _evaluation(), _proposal(), latency, ObserverInputs())
    update_clamped = _envelope().decide(
        (0.0, 0.0),
        _evaluation(),
        _proposal((0.8, -0.8), (0.8, -0.8)),
        latency,
        ObserverInputs(),
    )
    bound_clamped = _envelope().decide(
        (0.9, -0.9),
        _evaluation(),
        _proposal((1.2, -1.2), (0.1, -0.1)),
        latency,
        ObserverInputs(),
    )

    assert allowed.action is SafetyAction.ALLOW
    assert update_clamped.action is SafetyAction.CLAMP
    assert "update_norm_clamped" in update_clamped.blockers
    assert bound_clamped.action is SafetyAction.CLAMP
    assert bound_clamped.applied_parameters == (1.0, -1.0)


def test_safety_envelope_holds_or_aborts_before_application() -> None:
    """Preserve current parameters for latency and observer interlocks."""
    evaluation = _evaluation()
    hold_latency = LatencyPolicy(1.0).assess(evaluation, apply_at_ms=12.0)
    held = _envelope().decide((0.0, 0.0), evaluation, _proposal(), hold_latency, ObserverInputs())
    identity_abort = _envelope().decide(
        (0.0, 0.0),
        evaluation,
        _proposal(),
        LatencyPolicy(5.0).assess(evaluation, apply_at_ms=11.0),
        ObserverInputs(identity_action="abort", identity_reason="witness unsupported"),
    )
    identity_hold = _envelope().decide(
        (0.0, 0.0),
        evaluation,
        _proposal(),
        LatencyPolicy(5.0).assess(evaluation, apply_at_ms=11.0),
        ObserverInputs(identity_action="hold"),
    )

    assert held.action is SafetyAction.HOLD
    assert held.applied_parameters == (0.0, 0.0)
    assert identity_abort.action is SafetyAction.ABORT
    assert identity_abort.reason == "witness unsupported"
    assert identity_hold.action is SafetyAction.HOLD
    assert "requested hold" in identity_hold.reason


def test_l16_interlock_is_conservative_and_identity_has_precedence() -> None:
    """Map adjust/halt to hold/abort while retaining the identity interlock."""
    evaluation = _evaluation()
    latency = LatencyPolicy(5.0).assess(evaluation, apply_at_ms=11.0)
    adjust = _envelope().decide(
        (0.0, 0.0),
        evaluation,
        _proposal(),
        latency,
        ObserverInputs(l16_action="adjust", l16_reason="bounded heuristic"),
    )
    halt = _envelope().decide(
        (0.0, 0.0),
        evaluation,
        _proposal(),
        latency,
        ObserverInputs(l16_action="halt"),
    )
    identity = _envelope().decide(
        (0.0, 0.0),
        evaluation,
        _proposal(),
        latency,
        ObserverInputs(
            identity_action="hold",
            identity_reason="identity first",
            l16_action="halt",
            l16_reason="lower-priority abort",
        ),
    )
    continued = _envelope().decide(
        (0.0, 0.0),
        evaluation,
        _proposal(),
        latency,
        ObserverInputs(l16_action="continue", l16_reason="bounded continue"),
    )

    assert adjust.action is SafetyAction.HOLD
    assert adjust.blockers == ("l16_director",)
    assert halt.action is SafetyAction.ABORT
    assert "requested halt" in halt.reason
    assert identity.action is SafetyAction.HOLD
    assert identity.reason == "identity first"
    assert identity.blockers == ("identity_observer",)
    assert continued.action is SafetyAction.ALLOW


def test_safety_envelope_rejects_gradient_and_dimension_failures() -> None:
    """Abort excessive gradients and inconsistent dimensions."""
    latency = LatencyPolicy(5.0).assess(_evaluation(), apply_at_ms=11.0)
    too_large = _envelope().decide(
        (0.0, 0.0), _evaluation((2.0, 0.0)), _proposal(), latency, ObserverInputs()
    )
    gradient_mismatch = _envelope().decide(
        (0.0, 0.0), _evaluation((0.1,)), _proposal(), latency, ObserverInputs()
    )
    proposal_mismatch = _envelope().decide(
        (0.0, 0.0),
        _evaluation(),
        _proposal((0.1,), (0.1,)),
        latency,
        ObserverInputs(),
    )
    incomplete = _envelope().decide(
        (0.0, 0.0), None, None, replace(latency, action=SafetyAction.ALLOW), ObserverInputs()
    )

    assert too_large.action is SafetyAction.ABORT
    assert too_large.blockers == ("gradient_envelope",)
    assert gradient_mismatch.blockers == ("gradient_dimension",)
    assert proposal_mismatch.blockers == ("controller_dimension",)
    assert incomplete.blockers == ("incomplete_loop_step",)


def test_safety_envelope_validates_configuration_and_runtime_vectors() -> None:
    """Reject invalid ceilings and empty runtime vectors."""
    with pytest.raises(ValueError, match="ceilings"):
        SafetyEnvelope(0.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="current_parameters"):
        _envelope().decide(
            (),
            _evaluation(),
            _proposal(),
            LatencyPolicy(5.0).assess(_evaluation(), apply_at_ms=11.0),
            ObserverInputs(),
        )
