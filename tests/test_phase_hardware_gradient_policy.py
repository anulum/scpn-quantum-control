# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Hardware Gradient Policy
"""Tests for hardware-gradient policy readiness evidence."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from scpn_quantum_control.phase import (
    HardwareGradientPolicy,
    HardwareGradientRequest,
    assert_hardware_gradient_policy_approved,
    evaluate_hardware_gradient_policy,
    run_hardware_gradient_policy_readiness_suite,
)

_EVIDENCE_IDS = {
    "backend_calibration_id": "cal-2026-06-05-ibm-kingston",
    "no_qpu_gate_id": "no-qpu-gate-2026-06-05",
    "claim_boundary_id": "claim-boundary-2026-06-05",
    "cost_budget_id": "qpu-budget-2026-06-05",
}


def test_hardware_gradient_policy_approves_bounded_dry_run() -> None:
    """A bounded, evidenced, allowlisted request is approved as a dry run."""
    request = HardwareGradientRequest(
        provider="ibm_quantum",
        backend="ibm_quantum",
        n_params=2,
        shots=512,
        allow_hardware=True,
        evidence_ids=_EVIDENCE_IDS,
    )

    decision = evaluate_hardware_gradient_policy(request)

    assert decision.approved
    assert not decision.fail_closed
    assert decision.mode == "dry_run"
    assert decision.requires_hardware_approval
    assert decision.evaluations == 4
    assert decision.estimated_total_shots == 2048
    assert decision.missing_evidence == ()
    assert "dry-run hardware-gradient policy" in decision.claim_boundary
    assert decision.to_dict()["evidence_ids"] == _EVIDENCE_IDS


def test_hardware_gradient_policy_blocks_without_explicit_hardware_approval() -> None:
    """A request without allow_hardware is blocked."""
    request = HardwareGradientRequest(
        provider="ibm_quantum",
        backend="ibm_quantum",
        n_params=2,
        shots=512,
        evidence_ids=_EVIDENCE_IDS,
    )

    decision = evaluate_hardware_gradient_policy(request)

    assert decision.fail_closed
    assert not decision.approved
    assert "allow_hardware=True is required" in decision.failure_reason
    assert decision.estimated_total_shots == 2048


def test_hardware_gradient_policy_blocks_unknown_provider_and_backend() -> None:
    """An unlisted provider and an unregistered backend alias are blocked."""
    request = HardwareGradientRequest(
        provider="unregistered_qpu",
        backend="mystery_backend",
        n_params=2,
        shots=512,
        allow_hardware=True,
        evidence_ids=_EVIDENCE_IDS,
    )

    decision = evaluate_hardware_gradient_policy(request)

    assert decision.fail_closed
    assert "provider 'unregistered_qpu' is not allowlisted" in decision.failure_reason
    assert (
        "backend 'mystery_backend' is not a registered hardware alias" in decision.failure_reason
    )


def test_hardware_gradient_policy_blocks_excessive_shot_budget() -> None:
    """An estimated total shot count above the policy budget is blocked."""
    policy = HardwareGradientPolicy(max_total_shots=4_000)
    request = HardwareGradientRequest(
        provider="ibm_quantum",
        backend="ibm_quantum",
        n_params=4,
        shots=1_000,
        allow_hardware=True,
        evidence_ids=_EVIDENCE_IDS,
    )

    decision = evaluate_hardware_gradient_policy(request, policy=policy)

    assert decision.fail_closed
    assert decision.evaluations == 8
    assert decision.estimated_total_shots == 8_000
    assert "estimated total shots 8000 exceed policy maximum 4000" in decision.failure_reason


def test_hardware_gradient_policy_blocks_missing_evidence_ids() -> None:
    """A request without the required evidence IDs is blocked and names them."""
    request = HardwareGradientRequest(
        provider="ibm_quantum",
        backend="ibm_quantum",
        n_params=2,
        shots=512,
        allow_hardware=True,
        evidence_ids={"backend_calibration_id": "cal-1"},
    )

    decision = evaluate_hardware_gradient_policy(request)

    assert decision.fail_closed
    assert decision.missing_evidence == (
        "claim_boundary_id",
        "cost_budget_id",
        "no_qpu_gate_id",
    )
    assert "missing required evidence IDs" in decision.failure_reason


def test_hardware_gradient_policy_live_execution_requires_ticket() -> None:
    """Live execution is blocked without a ticket and approved with one."""
    request = HardwareGradientRequest(
        provider="ibm_quantum",
        backend="ibm_quantum",
        n_params=1,
        shots=256,
        allow_hardware=True,
        evidence_ids=_EVIDENCE_IDS,
        dry_run_only=False,
    )

    blocked = evaluate_hardware_gradient_policy(request)
    approved = evaluate_hardware_gradient_policy(
        HardwareGradientRequest(
            provider=request.provider,
            backend=request.backend,
            n_params=request.n_params,
            shots=request.shots,
            allow_hardware=request.allow_hardware,
            evidence_ids=request.evidence_ids,
            dry_run_only=False,
            live_execution_ticket="QPU-LIVE-2026-06-05-001",
        )
    )

    assert blocked.fail_closed
    assert (
        "live hardware-gradient execution requires live_execution_ticket" in blocked.failure_reason
    )
    assert approved.approved
    assert approved.mode == "live_ticketed"
    assert approved.to_dict()["has_live_execution_ticket"] is True


def test_hardware_gradient_policy_readiness_suite_records_boundaries() -> None:
    """The readiness suite approves one dry run and blocks the five boundary scenarios."""
    suite = run_hardware_gradient_policy_readiness_suite()
    payload = suite.to_dict()

    assert suite.passed
    assert suite.record_count == 6
    assert suite.approved_count == 1
    assert suite.blocked_count == 5
    assert suite.live_execution_approved_count == 0
    assert payload["records"][0]["mode"] == "dry_run"
    assert json.loads(json.dumps(payload))["passed"] is True


def test_assert_hardware_gradient_policy_approved_raises_for_blocked_decision() -> None:
    """Asserting approval of a blocked decision raises with its failure reason."""
    decision = evaluate_hardware_gradient_policy(
        HardwareGradientRequest(
            provider="ibm_quantum",
            backend="ibm_quantum",
            n_params=1,
            shots=256,
        )
    )

    with pytest.raises(ValueError, match="hardware gradient policy blocked"):
        assert_hardware_gradient_policy_approved(decision)


def _bounded_request(**changes: Any) -> HardwareGradientRequest:
    """Return the approved bounded dry-run request with some fields changed."""
    request = HardwareGradientRequest(
        provider="ibm_quantum",
        backend="ibm_quantum",
        n_params=2,
        shots=512,
        allow_hardware=True,
        evidence_ids=_EVIDENCE_IDS,
    )
    return replace(request, **changes)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"n_params": 0}, "n_params must be positive"),
        ({"n_params": 40}, "n_params 40 exceed policy maximum 32"),
        ({"shift_terms": 0}, "shift_terms must be positive"),
        ({"shift_terms": 5}, "shift_terms 5 exceed policy maximum 4"),
        ({"shots": 0}, "shots must be positive for hardware-gradient policy approval"),
        ({"shots": 5000}, "shots per evaluation 5000 exceed policy maximum 4096"),
    ],
)
def test_hardware_gradient_policy_names_each_out_of_bounds_request(
    changes: dict[str, Any], reason: str
) -> None:
    """Non-positive or excessive parameters, shift terms and shots are each named as a blocker."""
    decision = evaluate_hardware_gradient_policy(_bounded_request(**changes))

    assert not decision.approved
    assert reason in decision.reasons


def test_hardware_gradient_policy_falls_back_when_the_backend_planner_cannot_plan() -> None:
    """Without a usable plan the policy still blocks and judges approval by the backend alias.

    A zero parameter count is never sent to the planner; a float count such as one read
    from JSON passes the positivity check but the planner refuses it as not an integer.
    """
    no_plan = evaluate_hardware_gradient_policy(_bounded_request(n_params=0))
    unplannable = evaluate_hardware_gradient_policy(_bounded_request(n_params=2.0))
    unknown = evaluate_hardware_gradient_policy(_bounded_request(n_params=0, backend="nowhere"))

    for decision in (no_plan, unplannable, unknown):
        assert "backend planner could not produce a hardware-gradient plan" in decision.reasons
        assert decision.backend_plan_method == "unsupported"
    assert no_plan.requires_hardware_approval is True
    assert unplannable.requires_hardware_approval is True
    assert unknown.requires_hardware_approval is False


def test_hardware_gradient_policy_carries_the_planner_reasons_for_an_unsupported_plan() -> None:
    """A hardware alias planned without hardware permission carries the planner's own reasons."""
    decision = evaluate_hardware_gradient_policy(_bounded_request(allow_hardware=False))

    assert not decision.approved
    assert len(decision.reasons) > 1
    assert decision.backend_plan_family != "unsupported"


def test_assert_hardware_gradient_policy_approved_accepts_an_approved_decision() -> None:
    """An approved decision passes the assertion without raising."""
    decision = evaluate_hardware_gradient_policy(_bounded_request())

    assert_hardware_gradient_policy_approved(decision)
    assert decision.approved
