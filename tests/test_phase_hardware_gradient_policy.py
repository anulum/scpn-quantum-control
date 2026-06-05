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
