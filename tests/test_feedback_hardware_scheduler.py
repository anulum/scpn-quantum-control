# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for approval-gated feedback scheduler
"""Tests for approval-gated hardware feedback scheduling."""

from __future__ import annotations

import pytest

from scpn_quantum_control.hardware.feedback_hardware_scheduler import (
    ApprovalGatedFeedbackHardwareScheduler,
    HardwareApprovalRecord,
    hash_package_manifest,
)
from scpn_quantum_control.hardware.feedback_loop import FeedbackCommand, FeedbackResult


def _manifest() -> dict[str, object]:
    return {
        "experiment_id": "s1",
        "budget": {"total_reserved_seconds": 4.0},
        "dossier": {"claim_boundary": "no broad claim"},
    }


def _approval(manifest: dict[str, object], *, approved: bool = True) -> HardwareApprovalRecord:
    return HardwareApprovalRecord(
        approval_id="approved-s1",
        approver="Miroslav Sotek",
        package_hash=hash_package_manifest(manifest),
        max_qpu_seconds=4.0,
        allowed_provider="ibm_runtime",
        approved=approved,
        notes="explicit S1 approval",
    )


def test_approval_gated_scheduler_fails_closed_without_approval() -> None:
    manifest = _manifest()
    scheduler = ApprovalGatedFeedbackHardwareScheduler(
        provider="ibm_runtime",
        package_manifest=manifest,
        approval=_approval(manifest, approved=False),
        submitter=lambda command, package: FeedbackResult(qpu_seconds=1.0),
    )

    with pytest.raises(PermissionError, match="approved=True"):
        scheduler.submit(FeedbackCommand(payload={}, estimated_qpu_seconds=1.0))


def test_approval_gated_scheduler_records_approved_submission() -> None:
    manifest = _manifest()

    def submitter(command: FeedbackCommand, package: dict[str, object]) -> FeedbackResult:
        assert package["experiment_id"] == "s1"
        return FeedbackResult(job_id="job-1", qpu_seconds=1.5, metadata={"ok": True})

    scheduler = ApprovalGatedFeedbackHardwareScheduler(
        provider="ibm_runtime",
        package_manifest=manifest,
        approval=_approval(manifest),
        submitter=submitter,
    )

    result = scheduler.submit(
        FeedbackCommand(payload={"arm": "feedback"}, label="feedback", estimated_qpu_seconds=1.0)
    )

    assert result.job_id == "job-1"
    assert scheduler.spent_qpu_seconds == 1.5
    assert scheduler.submissions[0].approval_id == "approved-s1"
    assert scheduler.submissions[0].metadata["package_hash"] == hash_package_manifest(manifest)


def test_approval_gated_scheduler_rejects_provider_and_hash_mismatch() -> None:
    manifest = _manifest()
    approval = _approval(manifest)
    provider_calls: list[FeedbackCommand] = []

    def record_provider_call(
        command: FeedbackCommand, package: dict[str, object]
    ) -> FeedbackResult:
        provider_calls.append(command)
        return FeedbackResult(qpu_seconds=0.0)

    provider_mismatch = ApprovalGatedFeedbackHardwareScheduler(
        provider="openqasm3_gate",
        package_manifest=manifest,
        approval=approval,
        submitter=record_provider_call,
    )
    with pytest.raises(PermissionError, match="provider"):
        provider_mismatch.submit(FeedbackCommand(payload={}))
    assert provider_calls == []

    stale_approval = HardwareApprovalRecord(
        approval_id="stale",
        approver="Miroslav Sotek",
        package_hash="stale-hash",
        max_qpu_seconds=4.0,
        allowed_provider="ibm_runtime",
        approved=True,
    )
    hash_mismatch = ApprovalGatedFeedbackHardwareScheduler(
        provider="ibm_runtime",
        package_manifest=manifest,
        approval=stale_approval,
        submitter=record_provider_call,
    )
    with pytest.raises(PermissionError, match="package hash"):
        hash_mismatch.submit(FeedbackCommand(payload={}))
    assert provider_calls == []


def test_approval_gated_scheduler_enforces_estimated_and_reported_qpu_budget() -> None:
    manifest = _manifest()
    provider_calls = 0

    def costly_submitter(command: FeedbackCommand, package: dict[str, object]) -> FeedbackResult:
        nonlocal provider_calls
        provider_calls += 1
        return FeedbackResult(qpu_seconds=5.0)

    scheduler = ApprovalGatedFeedbackHardwareScheduler(
        provider="ibm_runtime",
        package_manifest=manifest,
        approval=_approval(manifest),
        submitter=costly_submitter,
    )

    with pytest.raises(RuntimeError, match="command would exceed"):
        scheduler.submit(FeedbackCommand(payload={}, estimated_qpu_seconds=5.0))
    assert provider_calls == 0
    with pytest.raises(RuntimeError, match="provider result would exceed"):
        scheduler.submit(FeedbackCommand(payload={}, estimated_qpu_seconds=1.0))
    assert provider_calls == 1
    assert scheduler.submissions == ()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"approval_id": ""}, "approval_id"),
        ({"approver": ""}, "approver"),
        ({"package_hash": ""}, "package_hash"),
        ({"max_qpu_seconds": -0.1}, "max_qpu_seconds"),
        ({"allowed_provider": ""}, "allowed_provider"),
    ),
)
def test_hardware_approval_record_rejects_invalid_boundaries(
    kwargs: dict[str, object],
    message: str,
) -> None:
    params = {
        "approval_id": "approval",
        "approver": "Miroslav Sotek",
        "package_hash": "hash",
        "max_qpu_seconds": 1.0,
        "allowed_provider": "ibm_runtime",
    } | kwargs

    with pytest.raises(ValueError, match=message):
        HardwareApprovalRecord(**params)


def test_approval_gated_scheduler_rejects_empty_provider_and_manifest() -> None:
    manifest = _manifest()

    with pytest.raises(ValueError, match="provider"):
        ApprovalGatedFeedbackHardwareScheduler(
            provider="",
            package_manifest=manifest,
            approval=_approval(manifest),
            submitter=lambda command, package: FeedbackResult(qpu_seconds=0.0),
        )
    with pytest.raises(ValueError, match="package_manifest"):
        ApprovalGatedFeedbackHardwareScheduler(
            provider="ibm_runtime",
            package_manifest={},
            approval=_approval(manifest),
            submitter=lambda command, package: FeedbackResult(qpu_seconds=0.0),
        )
