# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for approval-gated feedback scheduler
"""Tests for approval-gated hardware feedback scheduling."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from typing import Any, TypedDict

import pytest

from scpn_quantum_control.hardware.feedback_hardware_scheduler import (
    ApprovalGatedFeedbackHardwareScheduler,
    HardwareApprovalRecord,
    hash_package_manifest,
)
from scpn_quantum_control.hardware.feedback_loop import FeedbackCommand, FeedbackResult


class ApprovalOverrides(TypedDict, total=False):
    """Type-valid approval fields with invalid semantic boundaries."""

    approval_id: str
    approver: str
    package_hash: str
    max_qpu_seconds: float
    allowed_provider: str


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


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_approval_rejects_nonfinite_budget_before_provider(value: float) -> None:
    """An unbounded approval must fail before a provider can receive work."""
    calls = 0

    def submitter(command: FeedbackCommand, package: Mapping[str, Any]) -> FeedbackResult:
        nonlocal calls
        calls += 1
        return FeedbackResult(qpu_seconds=0.0)

    manifest = _manifest()
    with pytest.raises(ValueError, match="max_qpu_seconds"):
        scheduler = ApprovalGatedFeedbackHardwareScheduler(
            provider="ibm_runtime",
            package_manifest=manifest,
            approval=replace(_approval(manifest), max_qpu_seconds=value),
            submitter=submitter,
        )
        scheduler.submit(FeedbackCommand(payload={}, estimated_qpu_seconds=1.0))
    assert calls == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_command_and_result_reject_nonfinite_qpu_accounting(value: float) -> None:
    """Invalid estimates and provider usage cannot enter budget arithmetic."""
    with pytest.raises(ValueError, match="estimated_qpu_seconds"):
        FeedbackCommand(payload={}, estimated_qpu_seconds=value)
    with pytest.raises(ValueError, match="qpu_seconds"):
        FeedbackResult(qpu_seconds=value)


@pytest.mark.parametrize("payload", ['"false"', '"true"', "1"])
def test_approval_requires_exact_boolean(payload: str) -> None:
    """Decoded truthy values are not explicit boolean approval."""
    import json

    with pytest.raises(ValueError, match="approved"):
        replace(_approval(_manifest()), approved=json.loads(payload))


def test_approval_gated_scheduler_fails_closed_without_approval() -> None:
    """Reject a submission unless the approval explicitly permits it."""
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
    """Record provider identity and package evidence for approved work."""
    manifest = _manifest()

    def submitter(command: FeedbackCommand, package: Mapping[str, Any]) -> FeedbackResult:
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
    assert scheduler.submissions[0].metadata["backend_descriptor"] == "qiskit_ibm"
    assert scheduler.submissions[0].metadata["provider"] == "ibm_quantum"


def test_concurrent_dispatch_cannot_spend_the_same_remaining_budget() -> None:
    """Reject a second provider dispatch while the first outcome is unknown."""
    entered, finish = Event(), Event()
    calls = 0

    def submitter(command: FeedbackCommand, package: Mapping[str, Any]) -> FeedbackResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert finish.wait(3.0), "first provider callback was not released"
        return FeedbackResult(qpu_seconds=3.0)

    manifest = _manifest()
    scheduler = ApprovalGatedFeedbackHardwareScheduler(
        provider="ibm_runtime",
        package_manifest=manifest,
        approval=_approval(manifest),
        submitter=submitter,
    )
    command = FeedbackCommand(payload={}, estimated_qpu_seconds=3.0)
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(scheduler.submit, command)
        try:
            assert entered.wait(3.0), "first provider callback never entered"
            with pytest.raises(RuntimeError, match="in progress"):
                scheduler.submit(command)
        finally:
            finish.set()
            first_error = first.exception(timeout=3.0)
    assert first_error is None
    assert calls == 1 and scheduler.spent_qpu_seconds == 3.0
    assert len(scheduler.submissions) == 1


@pytest.mark.parametrize("error", [TimeoutError, KeyboardInterrupt])
def test_unknown_provider_outcome_blocks_retry(error: type[BaseException]) -> None:
    """An exception after dispatch is not evidence that no QPU usage occurred."""
    calls = 0

    def submitter(command: FeedbackCommand, package: Mapping[str, Any]) -> FeedbackResult:
        nonlocal calls
        calls += 1
        raise error("provider outcome unavailable")

    manifest = _manifest()
    scheduler = ApprovalGatedFeedbackHardwareScheduler(
        provider="ibm_runtime",
        package_manifest=manifest,
        approval=_approval(manifest),
        submitter=submitter,
    )
    command = FeedbackCommand(payload={}, estimated_qpu_seconds=1.0)
    with pytest.raises(error):
        scheduler.submit(command)
    with pytest.raises(RuntimeError, match="reconciliation"):
        scheduler.submit(command)
    assert calls == 1
    assert scheduler.requires_reconciliation
    assert scheduler.spent_qpu_seconds == 0.0
    assert len(scheduler.submissions) == 1
    assert scheduler.submissions[0].result_qpu_seconds is None
    assert scheduler.submissions[0].job_id is None
    assert scheduler.submissions[0].estimated_qpu_seconds == 1.0


def test_unknown_outcome_preserves_previously_reported_usage() -> None:
    """An uncertain second job neither erases known usage nor fabricates a refund."""
    calls = 0

    def submitter(command: FeedbackCommand, package: Mapping[str, Any]) -> FeedbackResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            return FeedbackResult(job_id="known-job", qpu_seconds=1.5)
        raise TimeoutError("unknown job")

    manifest = _manifest()
    scheduler = ApprovalGatedFeedbackHardwareScheduler(
        provider="ibm_runtime",
        package_manifest=manifest,
        approval=_approval(manifest),
        submitter=submitter,
    )
    command = FeedbackCommand(payload={}, estimated_qpu_seconds=1.0)
    scheduler.submit(command)
    with pytest.raises(TimeoutError):
        scheduler.submit(command)
    assert scheduler.spent_qpu_seconds == 1.5
    assert [row.result_qpu_seconds for row in scheduler.submissions] == [1.5, None]
    assert scheduler.submissions[0].job_id == "known-job"
    assert scheduler.requires_reconciliation


def test_invalid_provider_result_quarantines_dispatch() -> None:
    """Malformed decoded provider output cannot silently authorize a retry."""
    import json

    manifest = _manifest()
    scheduler = ApprovalGatedFeedbackHardwareScheduler(
        provider="ibm_runtime",
        package_manifest=manifest,
        approval=_approval(manifest),
        submitter=lambda command, package: json.loads("null"),
    )
    with pytest.raises(TypeError, match="FeedbackResult"):
        scheduler.submit(FeedbackCommand(payload={}))
    assert scheduler.requires_reconciliation
    assert scheduler.submissions[0].result_qpu_seconds is None


def test_reentrant_refusal_and_provider_edits_preserve_caller_state() -> None:
    """Reject reentry without deadlock; provider edits never mutate the caller."""
    manifest = _manifest()
    payload = {"gates": ["x"]}

    def submitter(command: FeedbackCommand, package: Mapping[str, Any]) -> FeedbackResult:
        command.payload["gates"].append("provider-local")
        with pytest.raises(RuntimeError, match="in progress"):
            scheduler.submit(command)
        return FeedbackResult(qpu_seconds=1.0)

    scheduler = ApprovalGatedFeedbackHardwareScheduler(
        provider="ibm_runtime",
        package_manifest=manifest,
        approval=_approval(manifest),
        submitter=submitter,
    )
    command = FeedbackCommand(payload=payload, estimated_qpu_seconds=1.0)
    scheduler.submit(command)
    scheduler.submit(command)
    assert payload == {"gates": ["x"]}
    assert scheduler.spent_qpu_seconds == 2.0
    assert not scheduler.requires_reconciliation
    record_metadata = scheduler.submissions[0].metadata
    assert isinstance(record_metadata, dict)
    record_metadata["package_hash"] = "tampered"
    assert scheduler.submissions[0].metadata["package_hash"] == scheduler.package_hash


@pytest.mark.parametrize("attribute", ["provider", "backend_descriptor", "approval", "submitter"])
def test_dispatch_configuration_cannot_be_replaced(attribute: str) -> None:
    """Configuration replacement must not bypass or retag an active approval."""
    manifest = _manifest()
    scheduler = ApprovalGatedFeedbackHardwareScheduler(
        provider="ibm_runtime",
        package_manifest=manifest,
        approval=_approval(manifest),
        submitter=lambda command, package: FeedbackResult(),
    )
    original = getattr(scheduler, attribute)
    with pytest.raises(AttributeError):
        setattr(scheduler, attribute, None)
    assert getattr(scheduler, attribute) is original


def test_scheduler_snapshots_nested_approved_manifest() -> None:
    """Caller-owned nested lists must not alter the approved provider payload."""
    circuits = ["approved"]
    manifest: dict[str, object] = {"circuits": circuits}
    approval = _approval(manifest)
    received: list[str] = []

    def submitter(command: FeedbackCommand, package: Mapping[str, Any]) -> FeedbackResult:
        received.append(hash_package_manifest(package))
        return FeedbackResult(qpu_seconds=0.0)

    scheduler = ApprovalGatedFeedbackHardwareScheduler(
        provider="ibm_runtime",
        package_manifest=manifest,
        approval=approval,
        submitter=submitter,
    )
    circuits.append("unapproved")
    scheduler.submit(FeedbackCommand(payload={}))
    assert received == [approval.package_hash]
    assert scheduler.package_hash == approval.package_hash


def test_provider_cannot_mutate_future_submission_manifest() -> None:
    """Each dispatch receives an isolated copy of the approved package."""
    manifest: dict[str, object] = {"circuits": ["approved"]}
    approval = _approval(manifest)
    received: list[str] = []

    def submitter(command: FeedbackCommand, package: Mapping[str, Any]) -> FeedbackResult:
        received.append(hash_package_manifest(package))
        package["circuits"].append("provider-local")
        return FeedbackResult(qpu_seconds=0.0)

    scheduler = ApprovalGatedFeedbackHardwareScheduler(
        provider="ibm_runtime",
        package_manifest=manifest,
        approval=approval,
        submitter=submitter,
    )
    scheduler.submit(FeedbackCommand(payload={}))
    scheduler.submit(FeedbackCommand(payload={}))
    assert received == [approval.package_hash, approval.package_hash]
    assert hash_package_manifest(scheduler.package_manifest) == approval.package_hash
    assert hash_package_manifest(manifest) == approval.package_hash


def test_manifest_inspection_cannot_change_dispatch_payload() -> None:
    """A caller inspecting a manifest cannot edit the stored approval snapshot."""
    manifest: dict[str, object] = {"circuits": ["approved"]}
    approval = _approval(manifest)
    scheduler = ApprovalGatedFeedbackHardwareScheduler(
        provider="ibm_runtime",
        package_manifest=manifest,
        approval=approval,
        submitter=lambda command, package: FeedbackResult(qpu_seconds=0.0),
    )
    scheduler.package_manifest["circuits"].append("inspection-edit")
    assert hash_package_manifest(scheduler.package_manifest) == approval.package_hash


def test_approval_gated_scheduler_rejects_provider_and_hash_mismatch() -> None:
    """Reject provider and package evidence that diverges from approval."""
    manifest = _manifest()
    approval = _approval(manifest)
    provider_calls: list[FeedbackCommand] = []

    def record_provider_call(
        command: FeedbackCommand, package: Mapping[str, Any]
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


def test_approval_gated_scheduler_rejects_non_submit_descriptor() -> None:
    """Reject descriptors that expose simulation without live submission."""
    manifest = _manifest()
    approval = HardwareApprovalRecord(
        approval_id="local-sim",
        approver="Miroslav Sotek",
        package_hash=hash_package_manifest(manifest),
        max_qpu_seconds=4.0,
        allowed_provider="qiskit_aer",
        approved=True,
    )
    scheduler = ApprovalGatedFeedbackHardwareScheduler(
        provider="qiskit_aer",
        package_manifest=manifest,
        approval=approval,
        submitter=lambda command, package: FeedbackResult(qpu_seconds=0.0),
    )

    with pytest.raises(PermissionError, match="does not expose live submission"):
        scheduler.submit(FeedbackCommand(payload={}, estimated_qpu_seconds=1.0))


def test_approval_gated_scheduler_accepts_descriptor_name_or_provider() -> None:
    """Accept either registered descriptor name or provider identity."""
    manifest = _manifest()
    descriptor_name_approval = HardwareApprovalRecord(
        approval_id="approved-by-descriptor",
        approver="Miroslav Sotek",
        package_hash=hash_package_manifest(manifest),
        max_qpu_seconds=4.0,
        allowed_provider="qiskit_ibm",
        approved=True,
    )
    provider_approval = HardwareApprovalRecord(
        approval_id="approved-by-provider",
        approver="Miroslav Sotek",
        package_hash=hash_package_manifest(manifest),
        max_qpu_seconds=4.0,
        allowed_provider="ibm_quantum",
        approved=True,
    )

    for approval in (descriptor_name_approval, provider_approval):
        scheduler = ApprovalGatedFeedbackHardwareScheduler(
            provider="qiskit_ibm",
            package_manifest=manifest,
            approval=approval,
            submitter=lambda command, package: FeedbackResult(job_id="job-1", qpu_seconds=0.5),
        )
        result = scheduler.submit(FeedbackCommand(payload={}, estimated_qpu_seconds=0.5))
        assert result.job_id == "job-1"


def test_approval_gated_scheduler_enforces_estimated_and_reported_qpu_budget() -> None:
    """Enforce the approved QPU budget before and after provider execution."""
    manifest = _manifest()
    provider_calls = 0

    def costly_submitter(command: FeedbackCommand, package: Mapping[str, Any]) -> FeedbackResult:
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
    assert not scheduler.requires_reconciliation
    assert scheduler.submissions == ()
    assert scheduler.spent_qpu_seconds == 0.0
    with pytest.raises(RuntimeError, match="provider result would exceed"):
        scheduler.submit(FeedbackCommand(payload={}, estimated_qpu_seconds=1.0))
    assert provider_calls == 1
    assert scheduler.spent_qpu_seconds == 5.0
    assert len(scheduler.submissions) == 1
    assert scheduler.submissions[0].result_qpu_seconds == 5.0
    assert scheduler.submissions[0].estimated_qpu_seconds == 1.0


@pytest.mark.parametrize("estimate", [0.0, 1.0])
def test_overbudget_result_blocks_repeat_provider_dispatch(estimate: float) -> None:
    """Actual overspend must block even a zero-estimate follow-up."""
    manifest = _manifest()
    calls = 0

    def submitter(command: FeedbackCommand, package: Mapping[str, Any]) -> FeedbackResult:
        nonlocal calls
        calls += 1
        return FeedbackResult(job_id="overspent-job", qpu_seconds=5.0)

    scheduler = ApprovalGatedFeedbackHardwareScheduler(
        provider="ibm_runtime",
        package_manifest=manifest,
        approval=_approval(manifest),
        submitter=submitter,
    )
    with pytest.raises(RuntimeError, match="provider result would exceed"):
        scheduler.submit(FeedbackCommand(payload={}, estimated_qpu_seconds=1.0))
    with pytest.raises(RuntimeError, match="command would exceed"):
        scheduler.submit(FeedbackCommand(payload={}, estimated_qpu_seconds=estimate))
    assert calls == 1
    assert scheduler.spent_qpu_seconds == 5.0
    assert len(scheduler.submissions) == 1
    assert scheduler.submissions[0].job_id == "overspent-job"


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
    kwargs: ApprovalOverrides,
    message: str,
) -> None:
    """Reject empty identifiers and negative approval budgets."""
    params: ApprovalOverrides = {
        "approval_id": "approval",
        "approver": "Miroslav Sotek",
        "package_hash": "hash",
        "max_qpu_seconds": 1.0,
        "allowed_provider": "ibm_runtime",
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=message):
        HardwareApprovalRecord(**params)


def test_approval_gated_scheduler_rejects_empty_provider_and_manifest() -> None:
    """Reject schedulers without a provider or preregistered manifest."""
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
