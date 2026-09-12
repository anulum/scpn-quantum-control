# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Approval-gated feedback hardware scheduler
"""Approval-gated hardware scheduler boundary for S1 feedback jobs.

The scheduler in this module is a safety wrapper. It never discovers
credentials, creates provider sessions, or submits jobs on its own. A caller must
inject a provider submitter and an explicit approval record that matches the
preregistered package and QPU budget before any submission can pass.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from .backends import QuantumBackendDescriptor, describe_backend
from .feedback_loop import FeedbackCommand, FeedbackResult

ProviderSubmitter = Callable[[FeedbackCommand, Mapping[str, Any]], FeedbackResult]


@dataclass(frozen=True)
class HardwareApprovalRecord:
    """Explicit boolean approval and finite QPU-second limit for submission.

    Non-finite or negative limits and non-boolean approval values raise
    ``ValueError`` during construction, before provider dispatch is possible.
    A valid record with ``approved=False`` still refuses submission.
    """

    approval_id: str
    approver: str
    package_hash: str
    max_qpu_seconds: float
    allowed_provider: str
    approved: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        """Reject approval records that cannot identify or bound a submission."""
        if not self.approval_id:
            raise ValueError("approval_id must be non-empty")
        if not self.approver:
            raise ValueError("approver must be non-empty")
        if not self.package_hash:
            raise ValueError("package_hash must be non-empty")
        if not math.isfinite(self.max_qpu_seconds) or self.max_qpu_seconds < 0.0:
            raise ValueError("max_qpu_seconds must be finite and non-negative")
        if type(self.approved) is not bool:
            raise ValueError("approved must be a boolean")
        if not self.allowed_provider:
            raise ValueError("allowed_provider must be non-empty")


@dataclass(frozen=True)
class HardwareSubmissionRecord:
    """One dispatch attempt; None usage means the provider outcome is unknown."""

    approval_id: str
    provider: str
    command_label: str
    estimated_qpu_seconds: float
    result_qpu_seconds: float | None
    job_id: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ApprovalGatedFeedbackHardwareScheduler:
    """FeedbackScheduler-compatible wrapper for approved hardware submitters."""

    is_hardware = True

    def __init__(
        self,
        *,
        provider: str,
        package_manifest: Mapping[str, Any],
        approval: HardwareApprovalRecord,
        submitter: ProviderSubmitter,
    ) -> None:
        if not provider:
            raise ValueError("provider must be non-empty")
        if not package_manifest:
            raise ValueError("package_manifest must be non-empty")
        self._provider = provider
        self._backend_descriptor = _resolve_backend_descriptor(provider)
        self._package_manifest = deepcopy(dict(package_manifest))
        self._package_hash = hash_package_manifest(self._package_manifest)
        self._approval = approval
        self._submitter = submitter
        self._spent_qpu_seconds = 0.0
        self._submissions: list[HardwareSubmissionRecord] = []
        self._submission_lock = Lock()
        self._requires_reconciliation = False

    @property
    def provider(self) -> str:
        """Construction-time provider identity; cannot be replaced during dispatch."""
        return self._provider

    @property
    def backend_descriptor(self) -> QuantumBackendDescriptor | None:
        """Construction-time capability descriptor, or None for an unknown alias."""
        return self._backend_descriptor

    @property
    def approval(self) -> HardwareApprovalRecord:
        """Fixed approval governing this instance's cumulative usage."""
        return self._approval

    @property
    def submitter(self) -> ProviderSubmitter:
        """Fixed provider callback; replacing it requires a new scheduler."""
        return self._submitter

    @property
    def package_manifest(self) -> dict[str, Any]:
        """Detached copy of the construction-time approved package.

        Neither inspection nor provider-local edits change future dispatches.
        To change a package, construct a new scheduler with matching approval.
        """
        return deepcopy(self._package_manifest)

    @property
    def package_hash(self) -> str:
        """SHA256 of the stored package snapshot, fixed at construction."""
        return self._package_hash

    @property
    def spent_qpu_seconds(self) -> float:
        """Known reported usage, excluding unresolved attempts; never a refund."""
        return self._spent_qpu_seconds

    @property
    def requires_reconciliation(self) -> bool:
        """Whether an uncertain provider outcome permanently blocks this instance.

        Provider exceptions, including cancellation, do not prove zero usage.
        Reconcile external job/usage records before obtaining a fresh approval;
        this in-memory wrapper has no automatic retry or reset mechanism.

        """
        return self._requires_reconciliation

    @property
    def submissions(self) -> tuple[HardwareSubmissionRecord, ...]:
        """Detached attempt records, including unknown provider outcomes."""
        return deepcopy(tuple(self._submissions))

    def submit(self, command: FeedbackCommand) -> FeedbackResult:
        """Dispatch one isolated command under the fixed approval and usage limit.

        Parameters
        ----------
        command : FeedbackCommand
            Command with an estimated QPU cost in seconds. The provider receives
            deep copies of the command and approved manifest.

        Returns
        -------
        FeedbackResult
            Provider result, with reported QPU usage charged to this instance.

        Raises
        ------
        PermissionError
            Approval, provider capability or package identity does not match.
        RuntimeError
            Another call is active, an outcome needs reconciliation, or the
            estimated or reported cumulative usage exceeds the approval.
        TypeError
            The provider does not return a FeedbackResult.

        Notes
        -----
        Provider exceptions (including interruption) propagate after recording
        unknown usage as None and permanently blocking further dispatch on this
        instance. A reported overrun retains its actual usage and job record.
        Pre-dispatch refusals do not consume budget or quarantine the instance.
        This is in-memory accounting, not a durable, cross-process billing ledger:
        reconcile uncertain jobs and obtain a remaining-budget approval before
        constructing a replacement. It cannot undo provider-side consumption.

        """
        if not self._submission_lock.acquire(blocking=False):
            raise RuntimeError("provider submission already in progress")
        try:
            if self.requires_reconciliation:
                raise RuntimeError("provider outcome requires reconciliation before further work")
            self._check_approval(command)
            submitted_command = deepcopy(command)
            manifest = self.package_manifest
            try:
                result = self.submitter(submitted_command, manifest)
                if not isinstance(result, FeedbackResult):
                    raise TypeError("submitter must return FeedbackResult")
            except BaseException:
                self._requires_reconciliation = True
                self._submissions.append(self._submission_record(command, None))
                raise
            projected_spend = self._spent_qpu_seconds + result.qpu_seconds
            self._spent_qpu_seconds = projected_spend
            self._submissions.append(self._submission_record(command, result))
            if projected_spend > self.approval.max_qpu_seconds:
                raise RuntimeError("provider result would exceed approved QPU budget")
            return result
        finally:
            self._submission_lock.release()

    def _submission_record(
        self, command: FeedbackCommand, result: FeedbackResult | None
    ) -> HardwareSubmissionRecord:
        return HardwareSubmissionRecord(
            approval_id=self.approval.approval_id,
            provider=self.provider,
            command_label=command.label,
            estimated_qpu_seconds=command.estimated_qpu_seconds,
            result_qpu_seconds=result.qpu_seconds if result is not None else None,
            job_id=result.job_id if result is not None else None,
            metadata={
                "package_hash": self.package_hash,
                "approval_notes": self.approval.notes,
                "backend_descriptor": (
                    self.backend_descriptor.name if self.backend_descriptor else ""
                ),
                "provider": self.backend_descriptor.provider
                if self.backend_descriptor
                else self.provider,
            },
        )

    def _check_approval(self, command: FeedbackCommand) -> None:
        if not self.approval.approved:
            raise PermissionError("hardware scheduler requires approved=True")
        allowed_provider_names = {self.provider}
        if self.backend_descriptor is not None:
            allowed_provider_names.update(
                {self.backend_descriptor.name, self.backend_descriptor.provider}
            )
            if not self.backend_descriptor.can_submit:
                raise PermissionError(
                    f"backend descriptor {self.backend_descriptor.name!r} "
                    "does not expose live submission"
                )
        if self.approval.allowed_provider not in allowed_provider_names:
            raise PermissionError("approval provider does not match scheduler provider")
        if self.approval.package_hash != self.package_hash:
            raise PermissionError("approval package hash does not match manifest")
        projected_estimate = self._spent_qpu_seconds + command.estimated_qpu_seconds
        if projected_estimate > self.approval.max_qpu_seconds:
            raise RuntimeError("command would exceed approved QPU budget")


def hash_package_manifest(package_manifest: Mapping[str, Any]) -> str:
    """Return a stable SHA256 hash for a preregistered package manifest."""
    encoded = json.dumps(package_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_backend_descriptor(provider: str) -> QuantumBackendDescriptor | None:
    """Return a backend descriptor for known provider aliases."""
    aliases = {
        "ibm_runtime": "qiskit_ibm",
        "ibm_quantum": "qiskit_ibm",
        "iqm_resonance": "iqm",
        "local_qiskit_aer": "qiskit_aer",
    }
    descriptor_name = aliases.get(provider, provider)
    try:
        return describe_backend(descriptor_name)
    except KeyError:
        return None
