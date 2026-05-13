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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .backends import QuantumBackendDescriptor, describe_backend
from .feedback_loop import FeedbackCommand, FeedbackResult

ProviderSubmitter = Callable[[FeedbackCommand, Mapping[str, Any]], FeedbackResult]


@dataclass(frozen=True)
class HardwareApprovalRecord:
    """Explicit approval needed before an S1 hardware scheduler can submit."""

    approval_id: str
    approver: str
    package_hash: str
    max_qpu_seconds: float
    allowed_provider: str
    approved: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.approval_id:
            raise ValueError("approval_id must be non-empty")
        if not self.approver:
            raise ValueError("approver must be non-empty")
        if not self.package_hash:
            raise ValueError("package_hash must be non-empty")
        if self.max_qpu_seconds < 0.0:
            raise ValueError("max_qpu_seconds must be non-negative")
        if not self.allowed_provider:
            raise ValueError("allowed_provider must be non-empty")


@dataclass(frozen=True)
class HardwareSubmissionRecord:
    """Auditable record for one approved provider submission."""

    approval_id: str
    provider: str
    command_label: str
    estimated_qpu_seconds: float
    result_qpu_seconds: float
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
        self.provider = provider
        self.backend_descriptor = _resolve_backend_descriptor(provider)
        self.package_manifest = dict(package_manifest)
        self.package_hash = hash_package_manifest(package_manifest)
        self.approval = approval
        self.submitter = submitter
        self._spent_qpu_seconds = 0.0
        self._submissions: list[HardwareSubmissionRecord] = []

    @property
    def spent_qpu_seconds(self) -> float:
        """Return cumulative QPU seconds reported through this scheduler."""
        return self._spent_qpu_seconds

    @property
    def submissions(self) -> tuple[HardwareSubmissionRecord, ...]:
        """Return immutable submission records."""
        return tuple(self._submissions)

    def submit(self, command: FeedbackCommand) -> FeedbackResult:
        """Submit one command only after all approval gates pass."""
        self._check_approval(command)
        result = self.submitter(command, self.package_manifest)
        projected_spend = self._spent_qpu_seconds + result.qpu_seconds
        if projected_spend > self.approval.max_qpu_seconds:
            raise RuntimeError("provider result would exceed approved QPU budget")
        self._spent_qpu_seconds = projected_spend
        self._submissions.append(
            HardwareSubmissionRecord(
                approval_id=self.approval.approval_id,
                provider=self.provider,
                command_label=command.label,
                estimated_qpu_seconds=command.estimated_qpu_seconds,
                result_qpu_seconds=result.qpu_seconds,
                job_id=result.job_id,
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
        )
        return result

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
