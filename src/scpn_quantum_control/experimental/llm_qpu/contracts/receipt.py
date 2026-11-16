# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU contracts
"""Provider batch receipts: real job IDs per cell, ambiguity never read as failure."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from uuid import UUID

from .approval import OwnerApproval
from .compiled import CompiledBundle
from .wire import (
    PROVIDER_BATCH_RECEIPT_SCHEMA,
    ArtifactHeader,
    _digest,
    _strict_json,
    _text,
    canonical_bytes,
)

_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_PHASE = re.compile(r"[a-z][a-z0-9_]{1,63}\Z")
_IBM_JOB_ID = re.compile(r"[a-z0-9]{20}\Z")
_ID_SOURCES = ("submit_response", "read_only_recovery")
_STATES = ("submitted", "recovery_required")


def _utc(value: object, *, name: str) -> str:
    """Return an ISO-8601 UTC second or refuse."""
    if type(value) is not str or _UTC.fullmatch(value) is None:
        raise ValueError(f"{name} must be an ISO-8601 UTC second")
    return value


def _canonical_uuid(value: str) -> bool:
    """Return whether the text is one canonical lowercase UUID."""
    try:
        return str(UUID(value)) == value
    except ValueError:
        return False


def _check_provider_job_id(provider: str, job_id: str) -> None:
    """Refuse an ID that does not have the provider's own job ID form."""
    if provider == "iqm" and not _canonical_uuid(job_id):
        raise ValueError("an IQM job ID is a canonical UUID; a local handle is not a job ID")
    if provider == "ibm" and _IBM_JOB_ID.fullmatch(job_id) is None:
        raise ValueError("an IBM job ID is 20 lowercase alphanumerics; a local handle is not one")


@dataclass(frozen=True, slots=True)
class SubmittedJob:
    """One provider job of an attempt, or the ambiguous gap where its ID should be."""

    ordinal: int
    name: str
    payload_sha256: str
    cell_ordinals: tuple[int, ...]
    state: str
    provider_job_id: str | None
    id_source: str | None
    boundary_at_utc: str
    provider_created_at_utc: str | None
    blocker: str | None

    def __post_init__(self) -> None:
        """Refuse a submitted job without its ID and an ambiguous job posing as failed."""
        if type(self.ordinal) is not int or not 0 <= self.ordinal < 100_000:
            raise ValueError("job ordinal must be a bounded non-negative integer")
        if type(self.name) is not str or _PHASE.fullmatch(self.name) is None:
            raise ValueError("invalid provider job phase name")
        _digest(self.payload_sha256, name="submitted payload digest")
        self._check_cells()
        _utc(self.boundary_at_utc, name="submit boundary time")
        if self.state == "submitted":
            if type(self.provider_job_id) is not str or self.id_source not in _ID_SOURCES:
                raise ValueError("a submitted job needs its provider job ID and its source")
            if self.blocker is not None:
                raise ValueError("a submitted job cannot carry a blocker")
            if self.provider_created_at_utc is not None:
                _utc(self.provider_created_at_utc, name="provider creation time")
        elif self.state == "recovery_required":
            if self.provider_job_id is not None or self.id_source is not None:
                raise ValueError("an ambiguous job has no provider job ID yet")
            if self.provider_created_at_utc is not None:
                raise ValueError("an ambiguous job has no provider creation time")
            object.__setattr__(self, "blocker", _text(self.blocker, name="recovery blocker"))
        else:
            raise ValueError(
                "job state must be submitted or recovery_required; "
                "a missing job ID is ambiguous, never a failed zero-cost job"
            )

    def _check_cells(self) -> None:
        """Require a nonempty ordered tuple of distinct bundle cell indices."""
        cells = self.cell_ordinals
        if type(cells) is not tuple or not cells:
            raise ValueError("a job maps a nonempty tuple of compiled cells")
        for cell in cells:
            if type(cell) is not int or not 0 <= cell < 10_000_000:
                raise ValueError("cell ordinals must be bounded non-negative integers")
        if len(set(cells)) != len(cells):
            raise ValueError("a job cannot map one cell twice")

    def to_wire(self) -> dict[str, object]:
        """Return the exact job record."""
        return {
            "ordinal": self.ordinal,
            "name": self.name,
            "payload_sha256": self.payload_sha256,
            "cell_ordinals": list(self.cell_ordinals),
            "state": self.state,
            "provider_job_id": self.provider_job_id,
            "id_source": self.id_source,
            "boundary_at_utc": self.boundary_at_utc,
            "provider_created_at_utc": self.provider_created_at_utc,
            "blocker": self.blocker,
        }

    @classmethod
    def from_wire(cls, value: object) -> SubmittedJob:
        """Reject a job record with missing or extra fields."""
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("SubmittedJob fields mismatch")
        if type(value["cell_ordinals"]) is not list:
            raise ValueError("cell ordinals must be a list on wire")
        fields = {name: value[name] for name in cls.__dataclass_fields__}
        return cls(**{**fields, "cell_ordinals": tuple(value["cell_ordinals"])})


@dataclass(frozen=True, slots=True)
class ProviderBatchReceipt:
    """Design-only receipt binding one approved bundle to its provider jobs."""

    attempt_id: str
    compiled_bundle_digest: str
    bundle_payload_digest: str
    owner_approval_digest: str
    provider: str
    account_route: str
    device_name: str
    jobs: tuple[SubmittedJob, ...]
    header: ArtifactHeader

    def __post_init__(self) -> None:
        """Refuse HAL handles as job IDs, reused IDs and out-of-order jobs."""
        if type(self.attempt_id) is not str or not _canonical_uuid(self.attempt_id):
            raise ValueError("attempt ID must be canonical UUID text")
        for name in ("compiled_bundle_digest", "bundle_payload_digest", "owner_approval_digest"):
            _digest(getattr(self, name), name=name)
        if self.provider not in ("ibm", "iqm"):
            raise ValueError("receipt provider must be ibm or iqm")
        for name in ("account_route", "device_name"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        if type(self.jobs) is not tuple or not self.jobs:
            raise ValueError("a receipt needs a nonempty tuple of jobs")
        if any(type(job) is not SubmittedJob for job in self.jobs):
            raise ValueError("receipt jobs must be SubmittedJob records")
        if [job.ordinal for job in self.jobs] != list(range(len(self.jobs))):
            raise ValueError("receipt jobs must be ordered by ordinal from zero")
        self._check_job_ids()
        if (
            type(self.header) is not ArtifactHeader
            or self.header.object_kind != "provider_batch_receipt"
        ):
            raise ValueError("ProviderBatchReceipt requires its exact artifact header")
        if self.header.data_origin not in ("provider_metadata", "synthetic_classical"):
            raise ValueError("a receipt derives from provider responses or a fixture")
        parents = (self.compiled_bundle_digest, self.owner_approval_digest)
        self.header.validate_content(self._scientific_wire(), parents=parents)

    def _check_job_ids(self) -> None:
        """Refuse a provider ID of the wrong form, the attempt ID reused, or one ID twice."""
        seen: set[str] = set()
        for job in self.jobs:
            job_id = job.provider_job_id
            if job_id is None:
                continue
            _check_provider_job_id(self.provider, job_id)
            if job_id == self.attempt_id:
                raise ValueError("the local attempt ID is not a provider job ID")
            if job_id in seen:
                raise ValueError("one provider job ID cannot back two jobs")
            seen.add(job_id)

    def blockers(self) -> tuple[tuple[int, str], ...]:
        """Return each ambiguous job ordinal with the recovery it still needs."""
        return tuple((job.ordinal, str(job.blocker)) for job in self.jobs if job.blocker)

    def _scientific_wire(self) -> dict[str, object]:
        """Return the receipt content hashed by its header."""
        return {
            "schema": PROVIDER_BATCH_RECEIPT_SCHEMA,
            "object_kind": "provider_batch_receipt",
            "attempt_id": self.attempt_id,
            "compiled_bundle_digest": self.compiled_bundle_digest,
            "bundle_payload_digest": self.bundle_payload_digest,
            "owner_approval_digest": self.owner_approval_digest,
            "provider": self.provider,
            "account_route": self.account_route,
            "device_name": self.device_name,
            "jobs": [job.to_wire() for job in self.jobs],
        }

    def to_wire(self) -> dict[str, object]:
        """Return the detached versioned receipt."""
        return {**self._scientific_wire(), "header": self.header.to_wire()}

    @classmethod
    def from_wire(cls, value: object) -> ProviderBatchReceipt:
        """Reject unknown fields and restore the frozen receipt."""
        fields = set(cls.__dataclass_fields__) | {"schema", "object_kind"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("ProviderBatchReceipt fields mismatch")
        if (
            value["schema"] != PROVIDER_BATCH_RECEIPT_SCHEMA
            or value["object_kind"] != "provider_batch_receipt"
        ):
            raise ValueError("unknown ProviderBatchReceipt schema")
        if type(value["jobs"]) is not list:
            raise ValueError("receipt jobs must be a list on wire")
        plain = [name for name in cls.__dataclass_fields__ if name not in ("jobs", "header")]
        return cls(
            **{name: value[name] for name in plain},
            jobs=tuple(SubmittedJob.from_wire(job) for job in value["jobs"]),
            header=ArtifactHeader.from_wire(value["header"]),
        )


def validate_provider_batch_receipt(
    receipt: ProviderBatchReceipt, approval: OwnerApproval, bundle: CompiledBundle
) -> None:
    """Refuse a receipt for another bundle or approval, outside scope, or with lost cells."""
    if type(receipt) is not ProviderBatchReceipt:
        raise ValueError("receipt validation needs a ProviderBatchReceipt")
    bundle_digest = hashlib.sha256(canonical_bytes(bundle.to_wire())).hexdigest()
    if receipt.compiled_bundle_digest != bundle_digest:
        raise ValueError("receipt names a different compiled bundle")
    if receipt.bundle_payload_digest != bundle.payload_digest:
        raise ValueError("receipt names a different compiled payload")
    if (
        receipt.owner_approval_digest
        != hashlib.sha256(canonical_bytes(approval.to_wire())).hexdigest()
    ):
        raise ValueError("receipt names a different owner approval")
    if approval.compiled_bundle_digest != bundle_digest:
        raise ValueError("the approval covers a different compiled bundle")
    scope = (receipt.provider, receipt.account_route, receipt.device_name)
    if scope != (approval.provider, approval.account_route, approval.device_name):
        raise ValueError("receipt route or device differs from the approval")
    if len(receipt.jobs) > approval.max_jobs:
        raise ValueError("receipt has more provider jobs than approved")
    mapped = sorted(cell for job in receipt.jobs for cell in job.cell_ordinals)
    if mapped != list(range(len(bundle.cells))):
        raise ValueError("receipt does not map every compiled cell exactly once")
    if receipt.header.data_origin != bundle.header.data_origin:
        raise ValueError("receipt and bundle data origins differ")
    if receipt.header.data_origin == "provider_metadata" and (
        approval.signature.key_environment == "test"
    ):
        raise ValueError("a test-key approval cannot back a provider receipt")


def decode_provider_batch_receipt(raw: bytes) -> ProviderBatchReceipt:
    """Decode one provider batch receipt from strict canonical wire bytes."""
    return ProviderBatchReceipt.from_wire(_strict_json(raw))
