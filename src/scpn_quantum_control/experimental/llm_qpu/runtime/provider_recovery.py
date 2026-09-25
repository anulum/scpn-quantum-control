# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — read-only IQM job reconciliation
"""Bind an ambiguous IQM job only after retrieving its exact provider payload."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from importlib import import_module
from typing import Any, Protocol, cast
from uuid import UUID

from .attempt_journal import AttemptJournal, JournalStateError


class ProviderJob(Protocol):
    """IQM job's read-only payload surface."""

    def payload(self) -> tuple[list[object], object]:
        """Return circuits and all execution parameters."""


class ProviderClient(Protocol):
    """IQM client's read-only job lookup surface."""

    def get_job(self, job_id: UUID) -> ProviderJob:
        """Read an existing provider job by UUID."""


def canonical_request_sha256(circuits: list[object], parameters: object) -> str:
    """Reconstruct the SDK request and hash the W10 canonical JSON envelope."""
    if not is_dataclass(parameters) or isinstance(parameters, type):
        raise ValueError("provider job parameters are not an IQM dataclass")
    fields = asdict(parameters)
    if "circuits" in fields:
        raise ValueError("provider parameters unexpectedly contain circuits")
    model_type = cast(Any, import_module("iqm.station_control.interface.models")).RunRequest
    request = model_type.model_validate({**fields, "circuits": circuits})
    payload = request.model_dump(mode="json")
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def recover_existing_job(
    journal: AttemptJournal,
    attempt_id: str,
    ordinal: int,
    provider_job_id: str,
    client: ProviderClient,
) -> str:
    """Read an owner-identified IQM job and bind it only on an exact request match.

    This function never submits or retries. An unknown job ID or payload mismatch
    leaves the journal's ambiguous state unchanged.
    """
    snapshot = journal.snapshot(attempt_id)
    rows = snapshot["jobs"]
    if not isinstance(rows, list) or type(ordinal) is not int or not 0 <= ordinal < len(rows):
        raise ValueError("unknown attempt job ordinal")
    row = rows[ordinal]
    if not isinstance(row, dict) or row["state"] not in {"submitting", "recovery_required"}:
        raise JournalStateError("recovery needs an uncertain submit boundary")
    job = client.get_job(UUID(provider_job_id))
    circuits, parameters = job.payload()
    observed = canonical_request_sha256(circuits, parameters)
    journal.record_recovered(attempt_id, ordinal, provider_job_id, observed)
    return observed
