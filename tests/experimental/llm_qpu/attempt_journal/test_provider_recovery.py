# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — existing IQM job recovery acceptance
"""Exercise real IQM request models and SQLite recovery without paid submission."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from scpn_quantum_control.experimental.llm_qpu.runtime import (
    AttemptJournal,
    JobPlan,
    JournalStateError,
    recover_existing_job,
)
from scpn_quantum_control.experimental.llm_qpu.runtime.provider_recovery import (
    canonical_request_sha256,
)

iqm_client = pytest.importorskip("iqm.iqm_client")
iqm_models = pytest.importorskip("iqm.station_control.interface.models")


class _ExistingJob:
    """Read-only job payload surface for an offline paid-provider boundary."""

    def __init__(self, circuits: list[object], parameters: object) -> None:
        self.circuits = circuits
        self.parameters = parameters

    def payload(self) -> tuple[list[object], object]:
        return self.circuits, self.parameters


class _ExistingJobDirectory:
    """Lookup of an already-created job; no submit operation exists here."""

    def __init__(self, job_id: UUID, job: _ExistingJob) -> None:
        self.job_id = job_id
        self.job = job
        self.lookups = 0

    def get_job(self, job_id: UUID) -> _ExistingJob:
        self.lookups += 1
        if job_id != self.job_id:
            raise LookupError("unknown provider job")
        return self.job


def _request() -> tuple[object, list[object], object]:
    payload = {
        "circuits": [
            {
                "name": "circuit-0",
                "instructions": [
                    {
                        "name": "measure",
                        "locus": ["QB1"],
                        "args": {"key": "c_4_0_0"},
                        "implementation": None,
                    }
                ],
                "metadata": {},
            }
        ],
        "calibration_set_id": str(uuid4()),
        "shots": 1024,
    }
    request = iqm_models.RunRequest.model_validate(payload)
    normalized = request.model_dump(mode="python")
    circuits = normalized.pop("circuits")
    parameters = iqm_client.CircuitJobParameters(**normalized)
    assert dataclasses.is_dataclass(parameters)
    return request, circuits, parameters


def test_existing_provider_payload_binds_ambiguous_job(tmp_path: Path) -> None:
    """Cold SQLite recovery uses the real IQM request schema and no submit call."""
    request, circuits, parameters = _request()
    digest = canonical_request_sha256(circuits, parameters)
    assert (
        digest
        == hashlib.sha256(
            json.dumps(
                request.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    )
    journal = AttemptJournal(tmp_path / "private")
    attempt_id = str(uuid4())
    journal.prepare(attempt_id, "f" * 64, (JobPlan("sentinel_before", digest, 1, 1024),))
    journal.begin_submit(attempt_id, 0, digest)
    reopened = AttemptJournal(tmp_path / "private")
    provider_id = uuid4()
    existing = _ExistingJobDirectory(provider_id, _ExistingJob(circuits, parameters))

    assert recover_existing_job(reopened, attempt_id, 0, str(provider_id), existing) == digest
    row = reopened.snapshot(attempt_id)["jobs"][0]
    assert row["state"] == "submitted"
    assert row["provider_job_id"] == str(provider_id)
    assert existing.lookups == 1
    assert request.model_dump(mode="json")["shots"] == 1024
    with pytest.raises(JournalStateError, match="recovery needs"):
        recover_existing_job(reopened, attempt_id, 0, str(provider_id), existing)


def test_wrong_provider_payload_does_not_bind_or_retry(tmp_path: Path) -> None:
    """A different execution parameter leaves the uncertain row untouched."""
    _, circuits, parameters = _request()
    digest = canonical_request_sha256(circuits, parameters)
    journal = AttemptJournal(tmp_path / "private")
    attempt_id = str(uuid4())
    journal.prepare(attempt_id, "f" * 64, (JobPlan("sentinel_before", digest, 1, 1024),))
    journal.begin_submit(attempt_id, 0, digest)
    journal.mark_recovery_required(attempt_id, 0)
    altered = dataclasses.replace(parameters, shots=512)
    provider_id = uuid4()
    existing = _ExistingJobDirectory(provider_id, _ExistingJob(circuits, altered))

    with pytest.raises(JournalStateError, match="payload differs"):
        recover_existing_job(journal, attempt_id, 0, str(provider_id), existing)
    row = AttemptJournal(tmp_path / "private").snapshot(attempt_id)["jobs"][0]
    assert row["state"] == "recovery_required"
    assert row["provider_job_id"] is None
