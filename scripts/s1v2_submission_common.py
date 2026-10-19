# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — S1v2 submission journal and stage gates
"""Provider-neutral journal, stage gates and count custody for the S1v2 submitters.

A journal is written *before* every provider call. A stage whose journal
exists is never submitted again: a journal without a job identifier means the
provider call may or may not have happened, and only a person may resolve it.
The main stage refuses to run unless the same backend's pilot analysis passed
the preregistered coherence floor.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STAGES = ("pilot", "main")


class StageRefused(RuntimeError):
    """A stage gate refused to submit."""


def utc_stamp() -> str:
    """Return a compact UTC timestamp for file names."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 of bytes."""
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> str:
    """Write JSON atomically (temporary file, fsync, rename) and return its SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n").encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256_bytes(data)


def journal_path(out_dir: Path, provider: str, backend: str, stage: str) -> Path:
    """Return the single journal path of one provider/backend/stage."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    return out_dir / provider / f"{backend}_{stage}_journal.json"


def open_journal(
    path: Path,
    *,
    experiment_id: str,
    provider: str,
    backend: str,
    stage: str,
    logical_digest: str,
    isa_digest: str,
    labels: Sequence[str],
    shots: Mapping[str, int],
    layout: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the pre-submission journal, refusing if one already exists.

    Raises
    ------
    StageRefused
        If a journal for this backend and stage already exists.
    """
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        raise StageRefused(
            f"{stage} on {backend} already has a journal (status {existing.get('status')!r}); "
            "a stage is submitted at most once — retrieve it or resolve it by hand"
        )
    journal = {
        "schema": "s1v2_submission_journal_v1",
        "experiment_id": experiment_id,
        "provider": provider,
        "backend": backend,
        "stage": stage,
        "status": "submitting",
        "opened_utc": utc_stamp(),
        "logical_digest": logical_digest,
        "isa_digest": isa_digest,
        "labels": list(labels),
        "shots": dict(shots),
        "layout": dict(layout),
        "job_ids": [],
    }
    write_json(path, journal)
    return journal


def record_job(path: Path, journal: dict[str, Any], job_id: str) -> None:
    """Record a provider job identifier in the journal and mark it submitted."""
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("provider returned an empty job identifier")
    journal["job_ids"].append(job_id)
    journal["status"] = "submitted"
    journal["submitted_utc"] = utc_stamp()
    write_json(path, journal)


def require_pilot_go(pilot_analysis_path: Path, backend: str) -> dict[str, Any]:
    """Return the pilot analysis if it permits the main stage on this backend.

    Raises
    ------
    StageRefused
        If the analysis is for another backend or did not pass the floor.
    """
    analysis: dict[str, Any] = json.loads(pilot_analysis_path.read_text(encoding="utf-8"))
    if analysis.get("backend") != backend:
        raise StageRefused(f"pilot analysis is for {analysis.get('backend')!r}, not {backend!r}")
    if analysis.get("main_block_go") is not True:
        raise StageRefused("pilot coherence is below the preregistered floor; main stage stopped")
    return analysis


def coerce_counts(raw: Mapping[Any, Any], expected_shots: int) -> dict[str, int]:
    """Return counts with string keys and integer values summing to the shot number.

    Raises
    ------
    ValueError
        On a non-integer, negative or non-conserving count.
    """
    counts: dict[str, int] = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid count {value!r} for {key!r}")
        counts[str(key)] = counts.get(str(key), 0) + value
    if sum(counts.values()) != expected_shots:
        raise ValueError(f"counts sum to {sum(counts.values())}, expected {expected_shots}")
    return counts
