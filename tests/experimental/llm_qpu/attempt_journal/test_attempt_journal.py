# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — durable attempt journal acceptance
"""Exercise real SQLite custody and a process death at the submit boundary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from scpn_quantum_control.experimental.llm_qpu.runtime import (
    AttemptJournal,
    JobPlan,
    JournalStateError,
)


def _jobs() -> tuple[JobPlan, ...]:
    return (
        JobPlan("sentinel_before", "a" * 64, 2, 1024),
        JobPlan("data_first", "b" * 64, 48, 256),
    )


def test_real_process_death_keeps_submit_ambiguous(tmp_path: Path) -> None:
    """A killed local worker cannot turn a durable SUBMITTING row into retry authority.

    The tested boundary is local disk/process durability. A real paid provider
    accept-then-disconnect cannot be created inside repository CI; provider
    reconciliation remains a separate hardware qualification.
    """
    root = tmp_path / "private"
    attempt = str(uuid4())
    journal = AttemptJournal(root)
    journal.prepare(attempt, "f" * 64, _jobs())
    child = (
        "import os,sys; from pathlib import Path; "
        "from scpn_quantum_control.experimental.llm_qpu.runtime import AttemptJournal; "
        "AttemptJournal(Path(sys.argv[1])).begin_submit(sys.argv[2],0,sys.argv[3]); "
        "os._exit(0)"
    )
    subprocess.run(
        [sys.executable, "-c", child, str(root), attempt, "a" * 64],
        check=True,
        timeout=30,
    )

    reopened = AttemptJournal(root)
    assert reopened.snapshot(attempt)["jobs"][0]["state"] == "submitting"
    with pytest.raises(JournalStateError, match="never retry"):
        reopened.begin_submit(attempt, 0, "a" * 64)
    reopened.mark_recovery_required(attempt, 0)
    assert reopened.snapshot(attempt)["jobs"][0]["state"] == "recovery_required"
    with pytest.raises(JournalStateError, match="never retry"):
        reopened.begin_submit(attempt, 0, "a" * 64)
    assert reopened.snapshot(attempt)["jobs"][0]["provider_job_id"] is None


def test_ordered_jobs_need_verified_prior_result(tmp_path: Path) -> None:
    """Sentinel A must be verified before the next paid job can cross the boundary."""
    journal = AttemptJournal(tmp_path / "private")
    attempt = str(uuid4())
    journal.prepare(attempt, "f" * 64, _jobs())
    with pytest.raises(JournalStateError, match="earlier job"):
        journal.begin_submit(attempt, 1, "b" * 64)
    with pytest.raises(JournalStateError, match="payload differs"):
        journal.begin_submit(attempt, 0, "0" * 64)

    journal.begin_submit(attempt, 0, "a" * 64)
    provider_id = str(uuid4())
    journal.record_submitted(attempt, 0, provider_id)
    with pytest.raises(JournalStateError, match="earlier job"):
        journal.begin_submit(attempt, 1, "b" * 64)
    journal.record_verified(attempt, 0, "c" * 64)
    journal.begin_submit(attempt, 1, "b" * 64)
    journal.record_submitted(attempt, 1, str(uuid4()))
    rows = journal.snapshot(attempt)["jobs"]
    assert [(row["state"], row["payload_sha256"]) for row in rows] == [
        ("verified", "a" * 64),
        ("submitted", "b" * 64),
    ]
    assert rows[0]["provider_job_id"] == provider_id
    assert rows[0]["result_sha256"] == "c" * 64


def test_private_disk_and_attempt_identity(tmp_path: Path) -> None:
    """Only one immutable plan can own an attempt ID in a private SQLite root."""
    root = tmp_path / "private"
    journal = AttemptJournal(root)
    attempt = str(uuid4())
    journal.prepare(attempt, "f" * 64, _jobs())
    assert root.stat().st_mode & 0o777 == 0o700
    assert (root / "attempts.sqlite3").stat().st_mode & 0o777 == 0o600
    with pytest.raises(JournalStateError, match="already exists"):
        journal.prepare(attempt, "e" * 64, _jobs())
    assert journal.snapshot(attempt)["bundle_sha256"] == "f" * 64

    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        AttemptJournal(alias)
