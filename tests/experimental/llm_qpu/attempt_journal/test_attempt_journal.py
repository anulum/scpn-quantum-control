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
from typing import Any, cast
from uuid import uuid4

import pytest

from scpn_quantum_control.experimental.llm_qpu.runtime import (
    AttemptJournal,
    JobPlan,
    JournalStateError,
)


def _rows(journal: AttemptJournal, attempt: str) -> list[dict[str, Any]]:
    """Return the journal's job rows of one attempt."""
    return cast("list[dict[str, Any]]", journal.snapshot(attempt)["jobs"])


def _jobs() -> tuple[JobPlan, ...]:
    """Return the two-phase plan: a sentinel job, then the first data job."""
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
    assert _rows(reopened, attempt)[0]["state"] == "submitting"
    with pytest.raises(JournalStateError, match="never retry"):
        reopened.begin_submit(attempt, 0, "a" * 64)
    reopened.mark_recovery_required(attempt, 0)
    assert _rows(reopened, attempt)[0]["state"] == "recovery_required"
    with pytest.raises(JournalStateError, match="never retry"):
        reopened.begin_submit(attempt, 0, "a" * 64)
    assert _rows(reopened, attempt)[0]["provider_job_id"] is None


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
    rows = _rows(journal, attempt)
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


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (("X_bad", "a" * 64, 1, 1), "invalid provider job phase name"),
        ((7, "a" * 64, 1, 1), "invalid provider job phase name"),
        (("phase", "A" * 64, 1, 1), "provider request digest must be lowercase SHA-256"),
        (("phase", None, 1, 1), "provider request digest must be lowercase SHA-256"),
        (("phase", "a" * 64, 0, 1), "provider job circuit count must be positive"),
        (("phase", "a" * 64, True, 1), "provider job circuit count must be positive"),
        (("phase", "a" * 64, 1, 0), "provider job shots must be positive"),
        (("phase", "a" * 64, 1, 2.0), "provider job shots must be positive"),
    ],
)
def test_job_plan_refuses_malformed_identity(fields: tuple[Any, ...], message: str) -> None:
    """A job plan refuses a bad phase name, digest, circuit count or shot count."""
    with pytest.raises(ValueError, match=f"^{message}$"):
        JobPlan(*fields)


@pytest.mark.parametrize(
    ("attempt", "message"),
    [
        (7, "attempt or job ID must be UUID text"),
        ("not-a-uuid", "attempt or job ID must be UUID text"),
        (
            "{12345678-1234-5678-1234-567812345678}",
            "attempt or job ID must be canonical UUID text",
        ),
        ("12345678-1234-5678-1234-56781234567A", "attempt or job ID must be canonical UUID text"),
    ],
)
def test_journal_refuses_noncanonical_attempt_ids(
    tmp_path: Path, attempt: Any, message: str
) -> None:
    """Attempt and job IDs must be canonical lowercase UUID text before any query."""
    journal = AttemptJournal(tmp_path / "private")
    with pytest.raises(ValueError, match=f"^{message}$"):
        journal.snapshot(attempt)


def test_journal_refuses_unsafe_roots_and_databases(tmp_path: Path) -> None:
    """The root must be a private directory and the database a private regular file."""
    not_a_path: Any = str(tmp_path / "private")
    with pytest.raises(TypeError, match="^journal root must be a Path$"):
        AttemptJournal(not_a_path)
    for mode in (0o750, 0o705):
        shared = tmp_path / f"shared-{mode:o}"
        shared.mkdir(mode=0o700)
        shared.chmod(mode)
        with pytest.raises(
            ValueError, match=r"^journal directory must be owned and private \(0700\)$"
        ):
            AttemptJournal(shared)
    linked = tmp_path / "linked"
    linked.mkdir(mode=0o700)
    (tmp_path / "elsewhere.sqlite3").touch(mode=0o600)
    (linked / "attempts.sqlite3").symlink_to(tmp_path / "elsewhere.sqlite3")
    with pytest.raises(ValueError, match="^journal database must not be a symlink$"):
        AttemptJournal(linked)
    for mode in (0o640, 0o604):
        readable = tmp_path / f"readable-{mode:o}"
        readable.mkdir(mode=0o700)
        database = readable / "attempts.sqlite3"
        database.touch(mode=0o600)
        database.chmod(mode)
        with pytest.raises(
            ValueError, match=r"^journal database must be owned and private \(0600\)$"
        ):
            AttemptJournal(readable)


def test_prepare_refuses_invalid_inventories_and_digests(tmp_path: Path) -> None:
    """Preparation needs a bundle digest and a nonempty tuple of distinct plans."""
    journal = AttemptJournal(tmp_path / "private")
    attempt = str(uuid4())
    with pytest.raises(ValueError, match="^digest must be lowercase SHA-256$"):
        journal.prepare(attempt, "F" * 64, _jobs())
    listed: Any = list(_jobs())
    mixed: Any = (_jobs()[0], "data_first")
    for jobs in (listed, (), mixed):
        with pytest.raises(ValueError, match="^attempt needs a nonempty immutable job tuple$"):
            journal.prepare(attempt, "f" * 64, jobs)
    twice = (_jobs()[0], JobPlan("sentinel_before", "b" * 64, 1, 1))
    with pytest.raises(ValueError, match="^provider job phase names must be distinct$"):
        journal.prepare(attempt, "f" * 64, twice)
    with pytest.raises(JournalStateError, match="^unknown attempt ID$"):
        journal.snapshot(attempt)


def test_transitions_refuse_unknown_jobs_and_out_of_order_states(tmp_path: Path) -> None:
    """Every transition refuses bad ordinals, unknown jobs and a state it cannot follow."""
    journal = AttemptJournal(tmp_path / "private")
    attempt = str(uuid4())
    journal.prepare(attempt, "f" * 64, _jobs())
    bad_ordinal: Any = "0"
    for ordinal, error, message in (
        (-1, ValueError, "^job ordinal must be a nonnegative integer$"),
        (bad_ordinal, ValueError, "^job ordinal must be a nonnegative integer$"),
        (2, JournalStateError, "^unknown attempt job$"),
    ):
        with pytest.raises(error, match=message):
            journal.begin_submit(attempt, ordinal, "a" * 64)
    with pytest.raises(JournalStateError, match="^only an uncertain submission can require"):
        journal.mark_recovery_required(attempt, 0)
    with pytest.raises(JournalStateError, match="^provider job ID needs an active submitting"):
        journal.record_submitted(attempt, 0, str(uuid4()))
    with pytest.raises(JournalStateError, match="^raw result cannot verify an unsubmitted job$"):
        journal.record_verified(attempt, 0, "c" * 64)
    with pytest.raises(JournalStateError, match="^recovery needs an uncertain submit boundary$"):
        journal.record_recovered(attempt, 0, str(uuid4()), "a" * 64)
    assert [row["state"] for row in _rows(journal, attempt)] == ["prepared", "prepared"]


@pytest.mark.parametrize("uncertain", [False, True])
def test_recovery_binds_only_the_prepared_payload(tmp_path: Path, uncertain: bool) -> None:
    """An ambiguous submission is bound only with read-only proof of the same payload."""
    journal = AttemptJournal(tmp_path / "private")
    attempt = str(uuid4())
    journal.prepare(attempt, "f" * 64, _jobs())
    journal.begin_submit(attempt, 0, "a" * 64)
    if uncertain:
        journal.mark_recovery_required(attempt, 0)
    provider_id = str(uuid4())
    with pytest.raises(JournalStateError, match="^provider job payload differs from prepared"):
        journal.record_recovered(attempt, 0, provider_id, "d" * 64)
    assert _rows(journal, attempt)[0]["provider_job_id"] is None
    journal.record_recovered(attempt, 0, provider_id, "a" * 64)
    row = _rows(journal, attempt)[0]
    assert (row["state"], row["provider_job_id"]) == ("submitted", provider_id)
    with pytest.raises(JournalStateError, match="^recovery needs an uncertain submit boundary$"):
        journal.record_recovered(attempt, 0, provider_id, "a" * 64)
    journal.record_verified(attempt, 0, "c" * 64)
    assert _rows(journal, attempt)[0]["state"] == "verified"
