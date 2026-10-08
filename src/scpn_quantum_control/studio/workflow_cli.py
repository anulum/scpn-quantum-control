# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — checkpointed original workflow command line
"""Run original executive workflows with atomic lossless journal saves.

SIGINT requests cancellation at synchronous original action boundaries. It does
not claim that an in-flight numerical action was interrupted or disposed.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import signal
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from types import FrameType

from ..studio_workspace.json_transport import read_json, write_json
from .workflow_contracts import MAX_WORKFLOW_BYTES, parse_workflow
from .workflow_execution import run_workflow
from .workflow_journal import MAX_JOURNAL_BYTES, WorkflowJournal, parse_workflow_journal


def _read(path: Path, limit: int) -> str:
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("workflow document exceeds byte bound")
    return raw.decode("utf-8", errors="strict")


class _JournalStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.prior = path.read_bytes() if path.exists() else None

    def save(self, journal: WorkflowJournal) -> None:
        text = write_json(journal.to_dict()) + "\n"
        encoded = text.encode("utf-8")
        if len(encoded) > MAX_JOURNAL_BYTES:
            raise ValueError("workflow journal exceeds byte bound")
        current = self.path.read_bytes() if self.path.exists() else None
        if current != self.prior:
            raise OSError("checkpoint changed outside this workflow run")
        descriptor, name = tempfile.mkstemp(
            prefix=".workflow-", suffix=".partial", dir=self.path.parent
        )
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            self.prior = encoded
        finally:
            temporary.unlink(missing_ok=True)


def run(argv: Sequence[str] | None = None) -> int:
    """Dispatch one original graph with a locked atomic checkpoint file.

    Parameters
    ----------
    argv
        Explicit arguments, or process arguments when omitted. An existing
        journal requires --resume; --approve is current explicit host permission.

    Returns
    -------
    int
        Zero for complete, one for explicit partial/gated/failed, two for refused
        input or storage, and 130 for cooperative cancellation. Caller errors use
        fixed text; original result diagnostics remain in the journal.

    """
    parser = argparse.ArgumentParser(
        prog="scpn-studio-workflow",
        description="Run original Studio stages and retain their lossless attempt journal",
    )
    parser.add_argument("workflow", help="portable experiment_workflow.v1 JSON path")
    parser.add_argument("--journal", required=True, help="atomic original checkpoint path")
    parser.add_argument(
        "--resume", action="store_true", help="resume the exact matching original checkpoint"
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help="current explicit approval through the original executive gate",
    )
    ns = parser.parse_args(sys.argv[1:] if argv is None else argv)
    cancellation = False

    def cancel(signum: int, frame: FrameType | None) -> None:
        nonlocal cancellation
        cancellation = True

    previous = signal.getsignal(signal.SIGINT)
    try:
        definition = parse_workflow(read_json(_read(Path(ns.workflow), MAX_WORKFLOW_BYTES)))
        target = Path(ns.journal)
        if target.resolve() == Path(ns.workflow).resolve():
            raise ValueError("workflow source cannot be its own journal")
        lock_name = ".workflow-" + hashlib.sha256(os.fsencode(target.name)).hexdigest() + ".lock"
        with target.with_name(lock_name).open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if target.exists() != ns.resume:
                raise ValueError(
                    "existing checkpoints require resume; resume requires a checkpoint"
                )
            prior = (
                parse_workflow_journal(read_json(_read(target, MAX_JOURNAL_BYTES)), definition)
                if ns.resume
                else None
            )
            store = _JournalStore(target)
            signal.signal(signal.SIGINT, cancel)
            journal = run_workflow(
                definition,
                journal=prior,
                checkpoint=store.save,
                cancelled=lambda: cancellation,
                approved=ns.approve,
            )
        print(write_json(journal.to_dict()))
        return 0 if journal.state == "complete" else 130 if journal.state == "cancelled" else 1
    except (OSError, ValueError, KeyError, TypeError):
        print(
            "scpn-studio-workflow: source, checkpoint or original runtime refused", file=sys.stderr
        )
        return 2
    finally:
        signal.signal(signal.SIGINT, previous)


def main() -> None:
    """Invoke the registered checkpointed original workflow command."""
    raise SystemExit(run())


if __name__ == "__main__":
    main()
