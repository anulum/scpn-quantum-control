# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Collected pytest CI partition
"""Partition the actual admitted pytest items and record real worker execution."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import asdict
from pathlib import Path
from typing import cast

import pytest

from tools.ci_test_shards import RunIdentity, load_timings, partition_tests, write_json


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register explicit worker options without changing ordinary pytest runs.

    Parameters
    ----------
    parser
        Actual pytest command-line parser.

    """
    group = parser.getgroup("complete CI sharding")
    group.addoption("--ci-shard", type=int)
    group.addoption("--ci-shards", type=int, default=4)
    for name in ("output", "timings", "source-sha", "run-id", "attempt"):
        group.addoption("--ci-" + name)
    group.addoption("--ci-suite", default="python")


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Bind an explicitly requested worker before tests execute.

    Parameters
    ----------
    config
        Parsed pytest configuration, including normal marker admission.

    Raises
    ------
    pytest.UsageError
        If worker options or timing hints are invalid.

    """
    index = cast(int | None, config.getoption("ci_shard"))
    if index is None:
        return
    count = cast(int, config.getoption("ci_shards"))
    values = [
        config.getoption("ci_" + name)
        for name in ("output", "timings", "source_sha", "run_id", "attempt")
    ]
    if not 1 <= index <= count or any(not isinstance(value, str) or not value for value in values):
        raise pytest.UsageError(
            "CI worker requires valid indices, output, timing snapshot and run identity"
        )
    output, timing_path, sha, run, attempt = cast(list[str], values)
    try:
        timings = load_timings(Path(timing_path))
        snapshot = hashlib.sha256(Path(timing_path).read_bytes()).hexdigest()
    except (OSError, ValueError) as error:
        raise pytest.UsageError(str(error)) from error
    directory = Path(output)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise pytest.UsageError("CI worker output already contains data")
    # pytest-cov can create the otherwise empty parent before pytest_configure.
    directory.mkdir(parents=True, exist_ok=True)
    identity = RunIdentity(
        sha,
        run,
        attempt,
        f"{sys.version_info.major}.{sys.version_info.minor}",
        cast(str, config.getoption("ci_suite")),
    )
    config.pluginmanager.register(
        ShardExecution(index, count, directory, identity, timings, snapshot),
        "complete-ci-execution",
    )


class ShardExecution:
    """Record admission and actual setup/call/teardown for one isolated worker."""

    def __init__(
        self,
        index: int,
        count: int,
        output: Path,
        identity: RunIdentity,
        timings: dict[str, float],
        snapshot: str,
    ) -> None:
        """Create one recorder bound to its immutable scheduling snapshot.

        Parameters
        ----------
        index
            One-based worker identity.
        count
            Complete mandatory worker count.
        output
            Worker-owned diagnostic and indexed coverage directory.
        identity
            Expected source, workflow attempt, interpreter and suite.
        timings
            Validated observed scheduling durations.
        snapshot
            SHA-256 of the common immutable timing snapshot.

        """
        self.index = index
        self.count = count
        self.output = output
        self.identity = identity
        self.timings = timings
        self.snapshot = snapshot
        self.inventory: list[str] = []
        self.selected: list[str] = []
        self.completed: set[str] = set()
        self.seconds: dict[str, float] = {}

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        """Partition after normal plugins and marker filtering finish.

        Parameters
        ----------
        session
            Actual collected session, before its test loop.

        """
        self.inventory = sorted(item.nodeid for item in session.items)
        try:
            self.selected = partition_tests(self.inventory, self.count, self.timings)[
                self.index - 1
            ]
        except ValueError as error:
            raise pytest.UsageError(str(error)) from error
        owned = set(self.selected)
        removed = [item for item in session.items if item.nodeid not in owned]
        session.items[:] = [item for item in session.items if item.nodeid in owned]
        session.config.hook.pytest_deselected(items=removed)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Accumulate observed durations and completed execution identities.

        Parameters
        ----------
        report
            Real pytest setup, call or teardown report.

        """
        self.seconds[report.nodeid] = self.seconds.get(report.nodeid, 0.0) + report.duration
        if report.when == "teardown":
            self.completed.add(report.nodeid)

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        """Write diagnostic execution evidence; a separate success step seals it.

        Parameters
        ----------
        session
            Actual pytest session with its final collection mode.
        exitstatus
            Original pytest exit status, preserved without conversion to success.

        """
        write_json(
            self.output / "result.json",
            {
                "schema_version": 1,
                "identity": asdict(self.identity),
                "shard": self.index,
                "shards": self.count,
                "inventory": self.inventory,
                "selected": self.selected,
                "completed": sorted(self.completed),
                "seconds": {
                    node: seconds for node, seconds in self.seconds.items() if seconds > 0
                },
                "timing_snapshot_sha256": self.snapshot,
                "exitstatus": int(exitstatus),
                "collect_only": bool(session.config.option.collectonly),
                "sealed": False,
            },
        )
