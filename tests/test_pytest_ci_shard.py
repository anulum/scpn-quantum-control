# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Collected pytest shard integration tests
"""Exercise real marker selection, added admission and failed test custody."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _ci_shard_execution import (
    COHORT,
    IDENTITY,
    ROOT,
    execute,
    read_worker,
    run_workers,
    verification_command,
)


def test_weighted_runner_admits_new_real_cases_from_live_discovery(tmp_path: Path) -> None:
    """Grow an actual maintained cohort beyond its previous measured selection."""
    first, outcomes = run_workers(
        tmp_path / "first", cohort=[COHORT, "-k", "TestConfigureLogging"], with_coverage=False
    )
    assert all(result.returncode == 0 for result in outcomes), [
        result.stdout + result.stderr for result in outcomes
    ]
    hints = tmp_path / "observed.json"
    measured = execute([*verification_command(first), "--timings-output", str(hints)])
    assert measured.returncode == 0, measured.stdout + measured.stderr
    second, outcomes = run_workers(
        tmp_path / "second",
        timing_source=hints,
        cohort=[COHORT, "tests/test_config.py"],
        with_coverage=False,
    )
    assert all(result.returncode == 0 for result in outcomes), [
        result.stdout + result.stderr for result in outcomes
    ]
    admitted = execute(verification_command(second))
    assert admitted.returncode == 0
    assert json.loads(admitted.stdout)["tests"] > json.loads(measured.stdout)["tests"]
    assert set(read_worker(first)["inventory"]) < set(read_worker(second)["inventory"])


def test_actual_public_test_failure_cannot_be_sealed(tmp_path: Path) -> None:
    """Mutate an expectation in a copy of the real public logging cohort."""
    damaged = tmp_path / "test_logging_setup.py"
    source = (ROOT / COHORT).read_text(encoding="utf-8")
    assert "logging.getLogger().level == logging.INFO" in source
    damaged.write_text(
        source.replace(
            "logging.getLogger().level == logging.INFO",
            "logging.getLogger().level == logging.ERROR",
            1,
        ),
        encoding="utf-8",
    )
    artifacts, outcomes = run_workers(
        tmp_path / "failed", cohort=[str(damaged)], count=1, with_coverage=False
    )
    assert outcomes[0].returncode == 1
    assert "test_default_level_is_info" in outcomes[0].stdout
    original: dict[str, Any] = read_worker(artifacts)
    assert original["exitstatus"] == 1 and original["sealed"] is False
    result = execute(
        ["-m", "tools.ci_test_shards", "seal", str(artifacts / IDENTITY.directory(1))]
    )
    assert result.returncode == 1 and "successful test execution" in result.stderr
    assert execute(verification_command(artifacts, count=1)).returncode == 1


@pytest.mark.parametrize(
    "extra", [["--ci-shard", "0"], ["--ci-shard", "1"], ["--ci-shard", "1", "--ci-shards", "0"]]
)
def test_real_pytest_rejects_incomplete_worker_arguments(extra: list[str]) -> None:
    """Refuse partial worker commands before any maintained test executes."""
    result = execute(
        ["-m", "pytest", "--noconftest", "-q", "-p", "tools.pytest_ci_shard", COHORT, *extra]
    )
    assert result.returncode == 4 and "CI worker requires" in result.stderr


def test_plugin_preserves_ordinary_pytest_execution() -> None:
    """Keep a normal explicit plugin invocation equivalent to the maintained cohort."""
    result = execute(["-m", "pytest", "--noconftest", "-q", "-p", "tools.pytest_ci_shard", COHORT])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "passed" in result.stdout and "deselected" not in result.stdout


def test_collection_only_is_not_execution_evidence(tmp_path: Path) -> None:
    """Refuse a real collection-only result even when pytest itself exits zero."""
    artifacts, outcomes = run_workers(
        tmp_path / "collect",
        cohort=[COHORT, "--collect-only"],
        count=1,
        with_coverage=False,
        seal_success=False,
    )
    assert outcomes[0].returncode == 0
    result = execute(
        ["-m", "tools.ci_test_shards", "seal", str(artifacts / IDENTITY.directory(1))]
    )
    assert result.returncode == 1 and "successful test execution" in result.stderr


@pytest.mark.parametrize("fault", ["corrupt", "absent", "existing-output", "empty-selection"])
def test_real_pytest_refuses_invalid_snapshot_and_discovery(tmp_path: Path, fault: str) -> None:
    """Refuse damaged inputs and empty admission before executing the cohort."""
    timing = tmp_path / "timings.json"
    if fault != "absent":
        timing.write_text(
            "{" if fault == "corrupt" else '{"schema_version":1,"seconds":{}}', encoding="utf-8"
        )
    output = tmp_path / "output"
    if fault == "existing-output":
        output.mkdir()
        (output / "result.json").write_text("{}", encoding="utf-8")
    args = [
        "-m",
        "pytest",
        "--noconftest",
        "-q",
        "-p",
        "tools.pytest_ci_shard",
        COHORT,
        "--ci-shard",
        "1",
        "--ci-shards",
        "1",
        "--ci-output",
        str(output),
        "--ci-timings",
        str(timing),
        "--ci-source-sha",
        IDENTITY.source_sha,
        "--ci-run-id",
        IDENTITY.run_id,
        "--ci-attempt",
        IDENTITY.attempt,
    ]
    if fault == "empty-selection":
        args.extend(["-k", "no_such_real_logging_case"])
    result = execute(args)
    assert result.returncode == 4, result.stdout + result.stderr
    assert "passed" not in result.stdout
