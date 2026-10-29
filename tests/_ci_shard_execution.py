# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Real CI shard integration execution
"""Run maintained repository tests through the actual CI plugin and CLI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from tools.ci_test_shards import RunIdentity

ROOT = Path(__file__).resolve().parents[1]
PYTHON = f"{sys.version_info.major}.{sys.version_info.minor}"


def _source_identity() -> str:
    """Return the workflow commit, the checked-out commit or the tested index tree.

    Returns
    -------
    str
        ``GITHUB_SHA`` when set, otherwise ``HEAD``; a checkout without a
        commit (the container image's synthetic index) yields its index tree,
        matching ``tools/run_ci_test_shards.sh``.

    """
    explicit = os.environ.get("GITHUB_SHA")
    if explicit:
        return explicit
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--verify", "--quiet", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if head.returncode == 0:
        return head.stdout.strip()
    return subprocess.check_output(["git", "-C", str(ROOT), "write-tree"], text=True).strip()


SOURCE_SHA = _source_identity()
IDENTITY = RunIdentity(
    SOURCE_SHA,
    os.environ.get("GITHUB_RUN_ID", "local-integration"),
    os.environ.get("GITHUB_RUN_ATTEMPT", "1"),
    PYTHON,
)
COHORT = "tests/test_logging_setup.py"


def execute(
    arguments: list[str], *, data_file: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Execute a real pytest or shard CLI command with bounded process time.

    Parameters
    ----------
    arguments
        Module invocation and explicit production command arguments.
    data_file
        Actual worker branch-data destination, when pytest-cov owns tracing.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Unmodified process status and captured diagnostic output.

    """
    env = dict(os.environ, PYTEST_ADDOPTS="")
    quality = env.get("CI_SHARD_TOOL_COVERAGE_DIR")
    args = [sys.executable]
    if quality and data_file is None:
        destination = Path(quality) / (".coverage.command-" + uuid.uuid4().hex)
        destination.parent.mkdir(parents=True, exist_ok=True)
        args.extend(
            [
                "-m",
                "coverage",
                "run",
                "--rcfile=/dev/null",
                "--branch",
                "--source=tools.ci_test_shards,tools.pytest_ci_shard",
                "--data-file=" + str(destination),
            ]
        )
    if data_file is not None:
        env["COVERAGE_FILE"] = str(data_file)
    result = subprocess.run(
        [*args, *arguments], cwd=ROOT, env=env, capture_output=True, text=True, timeout=90
    )
    if quality and data_file is not None and data_file.is_file():
        shutil.copyfile(data_file, Path(quality) / (".coverage.worker-" + uuid.uuid4().hex))
    return result


def run_workers(
    root: Path,
    *,
    count: int = 2,
    timing_source: Path | None = None,
    cohort: list[str] | None = None,
    with_coverage: bool = True,
    seal_success: bool = True,
) -> tuple[Path, list[subprocess.CompletedProcess[str]]]:
    """Execute a complete scoped repository cohort through independent workers.

    Parameters
    ----------
    root
        Independent invocation directory.
    count
        Worker count.
    timing_source
        Prior actual observations, or a cold missing snapshot.
    cohort
        Maintained test paths and optional real pytest selection flags.
    with_coverage
        Collect real branch data and seal it on successful workers.
    seal_success
        Seal successful executions, or retain diagnostic evidence for refusal tests.

    Returns
    -------
    tuple[Path, list[subprocess.CompletedProcess[str]]]
        Artifact root and actual pytest outcomes.

    """
    root.mkdir(parents=True)
    snapshot = root / "timings.json"
    source = timing_source or root / "absent-cache.json"
    prepared = execute(
        [
            "-m",
            "tools.ci_test_shards",
            "snapshot",
            "--source",
            str(source),
            "--output",
            str(snapshot),
        ]
    )
    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    artifacts = root / "artifacts"
    artifacts.mkdir()
    outcomes = []
    for index in range(1, count + 1):
        output = artifacts / IDENTITY.directory(index)
        args = [
            "-m",
            "pytest",
            "--noconftest",
            "-q",
            "-p",
            "tools.pytest_ci_shard",
            *(cohort or [COHORT]),
            "--ci-shard",
            str(index),
            "--ci-shards",
            str(count),
            "--ci-output",
            str(output),
            "--ci-timings",
            str(snapshot),
            "--ci-source-sha",
            IDENTITY.source_sha,
            "--ci-run-id",
            IDENTITY.run_id,
            "--ci-attempt",
            IDENTITY.attempt,
        ]
        data = output / f".coverage.shard-{index}" if with_coverage else None
        if with_coverage:
            args.extend(
                [
                    "--cov=tools.ci_test_shards",
                    "--cov=tools.pytest_ci_shard",
                    "--cov=scpn_quantum_control.logging_setup",
                    "--cov-branch",
                    "--cov-report=",
                    "--cov-fail-under=0",
                ]
            )
        result = execute(args, data_file=data)
        outcomes.append(result)
        if result.returncode == 0 and seal_success:
            sealed = execute(
                [
                    "-m",
                    "tools.ci_test_shards",
                    "seal",
                    str(output),
                    *(["--coverage"] if with_coverage else []),
                ]
            )
            assert sealed.returncode == 0, sealed.stdout + sealed.stderr
    return artifacts, outcomes


def verification_command(artifacts: Path, count: int = 2) -> list[str]:
    """Return the real aggregate CLI command for an integration invocation.

    Parameters
    ----------
    artifacts
        Same-run artifact directory.
    count
        Exact expected worker count.

    Returns
    -------
    list[str]
        Explicit CLI arguments using the current interpreter identity.

    """
    return [
        "-m",
        "tools.ci_test_shards",
        "verify",
        str(artifacts),
        "--source-sha",
        IDENTITY.source_sha,
        "--run-id",
        IDENTITY.run_id,
        "--attempt",
        IDENTITY.attempt,
        "--python",
        PYTHON,
        "--shards",
        str(count),
    ]


def read_worker(artifacts: Path, index: int = 1) -> dict[str, Any]:
    """Read actual runner evidence for an isolated corruption experiment.

    Parameters
    ----------
    artifacts
        Actual integration artifacts.
    index
        One-based worker index.

    Returns
    -------
    dict[str, Any]
        Actual JSON result emitted and sealed by the production runner.

    """
    payload: dict[str, Any] = json.loads(
        (artifacts / IDENTITY.directory(index) / "result.json").read_text()
    )
    return payload
