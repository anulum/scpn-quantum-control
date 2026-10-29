# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Same-run shard shell integration
"""Qualify the actual concurrent shell runner using maintained runtime tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import coverage
import pytest
from _ci_shard_execution import COHORT, ROOT


@pytest.mark.parametrize("failed", [False, True])
def test_real_concurrent_runner_preserves_complete_union_and_failures(
    tmp_path: Path, failed: bool
) -> None:
    """Run the actual shell process boundary and refuse a real assertion failure."""
    cohort = ROOT / COHORT
    if failed:
        cohort = tmp_path / "test_logging_setup.py"
        source = (ROOT / COHORT).read_text(encoding="utf-8")
        cohort.write_text(
            source.replace(
                "logging.getLogger().level == logging.INFO",
                "logging.getLogger().level == logging.ERROR",
                1,
            ),
            encoding="utf-8",
        )
    output = tmp_path / ".coverage.accepted"
    root = tmp_path / "workers"
    result = subprocess.run(
        [
            "bash",
            "tools/run_ci_test_shards.sh",
            "logging-integration",
            "2",
            str(root),
            str(tmp_path / "absent-timings.json"),
            str(output),
            "*/logging_setup.py",
            "thread",
            str(cohort),
        ],
        cwd=ROOT,
        env=dict(os.environ, CI_SHARD_PYTHON=sys.executable, PYTEST_ADDOPTS=""),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == (1 if failed else 0), result.stdout + result.stderr
    if failed:
        assert "test_default_level_is_info" in result.stdout
        assert not output.exists()
    else:
        data = coverage.CoverageData(basename=str(output))
        data.read()
        assert data.has_arcs() and any(
            path.endswith("/logging_setup.py") for path in data.measured_files()
        )
        inventories = [
            json.loads(path.read_text()) for path in (root / "artifacts").glob("*/result.json")
        ]
        assert len(inventories) == 2
        assert (
            sorted(node for result in inventories for node in result["selected"])
            == inventories[0]["inventory"]
        )
        assert all(result["sealed"] for result in inventories)


def test_shell_runner_refuses_bad_arguments_and_corrupt_timing_cache(tmp_path: Path) -> None:
    """Reject malformed control input before touching accepted coverage."""
    command = ["bash", "tools/run_ci_test_shards.sh"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    source = tmp_path / "bad.json"
    source.write_text("{", encoding="utf-8")
    output = tmp_path / ".coverage.accepted"
    for count in ("9", "2"):
        result = subprocess.run(
            [
                *command,
                "logging-integration",
                count,
                str(tmp_path / ("workers-" + count)),
                str(source),
                str(output),
                "*/logging_setup.py",
                "thread",
                COHORT,
            ],
            cwd=ROOT,
            env=dict(os.environ, CI_SHARD_PYTHON=sys.executable),
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode != 0
        assert not output.exists()
