# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — workflow job timeout gate tests
"""Check that the timeout gate refuses unbounded jobs and accepts the repository."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

from tools import audit_workflow_job_timeouts as gate


def _workflow(tmp_path: Path, body: str) -> Path:
    """Write one workflow file under a fresh repository root.

    Parameters
    ----------
    tmp_path
        Owned directory used as the repository root.
    body
        YAML text of the workflow.

    Returns
    -------
    Path
        The repository root.

    """
    workflows = tmp_path / ".github/workflows"
    workflows.mkdir(parents=True)
    (workflows / "build.yml").write_text(body, encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    ("job", "finding"),
    [
        ("runs-on: x\n    steps: []", "has no timeout-minutes"),
        (
            "runs-on: x\n    timeout-minutes: ${{ inputs.t }}",
            "timeout-minutes must be a literal integer",
        ),
        ("runs-on: x\n    timeout-minutes: true", "timeout-minutes must be a literal integer"),
        ("runs-on: x\n    timeout-minutes: 0", "timeout-minutes 0 is outside 1 to 359"),
        ("runs-on: x\n    timeout-minutes: 360", "timeout-minutes 360 is outside 1 to 359"),
    ],
)
def test_unbounded_jobs_are_refused(tmp_path: Path, job: str, finding: str) -> None:
    """Missing, non-literal and out-of-range timeouts each produce a finding.

    Parameters
    ----------
    tmp_path
        Owned repository root.
    job
        YAML body of the single job.
    finding
        Expected text of the finding.

    """
    repo = _workflow(tmp_path, f"jobs:\n  build:\n    {job}\n")
    assert gate.audit(repo) == [f".github/workflows/build.yml: job build {finding}"]
    assert gate.main(["--repo", str(repo)]) == 1


def test_bounded_and_reusable_jobs_pass(tmp_path: Path) -> None:
    """A bounded job passes, and a reusable-workflow call is checked in its own file.

    Parameters
    ----------
    tmp_path
        Owned repository root.

    """
    repo = _workflow(
        tmp_path,
        "jobs:\n  build:\n    runs-on: x\n    timeout-minutes: 30\n"
        "  call:\n    uses: ./.github/workflows/other.yml\n  note: text\n",
    )
    assert gate.audit(repo) == []
    assert gate.job_timeout_findings("w.yml", {"jobs": []}) == ["w.yml: jobs must be a mapping"]
    assert gate.job_timeout_findings("w.yml", None) == ["w.yml: jobs must be a mapping"]


def test_repository_workflows_are_all_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every job in this repository's workflows declares a bounded timeout.

    Parameters
    ----------
    monkeypatch
        Supplies the real command's argument vector.

    """
    assert gate.audit(gate.REPOSITORY_ROOT) == []
    monkeypatch.setattr(sys, "argv", ["audit_workflow_job_timeouts.py"])
    with pytest.raises(SystemExit) as exit_status:
        runpy.run_path(
            str(gate.REPOSITORY_ROOT / "tools/audit_workflow_job_timeouts.py"), run_name="__main__"
        )
    assert exit_status.value.code == 0
