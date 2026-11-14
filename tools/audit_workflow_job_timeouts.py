# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — bounded timeout for every workflow job
"""Fail when a workflow job could run without a bounded timeout.

A job without ``timeout-minutes`` runs for GitHub's default of 360 minutes. A
hung job then holds a runner for six hours and reports as cancelled instead of
failing near its real duration. This gate requires every job that runs steps
to declare a literal integer timeout between 1 and 359 minutes. Jobs that call
a reusable workflow cannot carry ``timeout-minutes``; the jobs inside the
called workflow are checked in their own file.

The bounds themselves come from measured hosted durations; this gate only
keeps a job from being added without one.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import yaml

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_TIMEOUT_MINUTES: Final = 360


def job_timeout_findings(path: str, workflow: object) -> list[str]:
    """Return one finding per job in ``workflow`` without a bounded timeout.

    Parameters
    ----------
    path
        Workflow path used in the findings.
    workflow
        Decoded workflow document.

    Returns
    -------
    list[str]
        Findings in job order; empty when every step-running job is bounded.

    """
    jobs = workflow.get("jobs") if isinstance(workflow, Mapping) else None
    if not isinstance(jobs, Mapping):
        return [f"{path}: jobs must be a mapping"]
    findings: list[str] = []
    for job_id, job in jobs.items():
        if not isinstance(job, Mapping) or "uses" in job:
            continue
        timeout = job.get("timeout-minutes")
        if timeout is None:
            findings.append(f"{path}: job {job_id} has no timeout-minutes")
        elif isinstance(timeout, bool) or not isinstance(timeout, int):
            findings.append(f"{path}: job {job_id} timeout-minutes must be a literal integer")
        elif not 1 <= timeout < DEFAULT_TIMEOUT_MINUTES:
            findings.append(
                f"{path}: job {job_id} timeout-minutes {timeout} is outside 1 to "
                f"{DEFAULT_TIMEOUT_MINUTES - 1}"
            )
    return findings


def audit(repo: Path) -> list[str]:
    """Check every workflow file of ``repo``.

    Parameters
    ----------
    repo
        Repository root holding ``.github/workflows``.

    Returns
    -------
    list[str]
        Findings across all workflows, in file order.

    """
    findings: list[str] = []
    for path in sorted((repo / ".github/workflows").glob("*.yml")):
        relative = path.relative_to(repo).as_posix()
        findings.extend(
            job_timeout_findings(relative, yaml.safe_load(path.read_text(encoding="utf-8")))
        )
    return findings


def main(argv: Sequence[str] | None = None) -> int:
    """Print findings and return 1 when any job is unbounded.

    Parameters
    ----------
    argv
        Command-line arguments; ``--repo`` selects the repository root.

    Returns
    -------
    int
        0 when every job is bounded, 1 otherwise.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPOSITORY_ROOT)
    args = parser.parse_args(argv)
    findings = audit(args.repo)
    for finding in findings:
        print(finding, file=sys.stderr)
    print(f"workflow job timeouts: {len(findings)} unbounded job(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
