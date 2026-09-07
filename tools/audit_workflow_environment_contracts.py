# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — workflow environment contract gate
"""Fail when a CI job invokes something the job itself never installs.

A job that has never run is not a job that passes. The ``julia-lint`` job was
added with a step that runs ``python -m pytest`` and an install step that
brought only the Julia tier, which carries no pytest; a second fault left the
job without a path to the first-party package, so ``tests/conftest.py`` could
not import it either. Neither fault was visible in review and neither could be
caught by ``actionlint``, which checks workflow syntax rather than whether a
job's environment supports its own commands.

This gate reads each job as a contract between what it installs and what it
runs, and checks three things a runner would otherwise discover the hard way:

* every ``python -m <module>`` a job invokes is provided by a requirement file
  that same job installs, or is part of the standard library;
* every job that runs pytest over ``tests/`` makes the first-party package
  importable, because the shared ``conftest.py`` imports it during collection;
* a tool pinned by version in more than one workflow is pinned to the same
  version in all of them.

The checks are static. They read the workflow YAML and the requirement locks,
run anywhere, and need no network, no runner and no interpreter beyond the one
executing them.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

#: Modules that ship with CPython, so no requirement file has to provide them.
STDLIB_MODULES: Final[frozenset[str]] = frozenset({"venv", "pip", "ensurepip", "json", "unittest"})

#: Import roots this repository owns; a path or an install makes them available.
FIRST_PARTY_ROOTS: Final[frozenset[str]] = frozenset(
    {"scpn_quantum_control", "oscillatools", "scpn", "tools", "scripts"}
)

#: Matches ``python -m <module>`` in a step body.
MODULE_INVOCATION: Final[re.Pattern[str]] = re.compile(r"python -m ([A-Za-z_][A-Za-z0-9_.]*)")

#: Matches the requirement files a ``pip install`` step names.
REQUIREMENT_INSTALL: Final[re.Pattern[str]] = re.compile(
    r"-r\s+(requirements[A-Za-z0-9._-]*\.txt)"
)

#: Matches the ways a job installs this project rather than a dependency.
#:
#: Three shapes are in use and all three count: an editable install, a wheel the
#: job builds and installs, and a plain ``pip install .``. Recognising only the
#: editable one reported two jobs as broken that are not.
PROJECT_INSTALL: Final[re.Pattern[str]] = re.compile(
    r"pip install[^\n]*(?:-e\s+\S|\.whl|--require-hashes\s+\\?\s*-r\s+/tmp/[^\s]*wheel)",
)

#: Matches a job writing PYTHONPATH into the step environment for later steps.
EXPORTED_PYTHONPATH: Final[re.Pattern[str]] = re.compile(r"PYTHONPATH=[^\n]*GITHUB_ENV")

#: Matches a pinned distribution in a requirement lock, extras included.
#:
#: Five entries in the base lock are written ``name[extra]==version``; a pattern
#: that required ``==`` to follow the name directly missed every one of them and
#: reported 381 failures where there were two.
PINNED_DISTRIBUTION: Final[re.Pattern[str]] = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?==", re.M
)


@dataclass(frozen=True)
class Finding:
    """One way a job's declared environment fails to support its own steps."""

    workflow: str
    job: str
    detail: str

    def __str__(self) -> str:
        """Render the finding as one reviewable line."""
        return f"{self.workflow}::{self.job}: {self.detail}"


def _normalise(name: str) -> str:
    """Return a distribution name in the form PyPI compares."""
    return re.sub(r"[-_.]+", "-", name).lower()


def distributions_in(lock: Path) -> frozenset[str]:
    """Return every distribution a requirement lock pins, normalised."""
    if not lock.is_file():
        return frozenset()
    text = lock.read_text(encoding="utf-8")
    return frozenset(_normalise(match) for match in PINNED_DISTRIBUTION.findall(text))


def _step_bodies(job: dict[str, Any]) -> list[str]:
    """Return the shell body of every step in a job."""
    steps = job.get("steps")
    if not isinstance(steps, list):
        return []
    bodies: list[str] = []
    for step in steps:
        if isinstance(step, dict) and isinstance(step.get("run"), str):
            bodies.append(step["run"])
    return bodies


def _pinned_action_versions(job: dict[str, Any]) -> dict[str, str]:
    """Return the ``version`` input each pinned action in a job requests."""
    steps = job.get("steps")
    if not isinstance(steps, list):
        return {}
    pinned: dict[str, str] = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        uses, inputs = step.get("uses"), step.get("with")
        if not isinstance(uses, str) or not isinstance(inputs, dict):
            continue
        version = inputs.get("version")
        if version is not None:
            pinned[uses.split("@")[0]] = str(version)
    return pinned


def _makes_first_party_importable(job: dict[str, Any], bodies: list[str]) -> bool:
    """Report whether a job puts this repository's own packages on the path.

    A job can do this three ways and all of them count: a ``PYTHONPATH`` in its
    ``env`` block, a ``PYTHONPATH`` it writes into ``$GITHUB_ENV`` from a step,
    or an install of the project itself — editable, from a wheel it builds, or
    from the source tree.
    """
    environment = job.get("env")
    if isinstance(environment, dict) and "PYTHONPATH" in environment:
        return True
    return any(EXPORTED_PYTHONPATH.search(body) or PROJECT_INSTALL.search(body) for body in bodies)


def audit_job(
    workflow: str, name: str, job: dict[str, Any], repo: Path
) -> tuple[list[Finding], dict[str, str]]:
    """Check one job's environment against the commands it runs."""
    findings: list[Finding] = []
    bodies = _step_bodies(job)

    provided: set[str] = set()
    for body in bodies:
        for lock in REQUIREMENT_INSTALL.findall(body):
            provided |= distributions_in(repo / lock)

    for body in bodies:
        for module in MODULE_INVOCATION.findall(body):
            root = module.split(".")[0]
            if root in STDLIB_MODULES or root in FIRST_PARTY_ROOTS:
                continue
            if _normalise(root) not in provided:
                findings.append(
                    Finding(
                        workflow,
                        name,
                        f"runs `python -m {module}` but installs no requirement file "
                        f"pinning {root!r}",
                    )
                )

    runs_repository_tests = any("pytest" in body and "tests/" in body for body in bodies)
    if runs_repository_tests and not _makes_first_party_importable(job, bodies):
        findings.append(
            Finding(
                workflow,
                name,
                "runs pytest over tests/ without PYTHONPATH or an editable install, "
                "so tests/conftest.py cannot import the package",
            )
        )

    return findings, _pinned_action_versions(job)


def audit(repo: Path) -> list[Finding]:
    """Check every workflow job and report the contracts that do not hold."""
    findings: list[Finding] = []
    pins: dict[str, dict[str, set[str]]] = {}

    for path in sorted((repo / ".github" / "workflows").glob("*.yml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        jobs = document.get("jobs") if isinstance(document, dict) else None
        if not isinstance(jobs, dict):
            continue
        for name, job in jobs.items():
            if not isinstance(job, dict):
                continue
            job_findings, job_pins = audit_job(path.name, str(name), job, repo)
            findings.extend(job_findings)
            for action, version in job_pins.items():
                pins.setdefault(action, {}).setdefault(version, set()).add(path.name)

    for action, versions in sorted(pins.items()):
        if len(versions) > 1:
            spread = "; ".join(
                f"{version} in {', '.join(sorted(files))}"
                for version, files in sorted(versions.items())
            )
            findings.append(
                Finding("<cross-workflow>", action, f"pinned to more than one version: {spread}")
            )

    return findings


def main(argv: list[str] | None = None) -> int:
    """Run the gate and return a process exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    arguments = parser.parse_args(argv)

    findings = audit(arguments.repo)
    if findings:
        print(f"{len(findings)} workflow environment contract failure(s):")
        for finding in findings:
            print(f"    {finding}")
        return 1
    print("Workflow environment contracts hold for every job")
    return 0


if __name__ == "__main__":
    sys.exit(main())
