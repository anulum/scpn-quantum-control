# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — advisory workflow step gate
"""Fail when a workflow tolerates a failure that the advisory policy does not record.

A job or a step that continues after an error, and a shell command whose
failure is discarded, report nothing when they break. Some are deliberate: a
lane that is being staged, a cleanup in an exit trap, a summary that may find
no file. Until now the reason lived in a workflow comment, when it was written
down at all, and nothing noticed a new one.

This gate reads every tracked workflow and finds three kinds of site: a job
with ``continue-on-error``, a step with ``continue-on-error``, and a ``run``
step that contains ``|| true``, ``|| :`` or ``set +e``. Each site must have a
row in ``tools/advisory_workflow_policy.json`` with the reason and exactly one
of three outcomes: ``promotion``, the condition under which the site becomes
blocking; ``permanent``, why it never decides a result; or ``undecided``, what
is still missing for either. An unrecorded site, a row without a site, a
tolerance count that differs from the recorded one and a malformed row fail
the gate. Undecided rows do not fail it; they are printed on every run.

The gate does not judge whether a reason is good, and it does not find other
ways of discarding a failure, such as a command that always exits zero.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml

WORKFLOW_ROOT: Final[str] = ".github/workflows"
DEFAULT_POLICY: Final[Path] = Path("tools/advisory_workflow_policy.json")
SCHEMA: Final[str] = "advisory_workflow_steps_v1"
JOB: Final[str] = "job-continue-on-error"
STEP: Final[str] = "step-continue-on-error"
SHELL: Final[str] = "shell-tolerance"
OUTCOMES: Final[tuple[str, ...]] = ("promotion", "permanent", "undecided")
_TOLERANCE: Final[re.Pattern[str]] = re.compile(r"\|\|\s*(?:true|:)(?![\w-])|\bset \+e\b")

Key = tuple[str, str, str, str]


@dataclass(frozen=True)
class Site:
    """One place where a workflow tolerates a failure."""

    workflow: str
    job: str
    step: str
    kind: str
    count: int

    @property
    def key(self) -> Key:
        """Identity of the site, independent of how often the tolerance occurs."""
        return self.workflow, self.job, self.step, self.kind

    def describe(self) -> str:
        """Return the site as ``workflow / job / step (kind)``."""
        where = f"{self.workflow} / {self.job}"
        return f"{where} / {self.step} ({self.kind})" if self.step else f"{where} ({self.kind})"


@dataclass(frozen=True)
class Row:
    """One recorded site with its reason and its outcome."""

    site: Site
    reason: str
    outcome: str
    statement: str


def _text(row: dict[str, object], field: str) -> str:
    """Return a required non-empty string field of a policy row.

    Raises
    ------
    ValueError
        If the field is absent, not a string or blank.

    """
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"advisory policy row needs a non-empty {field}: {row}")
    return value


def load_policy(path: Path) -> list[Row]:
    """Read and validate the advisory policy.

    Parameters
    ----------
    path
        JSON file with schema ``advisory_workflow_steps_v1``.

    Returns
    -------
    list[Row]
        The recorded sites in file order.

    Raises
    ------
    ValueError
        If the schema, a field, a kind, a count or an outcome is invalid, or a
        site is recorded twice.

    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        raise ValueError("unsupported advisory workflow policy schema")
    sites = raw.get("sites")
    if not isinstance(sites, list) or any(not isinstance(item, dict) for item in sites):
        raise ValueError("advisory policy sites must be a list of objects")
    rows: list[Row] = []
    seen: set[Key] = set()
    for item in sites:
        kind = _text(item, "kind")
        if kind not in (JOB, STEP, SHELL):
            raise ValueError(f"unknown advisory site kind: {kind}")
        step = "" if kind == JOB else _text(item, "step")
        count = item.get("count", 1)
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(f"advisory policy count must be a positive integer: {item}")
        if kind != SHELL and count != 1:
            raise ValueError(f"only a shell tolerance row records a count: {item}")
        outcomes = [name for name in OUTCOMES if name in item]
        if len(outcomes) != 1:
            raise ValueError(
                f"advisory policy row needs exactly one of {', '.join(OUTCOMES)}: {item}"
            )
        site = Site(_text(item, "workflow"), _text(item, "job"), step, kind, count)
        if site.key in seen:
            raise ValueError(f"advisory site is recorded twice: {site.describe()}")
        seen.add(site.key)
        rows.append(Row(site, _text(item, "reason"), outcomes[0], _text(item, outcomes[0])))
    return rows


def tracked_workflows(repo: Path) -> list[str]:
    """Return the tracked workflow files of ``repo`` in sorted order.

    Raises
    ------
    ValueError
        If Git cannot list the tracked files.

    """
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argument vector, no shell
            ["git", "ls-files", "-z", "--", WORKFLOW_ROOT],  # noqa: S607 - Git from PATH
            capture_output=True,
            text=True,
            cwd=repo,
            check=False,
        )
    except OSError as error:
        raise ValueError(f"cannot run git: {error}") from error
    if completed.returncode != 0:
        raise ValueError(f"git ls-files failed: {completed.stderr.strip()}")
    names = (name for name in completed.stdout.split("\0") if name)
    return sorted(name for name in names if name.endswith((".yml", ".yaml")))


def _tolerates(mapping: dict[str, object]) -> bool:
    """Return whether a job or step declares that it continues after an error."""
    return "continue-on-error" in mapping and mapping["continue-on-error"] is not False


def _label(step: dict[str, object]) -> str:
    """Return the name of a step, or the action or first command line it runs."""
    for field in ("name", "uses"):
        value = step.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    lines = str(step.get("run", "")).strip().splitlines()
    return lines[0].strip() if lines else ""


def scan(repo: Path) -> list[Site]:
    """Find every tolerated failure in the tracked workflows.

    Parameters
    ----------
    repo
        Repository root.

    Returns
    -------
    list[Site]
        Sites in workflow, job and step order.

    Raises
    ------
    ValueError
        If a workflow is not a YAML mapping, or a job holds two tolerating
        steps that cannot be told apart by name.

    """
    sites: list[Site] = []
    for name in tracked_workflows(repo):
        document = yaml.safe_load((repo / name).read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError(f"workflow is not a mapping: {name}")
        workflow = Path(name).name
        jobs = document.get("jobs")
        for job, body in (jobs if isinstance(jobs, dict) else {}).items():
            if not isinstance(body, dict):
                continue
            if _tolerates(body):
                sites.append(Site(workflow, str(job), "", JOB, 1))
            steps = body.get("steps")
            for step in steps if isinstance(steps, list) else []:
                if not isinstance(step, dict):
                    continue
                label = _label(step)
                if _tolerates(step):
                    sites.append(Site(workflow, str(job), label, STEP, 1))
                count = len(_TOLERANCE.findall(str(step.get("run", ""))))
                if count:
                    sites.append(Site(workflow, str(job), label, SHELL, count))
    keys = [site.key for site in sites]
    for site in sites:
        if keys.count(site.key) > 1:
            raise ValueError(f"steps cannot be told apart, name them: {site.describe()}")
    return sites


def compare(sites: Sequence[Site], rows: Sequence[Row]) -> list[str]:
    """Return every way the workflows depart from the advisory policy.

    Parameters
    ----------
    sites
        Tolerated failures found in the workflows.
    rows
        Recorded sites.

    Returns
    -------
    list[str]
        One message per problem; empty when every site is recorded exactly.

    """
    recorded = {row.site.key: row.site for row in rows}
    found = {site.key: site for site in sites}
    errors: list[str] = []
    for key, site in found.items():
        row = recorded.get(key)
        if row is None:
            errors.append(f"tolerated failure outside the advisory policy: {site.describe()}")
        elif row.count != site.count:
            errors.append(
                f"tolerance count changed: {site.describe()}: recorded {row.count}, found {site.count}"
            )
    for key, site in recorded.items():
        if key not in found:
            errors.append(f"advisory policy row has no site: {site.describe()}")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """Compare the tolerated failures of the workflows with the advisory policy.

    Parameters
    ----------
    argv
        Command-line arguments; ``None`` reads the process arguments.

    Returns
    -------
    int
        Zero when every site is recorded exactly, one otherwise.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    args = parser.parse_args(argv)
    repo: Path = args.repo
    try:
        rows = load_policy(repo / args.policy)
        sites = scan(repo)
        errors = compare(sites, rows)
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"advisory workflow step gate failed: {error}", file=sys.stderr)
        return 1
    for message in errors:
        print(message, file=sys.stderr)
    for row in rows:
        if row.outcome == "undecided":
            print(f"undecided: {row.site.describe()}: {row.statement}")
    counts = {name: sum(row.outcome == name for row in rows) for name in OUTCOMES}
    print(
        f"Advisory workflow steps: {len(sites)} sites; {len(rows)} recorded "
        f"({counts['promotion']} staged, {counts['permanent']} permanent, "
        f"{counts['undecided']} undecided); {len(errors)} problems"
    )
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
