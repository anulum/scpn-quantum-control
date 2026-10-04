# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — source surface inventory gate
"""Fail when a tracked source language or source root has no recorded owner.

Every lint, type and documentation gate names the paths it reads. A file in a
new language, or in a new top-level directory, is read by none of them and no
gate turns red. This gate closes that hole from the other side: it classifies
every tracked file by kind (its suffix, or its name when it has none) and
top-level root, and compares the result with a reviewed inventory.

Each inventory row covers one kind under one root and is in exactly one state:

* *gated*: it names at least one CI job and a command of that job, and nothing
  is recorded as missing;
* *open*: it records what is missing. An open row is visible debt, and with a
  comparison revision no row may be open that was not open there;
* *evidence*: files recorded together with results under ``data``, kept as
  they ran and not maintained.

File kinds that carry no source (documents, data, configuration, binaries) are
listed once as *outside* the inventory, each with its class.

A gate reference is checked against the workflow files: the workflow and job
must exist and the recorded command must occur in one of the job's ``run``
steps. The check is textual. It proves the command is still written there; it
does not prove the job passes or that the command reads every file of the row.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml

SCHEMA = "source_surface_inventory_v1"
DEFAULT_POLICY = Path("tools/source_surface_policy.json")
WORKFLOWS = Path(".github/workflows")
EVIDENCE_ROOT = "data"
TOP_LEVEL = "."


@dataclass(frozen=True)
class Gate:
    """One CI command recorded as reading a surface.

    Attributes
    ----------
    workflow
        File name of the workflow under ``.github/workflows``.
    job
        Job key inside that workflow.
    command
        Text that must occur in one of the job's ``run`` steps.

    """

    workflow: str
    job: str
    command: str


@dataclass(frozen=True)
class Surface:
    """One file kind under one top-level root.

    Attributes
    ----------
    kind
        Lower-case file suffix including the dot, or the file name when the
        file has no suffix.
    root
        First path component, or ``"."`` for files at the repository root.
    gates
        CI commands recorded as reading the surface.
    missing
        What is not enforced yet; empty when nothing is recorded as missing.
    evidence
        Why the files are recorded evidence; empty for maintained source.

    """

    kind: str
    root: str
    gates: tuple[Gate, ...]
    missing: str
    evidence: str

    @property
    def state(self) -> str:
        """Return ``"open"``, ``"evidence"`` or ``"gated"``."""
        if self.missing:
            return "open"
        if self.evidence:
            return "evidence"
        return "gated"

    @property
    def label(self) -> str:
        """Return the row's name as used in messages."""
        return f"{self.kind} under {self.root}"


@dataclass(frozen=True)
class Policy:
    """The reviewed inventory.

    Attributes
    ----------
    surfaces
        Inventory rows keyed by ``(kind, root)``.
    outside
        File kinds that carry no source, mapped to their class.

    """

    surfaces: dict[tuple[str, str], Surface]
    outside: dict[str, str]


def _text(row: dict[str, object], key: str, where: str, *, required: bool) -> str:
    """Return the string stored under ``key`` of ``row``.

    Raises
    ------
    ValueError
        If the value is not a string, or is absent or blank while required.

    """
    value = row.get(key, "")
    if not isinstance(value, str):
        raise ValueError(f"{where}: {key} must be a string")
    if required and not value.strip():
        raise ValueError(f"{where}: {key} must be a non-empty string")
    return value.strip()


def _gate(raw: object, where: str) -> Gate:
    """Build a gate reference from its JSON form.

    Raises
    ------
    ValueError
        If the entry is not an object with the three required strings.

    """
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: a gate must be an object")
    return Gate(
        workflow=_text(raw, "workflow", where, required=True),
        job=_text(raw, "job", where, required=True),
        command=_text(raw, "command", where, required=True),
    )


def _surface(raw: object, position: int) -> Surface:
    """Build an inventory row from its JSON form.

    Raises
    ------
    ValueError
        If the row is malformed or is not in exactly one state.

    """
    where = f"surface {position}"
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: a surface must be an object")
    kind = _text(raw, "kind", where, required=True)
    root = _text(raw, "root", where, required=True)
    where = f"{kind} under {root}"
    gates = raw.get("gates", [])
    if not isinstance(gates, list):
        raise ValueError(f"{where}: gates must be a list")
    surface = Surface(
        kind=kind,
        root=root,
        gates=tuple(_gate(gate, where) for gate in gates),
        missing=_text(raw, "missing", where, required=False),
        evidence=_text(raw, "evidence", where, required=False),
    )
    if surface.evidence and (surface.gates or surface.missing):
        raise ValueError(f"{where}: an evidence row records neither gates nor missing work")
    if surface.evidence and root != EVIDENCE_ROOT:
        raise ValueError(f"{where}: evidence rows are allowed under {EVIDENCE_ROOT} only")
    if not (surface.gates or surface.missing or surface.evidence):
        raise ValueError(f"{where}: a row needs a gate, missing work or an evidence reason")
    return surface


def parse_policy(text: str) -> Policy:
    """Parse the inventory from its JSON text.

    Parameters
    ----------
    text
        Content of the inventory file.

    Returns
    -------
    Policy
        The validated inventory.

    Raises
    ------
    ValueError
        If the text is not valid JSON, has another schema, repeats a row, or
        lists a kind both as a surface and as outside the inventory.

    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"inventory is not valid JSON: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError(f"inventory schema must be {SCHEMA}")
    rows = payload.get("surfaces")
    outside = payload.get("outside")
    if not isinstance(rows, list) or not isinstance(outside, dict):
        raise ValueError("inventory needs a surfaces list and an outside object")
    surfaces: dict[tuple[str, str], Surface] = {}
    for position, raw in enumerate(rows):
        surface = _surface(raw, position)
        if (surface.kind, surface.root) in surfaces:
            raise ValueError(f"{surface.label}: repeated row")
        surfaces[surface.kind, surface.root] = surface
    classes: dict[str, str] = {}
    for kind, label in outside.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"outside kind {kind}: the class must be a non-empty string")
        classes[kind] = label.strip()
    both = sorted({kind for kind, _root in surfaces} & classes.keys())
    if both:
        raise ValueError(f"kind listed as a surface and as outside: {', '.join(both)}")
    return Policy(surfaces=surfaces, outside=classes)


def load_policy(path: Path) -> Policy:
    """Read and parse the inventory file.

    Raises
    ------
    ValueError
        If the file cannot be read or does not hold a valid inventory.

    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot read the inventory: {error}") from error
    return parse_policy(text)


def _git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run Git in ``repo`` and return the finished process.

    Raises
    ------
    ValueError
        If Git cannot be started.

    """
    try:
        return subprocess.run(  # noqa: S603 - fixed argument vectors, no shell
            ["git", *arguments],  # noqa: S607 - Git is resolved from PATH like every repository tool
            capture_output=True,
            text=True,
            cwd=repo,
            check=False,
        )
    except OSError as error:
        raise ValueError(f"cannot run git: {error}") from error


def tracked_files(repo: Path) -> list[str]:
    """Return the paths Git tracks in ``repo``.

    Raises
    ------
    ValueError
        If Git cannot list the files.

    """
    completed = _git(repo, "ls-files", "-z")
    if completed.returncode != 0:
        raise ValueError(f"git ls-files failed: {completed.stderr.strip()}")
    return [name for name in completed.stdout.split("\0") if name]


def classify(name: str) -> tuple[str, str]:
    """Return the kind and top-level root of a tracked path.

    Parameters
    ----------
    name
        Path relative to the repository root, with forward slashes.

    Returns
    -------
    tuple[str, str]
        The lower-case suffix (or the file name when there is no suffix) and
        the first path component (``"."`` for a file at the root).

    """
    path = PurePosixPath(name)
    kind = path.suffix.lower() if path.suffix else path.name
    root = path.parts[0] if len(path.parts) > 1 else TOP_LEVEL
    return kind, root


def audit_inventory(names: Sequence[str], policy: Policy) -> list[str]:
    """Compare the tracked files with the inventory.

    Parameters
    ----------
    names
        Tracked paths.
    policy
        The reviewed inventory.

    Returns
    -------
    list[str]
        One message per file kind without a classification, per source root
        without a row, and per row or outside kind that no tracked file uses.

    """
    counts: Counter[tuple[str, str]] = Counter()
    examples: dict[tuple[str, str], str] = {}
    for name in sorted(names):
        key = classify(name)
        counts[key] += 1
        examples.setdefault(key, name)
    source_kinds = {kind for kind, _root in policy.surfaces}
    errors: list[str] = []
    for (kind, root), count in sorted(counts.items()):
        if kind in policy.outside or (kind, root) in policy.surfaces:
            continue
        example = examples[kind, root]
        if kind in source_kinds:
            errors.append(
                f"source root without an owner: {kind} under {root} ({count} files, e.g. {example})"
            )
        else:
            errors.append(
                f"unclassified file kind: {kind} under {root} ({count} files, e.g. {example})"
            )
    for key in sorted(policy.surfaces.keys() - counts.keys()):
        errors.append(f"inventory row has no tracked file left: {policy.surfaces[key].label}")
    present = {kind for kind, _root in counts}
    for kind in sorted(policy.outside.keys() - present):
        errors.append(f"outside kind has no tracked file left: {kind}")
    return errors


def job_commands(repo: Path, workflow: str) -> dict[str, str]:
    """Return the ``run`` text of every job in a workflow file.

    Parameters
    ----------
    repo
        Repository root.
    workflow
        File name under ``.github/workflows``.

    Returns
    -------
    dict[str, str]
        Job key mapped to the job's ``run`` steps joined into one line with
        shell line continuations removed and whitespace collapsed.

    Raises
    ------
    ValueError
        If the file is missing, unreadable or not a workflow mapping. The
        message is a single line.

    """
    path = repo / WORKFLOWS / workflow
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read workflow {workflow}: {_normalise(str(error))}") from error
    jobs = document.get("jobs") if isinstance(document, dict) else None
    if not isinstance(jobs, dict):
        raise ValueError(f"workflow {workflow} has no jobs mapping")
    commands: dict[str, str] = {}
    for job, body in jobs.items():
        steps = body.get("steps") if isinstance(body, dict) else None
        runs = [
            str(step["run"])
            for step in (steps if isinstance(steps, list) else [])
            if isinstance(step, dict) and "run" in step
        ]
        commands[str(job)] = _normalise("\n".join(runs))
    return commands


def _normalise(text: str) -> str:
    """Join shell line continuations and collapse whitespace."""
    return " ".join(text.replace("\\\n", " ").split())


def audit_gates(repo: Path, policy: Policy) -> list[str]:
    """Check every recorded gate against the workflow files.

    Parameters
    ----------
    repo
        Repository root.
    policy
        The reviewed inventory.

    Returns
    -------
    list[str]
        One message per gate whose workflow or job is missing, or whose
        command no ``run`` step of the job contains.

    """
    loaded: dict[str, dict[str, str] | str] = {}
    errors: list[str] = []
    for surface in sorted(policy.surfaces.values(), key=lambda row: (row.kind, row.root)):
        for gate in surface.gates:
            if gate.workflow not in loaded:
                try:
                    loaded[gate.workflow] = job_commands(repo, gate.workflow)
                except ValueError as error:
                    loaded[gate.workflow] = str(error)
            jobs = loaded[gate.workflow]
            if isinstance(jobs, str):
                errors.append(f"{surface.label}: {jobs}")
            elif gate.job not in jobs:
                errors.append(f"{surface.label}: workflow {gate.workflow} has no job {gate.job}")
            elif _normalise(gate.command) not in jobs[gate.job]:
                errors.append(
                    f"{surface.label}: job {gate.job} of {gate.workflow} does not run: {gate.command}"
                )
    return errors


def policy_at(repo: Path, revision: str, path: Path) -> Policy | None:
    """Return the inventory recorded at a revision.

    Parameters
    ----------
    repo
        Repository root.
    revision
        Commit to read from.
    path
        Inventory path relative to the repository root.

    Returns
    -------
    Policy or None
        The inventory at ``revision``; ``None`` when that commit has no
        inventory file, which is the case before the gate was introduced.

    Raises
    ------
    ValueError
        If ``revision`` is not a commit or its inventory is invalid.

    """
    if _git(repo, "rev-parse", "--verify", "--quiet", f"{revision}^{{commit}}").returncode != 0:
        raise ValueError(f"comparison revision is not a commit: {revision}")
    shown = _git(repo, "show", f"{revision}:{path.as_posix()}")
    if shown.returncode != 0:
        return None
    return parse_policy(shown.stdout)


def audit_new_debt(policy: Policy, base: Policy) -> list[str]:
    """Reject rows that are open now and were not open at the base.

    Parameters
    ----------
    policy
        The inventory of the working tree.
    base
        The inventory at the comparison revision.

    Returns
    -------
    list[str]
        One message per open row that the base records as gated, as evidence,
        or not at all.

    """
    errors: list[str] = []
    for key, surface in sorted(policy.surfaces.items()):
        before = base.surfaces.get(key)
        if surface.state == "open" and (before is None or before.state != "open"):
            was = "absent" if before is None else before.state
            errors.append(
                f"surface admitted without full enforcement: {surface.label} (base: {was}); "
                f"missing: {surface.missing}"
            )
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """Compare the tracked files and workflows with the inventory.

    Parameters
    ----------
    argv
        Command-line arguments; ``None`` reads the process arguments.

    Returns
    -------
    int
        Zero when every tracked file is classified, every row is used and
        every recorded gate is found; one otherwise.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--changed-against",
        metavar="REVISION",
        help="also reject rows that are open now and were not open at REVISION",
    )
    args = parser.parse_args(argv)
    repo: Path = args.repo
    try:
        policy = load_policy(repo / args.policy)
        names = tracked_files(repo)
        errors = audit_inventory(names, policy) + audit_gates(repo, policy)
        if args.changed_against is not None:
            base = policy_at(repo, args.changed_against, args.policy)
            if base is not None:
                errors.extend(audit_new_debt(policy, base))
    except ValueError as error:
        print(f"source surface inventory failed: {error}", file=sys.stderr)
        return 1
    for message in errors:
        print(message, file=sys.stderr)
    states = Counter(surface.state for surface in policy.surfaces.values())
    for surface in sorted(policy.surfaces.values(), key=lambda row: (row.kind, row.root)):
        if surface.state == "open":
            print(f"open: {surface.label}: {surface.missing}")
    print(
        f"Source surface inventory: {len(names)} tracked files; {len(policy.surfaces)} surfaces "
        f"({states['gated']} gated, {states['open']} open, {states['evidence']} evidence); "
        f"{len(policy.outside)} kinds outside; {len(errors)} problems"
    )
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
