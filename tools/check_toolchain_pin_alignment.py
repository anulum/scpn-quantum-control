#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — toolchain declaration alignment check
"""Pre-commit hook: keep every declaration of a tool in one generation.

Where a repository names the same tool more than once, those declarations are
one set with one answer. A pre-commit hook revision, a hash-locked requirements
pin, a ``pyproject.toml`` dependency range, a workflow action input, a pinned
``cargo install`` and a documented setup command that all name the same tool
must agree, or a local run predicts a verdict the gate will not give.

The check reads the live repository rather than a frozen allow-list. Every
remote hook repository must be classified as either a mirrored Python
distribution whose pin is compared, or an explicitly reasoned non-Python hook,
so a tool added later cannot enter silently. Declarations that carry no version
are not inspected: a ``language: system`` hook runs whatever the environment
installed and cannot split from a pin.

This is an internal-consistency gate. Whether the agreed version is also the
newest compatible release is a separate, network-bound currency question that
belongs to the periodic dependency census, not to a commit hook.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version

ROOT: Final[Path] = Path(__file__).resolve().parent.parent

PRE_COMMIT_CONFIG: Final[str] = ".pre-commit-config.yaml"
"""Configuration owning the hook revisions."""

PYPROJECT: Final[str] = "pyproject.toml"
"""Project metadata owning dependency ranges."""

REQUIREMENT_GLOB: Final[str] = "requirements*.txt"
"""Requirement files whose pins define the installed CI toolchain."""

WORKFLOW_GLOB: Final[str] = ".github/workflows/*.yml"
"""Workflow files whose inputs and install commands pin tools."""

DOCUMENTED_COMMAND_FILES: Final[tuple[str, ...]] = ("CONTRIBUTING.md",)
"""Documents whose setup commands pin a tool version for contributors."""

LOCAL_HOOK_REPO: Final[str] = "local"
"""Repository marker for in-repository hooks, which carry no revision."""

HOOK_DISTRIBUTIONS: Final[dict[str, str]] = {
    "https://github.com/astral-sh/ruff-pre-commit": "ruff",
    "https://github.com/pre-commit/mirrors-mypy": "mypy",
}
"""Hook repositories that mirror a PyPI distribution pinned in requirements."""

NON_PYTHON_HOOK_REPOS: Final[dict[str, str]] = {
    "https://github.com/gitleaks/gitleaks": (
        "upstream Go binary released outside PyPI; no requirements pin exists to compare"
    ),
}
"""Hook repositories deliberately outside the Python requirement generation."""

ACTION_TOOLS: Final[dict[str, str]] = {
    "pnpm/action-setup": "pnpm",
}
"""Workflow actions whose ``version`` input pins a tool the repository also names."""

_REQUIREMENT_PIN = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s\\;]+)")
_CARGO_INSTALL = re.compile(
    r"cargo install\s+(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"(?:\s+--[A-Za-z-]+)*\s+--version\s+(?P<version>[0-9][^\s`'\"]*)"
)
_COREPACK_PREPARE = re.compile(
    r"corepack prepare\s+(?P<name>[A-Za-z0-9@/._-]+)@(?P<version>[0-9][^\s`'\"]*)"
)


def normalise(name: str) -> str:
    """Return the comparison form of a distribution or tool name.

    Parameters
    ----------
    name
        Name as a declaration spells it.

    Returns
    -------
    str
        Lowercased name with ``_`` folded to ``-``.

    """
    return name.strip().lower().replace("_", "-")


@dataclass(frozen=True)
class ToolDeclaration:
    """One place where the repository states a version for a tool.

    Parameters
    ----------
    tool
        Normalised tool name.
    version
        Exact version the declaration names, or an empty string when the
        declaration is a range rather than a pin.
    specifier
        Raw text of the declaration, kept for range comparison and reporting.
    source
        Repository-relative file the declaration was read from.
    kind
        Declaration class, such as ``pre_commit_rev`` or ``requirement_pin``.

    """

    tool: str
    version: str
    specifier: str
    source: str
    kind: str

    @property
    def is_pin(self) -> bool:
        """Return whether this declaration names one exact version."""
        return bool(self.version)

    def describe(self) -> str:
        """Return a short ``source(kind)=value`` description for a report."""
        value = self.version if self.is_pin else self.specifier
        return f"{self.source}({self.kind})={value}"


@dataclass(frozen=True)
class AlignmentFinding:
    """One refusal describing declarations that do not agree.

    Parameters
    ----------
    kind
        Short machine-readable category of the disagreement.
    detail
        Specific reason naming the declarations and versions involved.

    """

    kind: str
    detail: str

    def __post_init__(self) -> None:
        """Refuse a finding that does not explain itself."""
        if not self.detail.strip():
            raise ValueError(f"{self.kind}: detail must explain the finding")


def collect_declarations(source_root: Path) -> tuple[ToolDeclaration, ...]:
    """Read every version declaration the repository makes for a tracked tool.

    Parameters
    ----------
    source_root
        Repository checkout to inspect.

    Returns
    -------
    tuple of ToolDeclaration
        Declarations grouped by collector, in source order.

    Raises
    ------
    ValueError
        If the pre-commit configuration is missing or unreadable, or no
        requirement file exists.

    """
    config_path = source_root / PRE_COMMIT_CONFIG
    if not config_path.is_file():
        raise ValueError(f"missing {PRE_COMMIT_CONFIG} in {source_root}")
    requirement_paths = sorted(source_root.glob(REQUIREMENT_GLOB))
    if not requirement_paths:
        raise ValueError(f"no {REQUIREMENT_GLOB} file in {source_root}")

    declarations: list[ToolDeclaration] = []
    declarations.extend(_collect_pre_commit(config_path))
    declarations.extend(_collect_requirements(requirement_paths, source_root))
    declarations.extend(_collect_pyproject(source_root))
    declarations.extend(_collect_workflows(source_root))
    declarations.extend(_collect_documented_commands(source_root))
    return tuple(declarations)


def check_toolchain_pin_alignment(source_root: Path) -> tuple[AlignmentFinding, ...]:
    """Compare every declaration of each tool against the others.

    Parameters
    ----------
    source_root
        Repository checkout owning the declarations.

    Returns
    -------
    tuple of AlignmentFinding
        Every disagreement found, unclassified hooks first and then by tool
        name. Empty means the declared toolchain is one generation everywhere.

    Raises
    ------
    ValueError
        If the declarations cannot be read.

    """
    declarations = collect_declarations(source_root)
    findings: list[AlignmentFinding] = _unclassified_hook_findings(source_root)

    by_tool: dict[str, list[ToolDeclaration]] = {}
    for declaration in declarations:
        by_tool.setdefault(declaration.tool, []).append(declaration)

    for tool in sorted(by_tool):
        findings.extend(_tool_findings(tool, by_tool[tool]))
    return tuple(findings)


def _tool_findings(tool: str, declarations: list[ToolDeclaration]) -> list[AlignmentFinding]:
    """Return every disagreement among one tool's declarations."""
    findings: list[AlignmentFinding] = []
    pins = [declaration for declaration in declarations if declaration.is_pin]
    ranges = [declaration for declaration in declarations if not declaration.is_pin]

    hook_pins = [pin for pin in pins if pin.kind == "pre_commit_rev"]
    requirement_pins = [pin for pin in pins if pin.kind == "requirement_pin"]
    if hook_pins and not requirement_pins:
        findings.append(
            AlignmentFinding(
                kind="hook_without_requirement_pin",
                detail=(
                    f"{tool}: {PRE_COMMIT_CONFIG} pins a revision but no "
                    f"{REQUIREMENT_GLOB} file pins the distribution to compare it with"
                ),
            )
        )

    versions = sorted({pin.version for pin in pins})
    if len(versions) > 1:
        locations = ", ".join(sorted(pin.describe() for pin in pins))
        findings.append(
            AlignmentFinding(
                kind="version_disagreement",
                detail=f"{tool} is declared in several versions: {locations}",
            )
        )
        return findings

    if versions and ranges:
        findings.extend(_range_findings(tool, versions[0], ranges))
    return findings


def _range_findings(
    tool: str, agreed: str, ranges: list[ToolDeclaration]
) -> list[AlignmentFinding]:
    """Return findings for ranges the agreed pinned version does not satisfy."""
    try:
        version = Version(agreed)
    except InvalidVersion:
        return [
            AlignmentFinding(
                kind="unparsable_version",
                detail=f"{tool}: pinned version {agreed!r} is not a valid version",
            )
        ]
    findings: list[AlignmentFinding] = []
    for declaration in ranges:
        requirement = Requirement(declaration.specifier)
        if not requirement.specifier.contains(version, prereleases=True):
            findings.append(
                AlignmentFinding(
                    kind="range_violation",
                    detail=(
                        f"{tool}: pinned {agreed} does not satisfy "
                        f"{declaration.source}({declaration.kind})="
                        f"{declaration.specifier!r}"
                    ),
                )
            )
    return findings


def _unclassified_hook_findings(source_root: Path) -> list[AlignmentFinding]:
    """Return a finding for every remote hook repository left unclassified."""
    findings: list[AlignmentFinding] = []
    for repo, _rev in _pre_commit_repos(source_root / PRE_COMMIT_CONFIG):
        if repo in HOOK_DISTRIBUTIONS or repo in NON_PYTHON_HOOK_REPOS:
            continue
        findings.append(
            AlignmentFinding(
                kind="unclassified_hook_repo",
                detail=(
                    f"{repo} is neither mapped to a pinned distribution nor recorded "
                    "as a non-Python hook; classify it before it can pass"
                ),
            )
        )
    return findings


def _pre_commit_repos(config_path: Path) -> list[tuple[str, str]]:
    """Return every remote hook repository and its pinned revision."""
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    repos = document.get("repos") if isinstance(document, dict) else None
    if not isinstance(repos, list) or not repos:
        raise ValueError(f"{PRE_COMMIT_CONFIG} declares no hook repositories")
    remote: list[tuple[str, str]] = []
    for entry in repos:
        repo = str(entry.get("repo", ""))
        if repo == LOCAL_HOOK_REPO:
            continue
        rev = entry.get("rev")
        if not isinstance(rev, str) or not rev.strip():
            raise ValueError(f"{repo}: remote hook repository declares no rev")
        remote.append((repo, rev.strip()))
    if not remote:
        raise ValueError(f"{PRE_COMMIT_CONFIG} declares no remote hook repository")
    return remote


def _collect_pre_commit(config_path: Path) -> list[ToolDeclaration]:
    """Return one declaration per classified remote hook repository."""
    declarations: list[ToolDeclaration] = []
    for repo, rev in _pre_commit_repos(config_path):
        distribution = HOOK_DISTRIBUTIONS.get(repo)
        if distribution is None:
            continue
        declarations.append(
            ToolDeclaration(
                tool=normalise(distribution),
                version=rev[1:] if rev.startswith("v") else rev,
                specifier=rev,
                source=PRE_COMMIT_CONFIG,
                kind="pre_commit_rev",
            )
        )
    return declarations


def _collect_requirements(paths: Iterable[Path], source_root: Path) -> list[ToolDeclaration]:
    """Return one declaration per tracked exact pin in the requirement files."""
    tracked = {normalise(name) for name in HOOK_DISTRIBUTIONS.values()}
    declarations: list[ToolDeclaration] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            match = _REQUIREMENT_PIN.match(line)
            if match is None:
                continue
            tool = normalise(match.group("name"))
            if tool not in tracked:
                continue
            declarations.append(
                ToolDeclaration(
                    tool=tool,
                    version=match.group("version"),
                    specifier=line.strip().rstrip("\\").strip(),
                    source=path.relative_to(source_root).as_posix(),
                    kind="requirement_pin",
                )
            )
    return declarations


def _collect_pyproject(source_root: Path) -> list[ToolDeclaration]:
    """Return one declaration per tracked dependency range in the project metadata."""
    path = source_root / PYPROJECT
    if not path.is_file():
        return []
    tracked = {normalise(name) for name in HOOK_DISTRIBUTIONS.values()}
    project = tomllib.loads(path.read_text(encoding="utf-8")).get("project")
    if not isinstance(project, dict):
        return []
    groups: list[list[str]] = []
    dependencies = project.get("dependencies")
    if isinstance(dependencies, list):
        groups.append([str(item) for item in dependencies])
    optional = project.get("optional-dependencies")
    if isinstance(optional, dict):
        groups.extend([str(item) for item in value] for value in optional.values())

    declarations: list[ToolDeclaration] = []
    for group in groups:
        for specifier in group:
            try:
                requirement = Requirement(specifier)
            except InvalidRequirement:
                continue
            tool = normalise(requirement.name)
            if tool not in tracked:
                continue
            declarations.append(
                ToolDeclaration(
                    tool=tool,
                    version="",
                    specifier=specifier,
                    source=PYPROJECT,
                    kind="pyproject_range",
                )
            )
    return declarations


def _collect_workflows(source_root: Path) -> list[ToolDeclaration]:
    """Return declarations from workflow action inputs and pinned installs."""
    declarations: list[ToolDeclaration] = []
    for path in sorted(source_root.glob(WORKFLOW_GLOB)):
        relative = path.relative_to(source_root).as_posix()
        text = path.read_text(encoding="utf-8")
        declarations.extend(_workflow_action_inputs(text, relative))
        for match in _CARGO_INSTALL.finditer(text):
            declarations.append(
                ToolDeclaration(
                    tool=normalise(match.group("name")),
                    version=match.group("version"),
                    specifier=match.group(0),
                    source=relative,
                    kind="workflow_install",
                )
            )
    return declarations


def _workflow_action_inputs(text: str, relative: str) -> list[ToolDeclaration]:
    """Return declarations from ``uses`` plus ``version`` action input pairs."""
    declarations: list[ToolDeclaration] = []
    pending: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if "uses:" in stripped and stripped.split("uses:", 1)[0].strip(" -") == "":
            reference = stripped.split("uses:", 1)[1].strip().split("@", 1)[0]
            pending = ACTION_TOOLS.get(reference)
            continue
        if pending is None:
            continue
        if stripped.startswith("version:"):
            declarations.append(
                ToolDeclaration(
                    tool=normalise(pending),
                    version=stripped.split("version:", 1)[1].strip().strip("\"'"),
                    specifier=stripped,
                    source=relative,
                    kind="workflow_action_input",
                )
            )
            pending = None
        elif stripped.startswith("- "):
            pending = None
    return declarations


def _collect_documented_commands(source_root: Path) -> list[ToolDeclaration]:
    """Return declarations from pinned setup commands in contributor documents."""
    declarations: list[ToolDeclaration] = []
    for name in DOCUMENTED_COMMAND_FILES:
        path = source_root / name
        if not path.is_file():
            continue
        for match in _COREPACK_PREPARE.finditer(path.read_text(encoding="utf-8")):
            declarations.append(
                ToolDeclaration(
                    tool=normalise(match.group("name")),
                    version=match.group("version"),
                    specifier=match.group(0),
                    source=name,
                    kind="documented_command",
                )
            )
    return declarations


def main(argv: Sequence[str] | None = None) -> int:
    """Run the toolchain declaration alignment check over a checkout.

    Parameters
    ----------
    argv
        Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` when every declaration agrees, ``1`` when a disagreement is
        found, and ``2`` when the declarations cannot be read.

    """
    parser = argparse.ArgumentParser(
        prog="check-toolchain-pin-alignment",
        description=(
            "Refuse a tool version that disagrees between any two declarations "
            "in this repository: hook revisions, hash-locked requirements, "
            "project ranges, workflow inputs, pinned installs and documented "
            "setup commands."
        ),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ROOT,
        help="Repository checkout to inspect.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print every collected declaration instead of only the findings.",
    )
    args = parser.parse_args(argv)
    try:
        declarations = collect_declarations(args.source_root)
        findings = check_toolchain_pin_alignment(args.source_root)
    except ValueError as error:
        print(f"toolchain declaration evidence unavailable: {error}", file=sys.stderr)
        return 2
    if args.list:
        for declaration in sorted(declarations, key=lambda row: (row.tool, row.source, row.kind)):
            print(f"{declaration.tool}: {declaration.describe()}")
    for finding in findings:
        print(f"{finding.kind}: {finding.detail}", file=sys.stderr)
    if findings:
        return 1
    tools = len({declaration.tool for declaration in declarations})
    print(
        f"toolchain declaration alignment: OK "
        f"({len(declarations)} declarations across {tools} tools agree)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
