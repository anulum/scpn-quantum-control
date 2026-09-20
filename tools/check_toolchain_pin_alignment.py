#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — toolchain pin alignment check
"""Pre-commit hook: keep every declared toolchain pin in one generation.

A quality tool that runs both as a pre-commit hook and from the hash-locked CI
requirements must be the same version in both places. When the two drift, a
locally clean commit can fail CI, or a CI-clean commit can fail the hook, and
the agreement between them becomes luck rather than contract.

The check reads the live configuration rather than a frozen allow-list. Every
remote hook repository must be classified as either a Python distribution whose
pin is compared, or an explicitly reasoned non-Python hook. A hook repository
that is neither is refused, so a tool added later cannot enter silently.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml

ROOT: Final[Path] = Path(__file__).resolve().parent.parent

PRE_COMMIT_CONFIG: Final[str] = ".pre-commit-config.yaml"
"""Configuration owning the hook revisions."""

REQUIREMENT_GLOB: Final[str] = "requirements*.txt"
"""Requirement files whose pins define the CI toolchain generation."""

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

_REQUIREMENT_PIN = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s\\;]+)")


@dataclass(frozen=True)
class HookPin:
    """One remote pre-commit hook repository and its pinned revision.

    Parameters
    ----------
    repo
        Hook repository URL exactly as the configuration declares it.
    rev
        Revision the configuration pins, such as ``v0.16.4``.

    """

    repo: str
    rev: str

    @property
    def version(self) -> str:
        """Return the revision without a leading ``v`` release-tag marker."""
        return self.rev[1:] if self.rev.startswith("v") else self.rev


@dataclass(frozen=True)
class AlignmentFinding:
    """One refusal describing a toolchain generation that does not agree.

    Parameters
    ----------
    kind
        Short machine-readable category of the disagreement.
    detail
        Specific reason naming the files and versions involved.

    """

    kind: str
    detail: str

    def __post_init__(self) -> None:
        """Refuse a finding that does not explain itself."""
        if not self.detail.strip():
            raise ValueError(f"{self.kind}: detail must explain the finding")


def load_hook_pins(config_path: Path) -> tuple[HookPin, ...]:
    """Read every remote hook repository and its pinned revision.

    Parameters
    ----------
    config_path
        Path of the pre-commit configuration.

    Returns
    -------
    tuple of HookPin
        Remote hook pins in configuration order. In-repository hooks are
        excluded because they carry no revision.

    Raises
    ------
    ValueError
        If the configuration has no ``repos`` list, or a remote repository
        declares no revision.

    """
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    repos = document.get("repos") if isinstance(document, dict) else None
    if not isinstance(repos, list) or not repos:
        raise ValueError(f"{PRE_COMMIT_CONFIG} declares no hook repositories")
    pins: list[HookPin] = []
    for entry in repos:
        repo = str(entry.get("repo", ""))
        if repo == LOCAL_HOOK_REPO:
            continue
        rev = entry.get("rev")
        if not isinstance(rev, str) or not rev.strip():
            raise ValueError(f"{repo}: remote hook repository declares no rev")
        pins.append(HookPin(repo=repo, rev=rev.strip()))
    if not pins:
        raise ValueError(f"{PRE_COMMIT_CONFIG} declares no remote hook repository")
    return tuple(pins)


def load_requirement_pins(paths: Sequence[Path]) -> dict[str, dict[str, str]]:
    """Read the pinned version of every distribution in each requirement file.

    Parameters
    ----------
    paths
        Requirement files to read.

    Returns
    -------
    dict
        Mapping of distribution name to ``{file name: version}``. Names are
        lowercased and ``_`` is normalised to ``-`` so a pin is found whichever
        spelling the file uses.

    """
    pins: dict[str, dict[str, str]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            match = _REQUIREMENT_PIN.match(line)
            if match is None:
                continue
            name = match.group("name").lower().replace("_", "-")
            pins.setdefault(name, {})[path.name] = match.group("version")
    return pins


def check_toolchain_pin_alignment(source_root: Path) -> tuple[AlignmentFinding, ...]:
    """Compare hook revisions with the pinned CI requirement generation.

    Parameters
    ----------
    source_root
        Repository checkout owning the configuration and requirement files.

    Returns
    -------
    tuple of AlignmentFinding
        Every disagreement found, in report order. Empty means the declared
        toolchain is one generation everywhere.

    Raises
    ------
    ValueError
        If the configuration cannot be read, or no requirement file exists.

    """
    config_path = source_root / PRE_COMMIT_CONFIG
    if not config_path.is_file():
        raise ValueError(f"missing {PRE_COMMIT_CONFIG} in {source_root}")
    requirement_paths = sorted(source_root.glob(REQUIREMENT_GLOB))
    if not requirement_paths:
        raise ValueError(f"no {REQUIREMENT_GLOB} file in {source_root}")

    hook_pins = load_hook_pins(config_path)
    requirement_pins = load_requirement_pins(requirement_paths)
    findings: list[AlignmentFinding] = []

    for pin in hook_pins:
        distribution = HOOK_DISTRIBUTIONS.get(pin.repo)
        if distribution is None:
            if pin.repo in NON_PYTHON_HOOK_REPOS:
                continue
            findings.append(
                AlignmentFinding(
                    kind="unclassified_hook_repo",
                    detail=(
                        f"{pin.repo} is neither mapped to a pinned distribution nor recorded "
                        "as a non-Python hook; classify it before it can pass"
                    ),
                )
            )
            continue
        declared = requirement_pins.get(distribution)
        if not declared:
            findings.append(
                AlignmentFinding(
                    kind="hook_without_requirement_pin",
                    detail=(
                        f"{pin.repo} mirrors {distribution} but no {REQUIREMENT_GLOB} file "
                        "pins that distribution"
                    ),
                )
            )
            continue
        versions = sorted(set(declared.values()))
        if len(versions) > 1:
            locations = ", ".join(
                f"{name}={version}" for name, version in sorted(declared.items())
            )
            findings.append(
                AlignmentFinding(
                    kind="requirement_version_split",
                    detail=f"{distribution} is pinned to several versions: {locations}",
                )
            )
            continue
        if pin.version != versions[0]:
            findings.append(
                AlignmentFinding(
                    kind="hook_generation_mismatch",
                    detail=(
                        f"{distribution}: {PRE_COMMIT_CONFIG} pins {pin.rev} while "
                        f"{', '.join(sorted(declared))} pin {versions[0]}"
                    ),
                )
            )
    return tuple(findings)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the toolchain pin alignment check over a checkout.

    Parameters
    ----------
    argv
        Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` when every declared pin agrees, ``1`` when a disagreement is
        found, and ``2`` when the configuration cannot be read.

    """
    parser = argparse.ArgumentParser(
        prog="check-toolchain-pin-alignment",
        description=(
            "Refuse a quality-tool pin that disagrees between the pre-commit "
            "configuration and the hash-locked CI requirements."
        ),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ROOT,
        help="Repository checkout to inspect.",
    )
    args = parser.parse_args(argv)
    try:
        findings = check_toolchain_pin_alignment(args.source_root)
    except ValueError as error:
        print(f"toolchain pin evidence unavailable: {error}", file=sys.stderr)
        return 2
    for finding in findings:
        print(f"{finding.kind}: {finding.detail}", file=sys.stderr)
    if findings:
        return 1
    print("toolchain pin alignment: OK (hook revisions match the pinned requirements)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
