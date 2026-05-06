# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — dependency drift checker
"""Check that requirements.txt mirrors pyproject runtime dependencies."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from packaging.requirements import Requirement


@dataclass(frozen=True)
class DependencyDriftReport:
    """Comparison between canonical pyproject dependencies and requirements."""

    project_dependencies: tuple[str, ...]
    runtime_requirements: tuple[str, ...]

    @property
    def in_sync(self) -> bool:
        """Return True when requirements pin the declared runtime surface."""
        return (
            not self.missing_from_requirements
            and not self.extra_in_requirements
            and not self.order_mismatch
            and not self.unsatisfied_pins
        )

    @property
    def missing_from_requirements(self) -> tuple[str, ...]:
        """Dependencies present in pyproject but absent from requirements."""
        requirement_names = _requirement_names(self.runtime_requirements)
        return tuple(
            dep
            for dep in self.project_dependencies
            if _requirement_name(dep) not in requirement_names
        )

    @property
    def extra_in_requirements(self) -> tuple[str, ...]:
        """Requirements present in requirements.txt but absent from pyproject."""
        project_names = _requirement_names(self.project_dependencies)
        return tuple(
            req for req in self.runtime_requirements if _requirement_name(req) not in project_names
        )

    @property
    def order_mismatch(self) -> bool:
        """Return True when the dependency names match but order differs."""
        return (
            _requirement_names(self.project_dependencies)
            != _requirement_names(self.runtime_requirements)
            and not self.missing_from_requirements
            and not self.extra_in_requirements
        )

    @property
    def unsatisfied_pins(self) -> tuple[str, ...]:
        """Pinned runtime requirements outside the declared pyproject ranges."""
        if self.missing_from_requirements or self.extra_in_requirements:
            return ()
        projects = {_requirement_name(dep): Requirement(dep) for dep in self.project_dependencies}
        unsatisfied: list[str] = []
        for requirement_text in self.runtime_requirements:
            requirement = Requirement(requirement_text)
            project = projects[requirement.name.lower()]
            if not _is_exact_pin(requirement) or not _pin_satisfies(requirement, project):
                unsatisfied.append(requirement_text)
        return tuple(unsatisfied)


def _requirement_name(requirement_text: str) -> str:
    """Return the canonical package name used by the dependency checker."""
    return Requirement(requirement_text).name.lower()


def _requirement_names(requirements: tuple[str, ...]) -> tuple[str, ...]:
    """Return requirement names in declared order."""
    return tuple(_requirement_name(requirement) for requirement in requirements)


def _is_exact_pin(requirement: Requirement) -> bool:
    """Return True when a requirement is pinned with a single equality specifier."""
    specs = tuple(requirement.specifier)
    return len(specs) == 1 and specs[0].operator == "=="


def _pin_satisfies(pinned: Requirement, declared: Requirement) -> bool:
    """Return True when a pinned version is admitted by the declared range."""
    version = next(iter(pinned.specifier)).version
    return declared.specifier.contains(version, prereleases=True)


def project_dependencies(pyproject_path: Path) -> tuple[str, ...]:
    """Read the top-level [project] dependencies list from pyproject.toml."""
    text = pyproject_path.read_text(encoding="utf-8")
    in_block = False
    dependencies: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "dependencies = [":
            in_block = True
            continue
        if in_block and stripped == "]":
            break
        if in_block and stripped.startswith('"'):
            dependencies.append(stripped.rstrip(",").strip('"'))
    if not dependencies:
        raise ValueError(f"No [project] dependencies block found in {pyproject_path}")
    return tuple(dependencies)


def runtime_requirements(requirements_path: Path) -> tuple[str, ...]:
    """Read non-comment runtime requirements from requirements.txt."""
    requirements: list[str] = []
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-r "):
            continue
        requirements.append(stripped)
    return tuple(requirements)


def dependency_drift_report(root: Path) -> DependencyDriftReport:
    """Build a drift report for a repository root."""
    return DependencyDriftReport(
        project_dependencies=project_dependencies(root / "pyproject.toml"),
        runtime_requirements=runtime_requirements(root / "requirements.txt"),
    )


def format_report(report: DependencyDriftReport) -> str:
    """Render a human-readable dependency drift report."""
    if report.in_sync:
        return "requirements.txt mirrors pyproject.toml runtime dependencies."

    lines = ["requirements.txt does not mirror pyproject.toml runtime dependencies."]
    if report.missing_from_requirements:
        lines.append("Missing from requirements.txt:")
        lines.extend(f"  - {dep}" for dep in report.missing_from_requirements)
    if report.extra_in_requirements:
        lines.append("Extra in requirements.txt:")
        lines.extend(f"  - {req}" for req in report.extra_in_requirements)
    if report.order_mismatch:
        lines.append("Dependency order differs; keep requirements.txt in pyproject order.")
    if report.unsatisfied_pins:
        lines.append("Pins outside declared pyproject ranges:")
        lines.extend(f"  - {req}" for req in report.unsatisfied_pins)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing pyproject.toml and requirements.txt.",
    )
    args = parser.parse_args(argv)

    report = dependency_drift_report(args.root)
    print(format_report(report))
    return 0 if report.in_sync else 1


if __name__ == "__main__":
    raise SystemExit(main())
