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


@dataclass(frozen=True)
class DependencyDriftReport:
    """Comparison between canonical pyproject dependencies and requirements."""

    project_dependencies: tuple[str, ...]
    runtime_requirements: tuple[str, ...]

    @property
    def in_sync(self) -> bool:
        """Return True when requirements match pyproject dependencies exactly."""
        return self.project_dependencies == self.runtime_requirements

    @property
    def missing_from_requirements(self) -> tuple[str, ...]:
        """Dependencies present in pyproject but absent from requirements."""
        return tuple(
            dep for dep in self.project_dependencies if dep not in self.runtime_requirements
        )

    @property
    def extra_in_requirements(self) -> tuple[str, ...]:
        """Requirements present in requirements.txt but absent from pyproject."""
        return tuple(
            req for req in self.runtime_requirements if req not in self.project_dependencies
        )

    @property
    def order_mismatch(self) -> bool:
        """Return True when the sets match but the mirror order differs."""
        return (
            not self.in_sync
            and not self.missing_from_requirements
            and not self.extra_in_requirements
        )


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
