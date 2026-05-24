# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Test quality audit helper
"""Audit pytest module names for forbidden non-specific bucket patterns."""

from __future__ import annotations

import argparse
import dataclasses
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOTS: tuple[str, ...] = ("src", "tools", "scripts")

FORBIDDEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^test_cov.*\.py$"),
    re.compile(r"^test_coverage.*\.py$"),
    re.compile(r".*_coverage.*\.py$"),
    re.compile(r".*_coverage_closure\.py$"),
    re.compile(r".*_final_gaps\.py$"),
    re.compile(r".*_remaining\.py$"),
    re.compile(r".*remaining.*\.py$"),
    re.compile(r".*_push\.py$"),
    re.compile(r".*coverage_100.*\.py$"),
    re.compile(r"^test_runner_coverage\.py$"),
    re.compile(r"^test_experiments_coverage\.py$"),
    re.compile(r"^test_batch.*\.py$"),
    re.compile(r"^test_new_modules.*\.py$"),
    re.compile(r"^test_e2e_new_modules\.py$"),
    re.compile(r"^test_round\d+.*\.py$"),
    re.compile(r"^test_misc.*\.py$"),
    re.compile(r"^test_final.*\.py$"),
)

MODULE_SPECIFIC_EXCEPTIONS: frozenset[str] = frozenset(
    {
        "test_audit_coverage_gaps.py",
    }
)


def _module_stems() -> frozenset[str]:
    """Return known production/tool/script module stems once per audit run."""

    stems: set[str] = set()
    for module_root in MODULE_ROOTS:
        base = ROOT / module_root
        if base.exists():
            stems.update(item.stem for item in base.rglob("*.py"))
    return frozenset(stems)


def _module_backed(path: Path, module_stems: frozenset[str]) -> bool:
    """Return whether a test module maps to a real module/tool/script stem."""

    stem = path.stem.removeprefix("test_")
    return stem in module_stems


@dataclasses.dataclass(frozen=True)
class TestQualityFinding:
    """One forbidden pytest module naming finding."""

    path: Path
    reason: str


def audit_test_paths(paths: Iterable[Path]) -> list[TestQualityFinding]:
    """Return forbidden non-specific pytest module names."""

    findings: list[TestQualityFinding] = []
    module_stems = _module_stems()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        name = path.name
        if name in MODULE_SPECIFIC_EXCEPTIONS:
            continue
        if _module_backed(path, module_stems):
            continue
        if any(pattern.fullmatch(name) for pattern in FORBIDDEN_PATTERNS):
            findings.append(
                TestQualityFinding(
                    path=path,
                    reason=(
                        "non-specific bucket pytest modules are forbidden; move "
                        "behaviour into domain-specific regression or contract tests"
                    ),
                )
            )
    return findings


def repository_test_paths(root: Path) -> list[Path]:
    """Return tracked and worktree pytest modules below ``tests``."""

    try:
        proc = subprocess.run(
            ["git", "ls-files", "tests/*.py"],
            check=True,
            cwd=root,
            text=True,
            capture_output=True,
        )
        tracked = {Path(line) for line in proc.stdout.splitlines() if line}
    except (FileNotFoundError, subprocess.CalledProcessError):
        tracked = set()
    worktree = {path.relative_to(root) for path in (root / "tests").glob("*.py")}
    return sorted(tracked | worktree, key=lambda item: item.as_posix())


def format_findings(findings: Iterable[TestQualityFinding]) -> str:
    """Render findings for commit hooks and CI logs."""

    rows = [f"{finding.path.as_posix()}: {finding.reason}" for finding in findings]
    if not rows:
        return "test-quality audit passed: no forbidden non-specific test modules"
    return "test-quality audit failed:\n" + "\n".join(f"- {row}" for row in rows)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to the parent of tools/.",
    )
    args = parser.parse_args(argv)

    findings = audit_test_paths(repository_test_paths(args.root))
    print(format_findings(findings))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
