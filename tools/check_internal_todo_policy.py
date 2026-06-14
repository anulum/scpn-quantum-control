#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Canonical internal TODO policy guard
"""Verify the canonical internal TODO and internal-artefact tracking policy.

How it works:
  1. The single canonical internal task queue is ``docs/internal/TODO.md``. The
     working tree must hold no competing TODO-named queue anywhere else; a file
     whose name is ``TODO``/``TODO.md``/``TODO.txt``/``TODO.rst`` outside the
     canonical path is a competing queue and fails the check.
  2. Internal and planning trees (``docs/internal/``, ``docs/superpowers/``,
     ``.coordination/``) must stay local. No file under them may be tracked in
     git; a tracked file there fails the check.
  3. A missing canonical TODO is reported as a non-blocking notice, because the
     queue is gitignored and legitimately absent on a fresh checkout.

Exit status is non-zero only when a blocking violation is found, so the guard is
safe to run on a fresh clone.

Usage:
  python tools/check_internal_todo_policy.py             # check the current repo
  python tools/check_internal_todo_policy.py --root DIR  # check another worktree
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

CANONICAL_TODO = Path("docs/internal/TODO.md")
LOCAL_ONLY_TREES = ("docs/internal", "docs/superpowers", ".coordination")
_TODO_BASENAME = re.compile(r"^todo(\.(md|txt|rst))?$", re.IGNORECASE)
_SKIP_DIRS = frozenset(
    {
        ".git",
        "__pycache__",
        "node_modules",
        "build",
        "dist",
        ".eggs",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "ARCHIVE",
        "BACKUP",
    }
)


@dataclass(frozen=True)
class Finding:
    """A single policy observation; ``blocking`` drives the process exit status."""

    category: str
    path: str
    detail: str
    blocking: bool


def _repo_root(start: Path) -> Path:
    result = subprocess.run(
        ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def tracked_local_only_paths(root: Path) -> list[str]:
    """Return git-tracked paths under the local-only trees (policy violations)."""
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--", *LOCAL_ONLY_TREES],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _iter_files(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name for name in dirnames if name not in _SKIP_DIRS and not name.startswith(".venv")
        ]
        for name in filenames:
            yield Path(dirpath) / name


def competing_todo_files(root: Path) -> list[Path]:
    """Return TODO-named files outside the canonical queue (competing queues)."""
    canonical = (root / CANONICAL_TODO).resolve()
    return [
        path
        for path in _iter_files(root)
        if _TODO_BASENAME.match(path.name) and path.resolve() != canonical
    ]


def check_internal_todo_policy(root: Path) -> list[Finding]:
    """Evaluate the canonical-TODO and internal-tracking policy for ``root``."""
    findings: list[Finding] = []
    if not (root / CANONICAL_TODO).is_file():
        findings.append(
            Finding(
                "missing-canonical-todo",
                str(CANONICAL_TODO),
                "the single canonical internal TODO is absent (gitignored, may be a fresh checkout)",
                blocking=False,
            )
        )
    for tracked in tracked_local_only_paths(root):
        findings.append(
            Finding(
                "tracked-internal-path",
                tracked,
                "internal/planning artefact must stay local, not tracked in git",
                blocking=True,
            )
        )
    for path in competing_todo_files(root):
        findings.append(
            Finding(
                "competing-todo",
                str(path.relative_to(root)),
                "competing TODO queue; the only queue is docs/internal/TODO.md",
                blocking=True,
            )
        )
    return findings


def format_findings(findings: list[Finding]) -> str:
    """Render findings as a single human-readable block."""
    if not findings:
        return "internal TODO policy: OK (1 canonical queue, no tracked internal artefacts)"
    blocking = sum(1 for finding in findings if finding.blocking)
    header = f"internal TODO policy: {blocking} blocking, {len(findings) - blocking} notice(s)"
    lines = [header]
    for finding in findings:
        tag = "FAIL" if finding.blocking else "note"
        lines.append(f"  [{tag}] {finding.category}: {finding.path} — {finding.detail}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    """Run the policy guard; return 1 only when a blocking violation is found."""
    parser = argparse.ArgumentParser(description="Canonical internal TODO policy guard")
    parser.add_argument("--root", default=None, help="worktree to check (default: current repo)")
    args = parser.parse_args(argv)
    start = Path(args.root).resolve() if args.root else Path.cwd()
    root = _repo_root(start)
    findings = check_internal_todo_policy(root)
    print(format_findings(findings))
    return 1 if any(finding.blocking for finding in findings) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
