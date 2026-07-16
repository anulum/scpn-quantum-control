#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Retired public-claims guard
"""Fail when a retired public claim reappears on a public claim surface.

How it works:
  1. ``data/retired_claims.json`` registers every retired claim as a regular
     expression together with the retraction date, the reason, and a list of
     ``allow_context`` regexes. A line that matches a claim pattern passes
     only when the line or one of its immediate neighbours (±1 line, so
     wrapped retraction prose still counts) matches one of that claim's
     allow-context regexes — i.e. when the occurrence is the retraction or a
     dated amendment, not a revival of the claim.
  2. The scanned public surfaces are the repository-root Markdown files
     (README, ROADMAP, RESULTS_SUMMARY, CHANGELOG, …) plus ``docs/``,
     ``paper/`` and ``results/`` (Markdown and LaTeX). Local-only trees
     (``docs/internal/``, ``docs/superpowers/``) are outside the public
     surface and are skipped, as are the register's ``exempt_paths`` —
     published records (preprint sources, paper submissions) are read-only
     by owner ruling (2026-07-16) and guarded separately by
     ``tools/check_published_record_freeze.py``; their corrections travel
     via live-surface amendments and new Zenodo versions.
  3. Any match without retraction context is a blocking finding: the guard
     prints ``file:line`` with the claim id and exits non-zero. This is the
     durable half of a claim retraction — the purge removes the text, the
     guard keeps it removed.

Usage:
  python tools/check_retired_claims.py                # check the current repo
  python tools/check_retired_claims.py --root DIR     # check another worktree
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG = Path("data/retired_claims.json")

#: Trees inside the public directories that are NOT public claim surfaces.
EXCLUDED_TREES = ("docs/internal", "docs/superpowers")

#: Directories whose Markdown/LaTeX content forms the public claim surface.
SURFACE_DIRS = ("docs", "paper", "results")

SURFACE_SUFFIXES = frozenset({".md", ".tex"})


@dataclass(frozen=True)
class RetiredClaim:
    """One retired claim: its pattern and the contexts that may restate it."""

    claim_id: str
    pattern: re.Pattern[str]
    retired: str
    reason: str
    allow_context: tuple[re.Pattern[str], ...]

    def occurrence_is_allowed(self, window: Sequence[str]) -> bool:
        """Report whether the occurrence carries retraction/amendment context.

        ``window`` is the matching line with its immediate neighbours, so a
        retraction sentence wrapped across lines still whitelists the match.
        """
        return any(context.search(line) for context in self.allow_context for line in window)


@dataclass(frozen=True)
class Finding:
    """One blocking occurrence of a retired claim without retraction context."""

    path: Path
    line_number: int
    claim_id: str
    line: str


def load_claims(config_path: Path) -> tuple[RetiredClaim, ...]:
    """Load and compile the retired-claims register."""
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    entries = payload.get("claims")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{config_path}: 'claims' must be a non-empty list")
    claims: list[RetiredClaim] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"{config_path}: claim entries must be objects")
        try:
            claims.append(
                RetiredClaim(
                    claim_id=str(entry["id"]),
                    pattern=re.compile(str(entry["pattern"])),
                    retired=str(entry["retired"]),
                    reason=str(entry["reason"]),
                    allow_context=tuple(
                        re.compile(str(context)) for context in entry["allow_context"]
                    ),
                )
            )
        except KeyError as exc:
            raise ValueError(f"{config_path}: claim entry missing key {exc}") from exc
    return tuple(claims)


def load_exempt_paths(config_path: Path) -> tuple[str, ...]:
    """Load exempt path prefixes (published read-only records) from the register."""
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    entries = payload.get("exempt_paths", [])
    if not isinstance(entries, list):
        raise ValueError(f"{config_path}: 'exempt_paths' must be a list when present")
    return tuple(str(entry) for entry in entries)


def _is_under(relative: str, prefixes: Sequence[str]) -> bool:
    return any(
        relative == prefix.rstrip("/") or relative.startswith(prefix.rstrip("/") + "/")
        for prefix in prefixes
    )


def iter_surface_files(root: Path, exempt: Sequence[str] = ()) -> Iterator[Path]:
    """Yield every public claim-surface file under the repository root."""
    for candidate in sorted(root.glob("*")):
        if (
            candidate.is_file()
            and candidate.suffix in SURFACE_SUFFIXES
            and not _is_under(candidate.name, exempt)
        ):
            yield candidate
    for surface_dir in SURFACE_DIRS:
        base = root / surface_dir
        if not base.is_dir():
            continue
        for candidate in sorted(base.rglob("*")):
            if not candidate.is_file() or candidate.suffix not in SURFACE_SUFFIXES:
                continue
            relative = candidate.relative_to(root).as_posix()
            if _is_under(relative, EXCLUDED_TREES) or _is_under(relative, exempt):
                continue
            yield candidate


def scan_file(path: Path, claims: Sequence[RetiredClaim], root: Path) -> list[Finding]:
    """Scan one file; return blocking findings."""
    findings: list[Finding] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        line_number = index + 1
        window = lines[max(0, index - 1) : index + 2]
        for claim in claims:
            if claim.pattern.search(line) and not claim.occurrence_is_allowed(window):
                findings.append(
                    Finding(
                        path=path.relative_to(root),
                        line_number=line_number,
                        claim_id=claim.claim_id,
                        line=line.strip(),
                    )
                )
    return findings


def scan_repo(
    root: Path, claims: Sequence[RetiredClaim], exempt: Sequence[str] = ()
) -> list[Finding]:
    """Scan every public surface file under the root."""
    findings: list[Finding] = []
    for path in iter_surface_files(root, exempt):
        findings.extend(scan_file(path, claims, root))
    return findings


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root to scan (default: current directory).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Retired-claims register (default: data/retired_claims.json under root).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    root = args.root.resolve()
    config_path = args.config if args.config is not None else root / DEFAULT_CONFIG
    claims = load_claims(config_path)
    exempt = load_exempt_paths(config_path)
    findings = scan_repo(root, claims, exempt)
    if findings:
        print(f"Retired-claims guard: {len(findings)} blocking finding(s):")
        for finding in findings:
            print(f"  {finding.path}:{finding.line_number}: [{finding.claim_id}] {finding.line}")
        print(
            "A retired claim may only appear with explicit retraction/amendment "
            "context (see data/retired_claims.json)."
        )
        return 1
    print(f"Retired-claims guard: no findings ({len(claims)} claim(s) enforced).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
