# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — p_h1 Open-Claim Guard
"""Public-claim guard for the open ``p_h1 = 0.72`` question.

The guard scans outward-facing Markdown surfaces and rejects wording that
turns the current ``p_h1 = 0.72`` threshold into a closed derivation,
first-principles result, universal constant, or measured TCBO reproduction.
It keeps the public claim aligned with the executable p_h1 audit: the
square-lattice match is a negative control, while the K_nm graph check keeps
the parameter open.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

P_H1_OPEN_GUARD_SCHEMA: Final = "scpn_qc_p_h1_open_guard_v1"
P_H1_OPEN_CLAIM_BOUNDARY: Final = (
    "p_h1 = 0.72 remains an open empirical/theoretical parameter; public "
    "surfaces must not present it as a closed derivation or measured reproduction."
)

_P_H1_TOKENS: Final = (
    "p_h1",
    "p_h",
    "p_{h",
    "p_{H",
)
_OPEN_MARKERS: Final = (
    "open empirical/theoretical parameter",
    "open empirical/theoretical",
    "open until reproduced",
    "open empirical",
    "open theoretical",
    "open question",
    "remains open",
    "still open",
    "not a derived",
    "not derived",
    "not a first-principles",
    "not a universal",
    "cannot be derived",
    "no known first-principles",
    "negative control",
    "falsified",
    "coincidence",
)
_FORBIDDEN_CLOSED_PATTERNS: Final = (
    r"derivation\s+closing\s+the\s+\$?p_\{?h_?1\}?\$?\s+gap",
    r"analytical\s+formula\s+behind\s+the\s+universal\s+\$?p_\{?h_?1\}?\$?\s+value",
    r"p_\{?h_?1\}?\s*=\s*0\.72[^.\n]{0,180}\b(first-principles|derived|derivation|universal constant|measured)",
    r"persistent\s+homology\s+\$?p_\{?h_?1\}?\$?\s*=\s*0\.72\s*\(measured",
)


@dataclass(frozen=True)
class P_H1OpenGuardViolation:
    """One public-surface wording violation for the p_h1 open question."""

    path: str
    reason: str
    excerpt: str

    def to_dict(self) -> dict[str, str]:
        """Return JSON-compatible violation data."""
        return {"path": self.path, "reason": self.reason, "excerpt": self.excerpt}


@dataclass(frozen=True)
class P_H1OpenGuardReport:
    """Complete p_h1 public-claim guard result."""

    schema: str
    checked_paths: tuple[str, ...]
    marker_hits: tuple[str, ...]
    violations: tuple[P_H1OpenGuardViolation, ...]
    claim_boundary: str = P_H1_OPEN_CLAIM_BOUNDARY

    @property
    def passed(self) -> bool:
        """Return whether every checked public surface respects the open boundary."""
        return not self.violations

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible guard report data."""
        return {
            "schema": self.schema,
            "checked_paths": list(self.checked_paths),
            "marker_hits": list(self.marker_hits),
            "violations": [violation.to_dict() for violation in self.violations],
            "claim_boundary": self.claim_boundary,
            "passed": self.passed,
        }


def public_markdown_paths(repo_root: Path) -> tuple[Path, ...]:
    """Return outward-facing Markdown paths that can carry p_h1 claims.

    Parameters
    ----------
    repo_root:
        Repository root used to discover root-level Markdown files and
        ``docs/**/*.md`` files.

    Returns
    -------
    tuple[Path, ...]
        Sorted public Markdown paths, excluding ignored internal and rendered
        documentation trees.
    """
    root_paths = sorted(repo_root.glob("*.md"))
    docs_root = repo_root / "docs"
    doc_paths = sorted(
        path
        for path in docs_root.rglob("*.md")
        if "internal" not in path.parts and "site" not in path.parts
    )
    return tuple(root_paths + doc_paths)


def validate_p_h1_open_claim_text(*, path: str, text: str) -> tuple[P_H1OpenGuardViolation, ...]:
    """Validate one public text surface against the p_h1 open-claim boundary.

    Parameters
    ----------
    path:
        Display path included in any returned violation.
    text:
        Markdown or prose content to scan.

    Returns
    -------
    tuple[P_H1OpenGuardViolation, ...]
        Violations found in the supplied content.
    """
    stripped = _strip_fenced_code(text)
    violations: list[P_H1OpenGuardViolation] = []
    lowered = stripped.lower()
    for pattern in _FORBIDDEN_CLOSED_PATTERNS:
        match = re.search(pattern, lowered)
        if match is not None:
            excerpt = _excerpt(stripped, match.start(), match.end())
            if any(marker in excerpt.lower() for marker in _OPEN_MARKERS):
                continue
            violations.append(
                P_H1OpenGuardViolation(
                    path=path,
                    reason=f"closed-claim pattern: {pattern}",
                    excerpt=excerpt,
                )
            )

    for paragraph in re.split(r"\n\s*\n", stripped):
        paragraph_lowered = paragraph.lower()
        collapsed_paragraph = _collapse_whitespace(paragraph_lowered)
        if "0.72" not in paragraph_lowered or not _mentions_p_h1(paragraph_lowered):
            continue
        if any(marker in collapsed_paragraph for marker in _OPEN_MARKERS):
            continue
        violations.append(
            P_H1OpenGuardViolation(
                path=path,
                reason="p_h1=0.72 paragraph lacks an open-question marker",
                excerpt=_collapse_whitespace(paragraph)[:240],
            )
        )
    return tuple(violations)


def run_p_h1_open_guard(
    repo_root: Path, paths: tuple[Path, ...] | None = None
) -> P_H1OpenGuardReport:
    """Scan public Markdown surfaces for p_h1 closed-claim drift.

    Parameters
    ----------
    repo_root:
        Repository root used for path display and path discovery.
    paths:
        Optional explicit paths. When omitted, :func:`public_markdown_paths`
        discovers the public Markdown surfaces.

    Returns
    -------
    P_H1OpenGuardReport
        Pass/fail report with checked paths, open-boundary marker hits, and
        wording violations.
    """
    active_paths = paths or public_markdown_paths(repo_root)
    violations: list[P_H1OpenGuardViolation] = []
    marker_hits: list[str] = []
    checked_paths: list[str] = []
    for path in active_paths:
        display = _display_path(path, repo_root)
        text = path.read_text(encoding="utf-8")
        if "0.72" in text and _mentions_p_h1(text.lower()):
            checked_paths.append(display)
            lowered = text.lower()
            if any(marker in lowered for marker in _OPEN_MARKERS):
                marker_hits.append(display)
        violations.extend(validate_p_h1_open_claim_text(path=display, text=text))
    return P_H1OpenGuardReport(
        schema=P_H1_OPEN_GUARD_SCHEMA,
        checked_paths=tuple(checked_paths),
        marker_hits=tuple(marker_hits),
        violations=tuple(violations),
    )


def _strip_fenced_code(text: str) -> str:
    without_fences = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    return re.sub(r"(?m)^(?: {4}|\t).*$", "", without_fences)


def _mentions_p_h1(lowered: str) -> bool:
    return any(token in lowered for token in _P_H1_TOKENS)


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _excerpt(text: str, start: int, end: int) -> str:
    prefix = max(0, start - 80)
    suffix = min(len(text), end + 80)
    return _collapse_whitespace(text[prefix:suffix])


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
