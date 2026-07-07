# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Changelog consistency guards
"""Guard the root/published changelog pair against version-surface drift.

``CHANGELOG.md`` is the canonical release history; ``docs/changelog.md`` is the
published curated summary that links back to it. These guards fail closed on
the three drift classes found on 2026-07-07: duplicate version headers in the
root file, summary entries whose release date contradicts the root file, and
root versions missing from the summary (either as a single header or covered
by an explicit range header such as ``[0.2.0–0.2.7]``).
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ROOT_CHANGELOG = _REPO_ROOT / "CHANGELOG.md"
_DOCS_CHANGELOG = _REPO_ROOT / "docs" / "changelog.md"

_VERSION_HEADER = re.compile(r"^## \[(?P<version>\d+\.\d+\.\d+)\] - (?P<date>.+?)\s*$")
_RANGE_HEADER = re.compile(
    r"^## \[(?P<lo>\d+\.\d+\.\d+)[–-](?P<hi>\d+\.\d+\.\d+)\] - (?P<date>.+?)\s*$"
)
_DATE_FORM = re.compile(r"^\d{4}-\d{2}-\d{2}( / \d{4}-\d{2}-\d{2})?$")


def _version_key(version: str) -> tuple[int, int, int]:
    """Return the numeric sort key for a dotted ``major.minor.patch`` version.

    Parameters
    ----------
    version : str
        Version string of the exact form ``"X.Y.Z"``.

    Returns
    -------
    tuple of int
        ``(major, minor, patch)`` suitable for ordering comparisons.
    """
    major, minor, patch = version.split(".")
    return (int(major), int(minor), int(patch))


def _single_version_headers(path: Path) -> list[tuple[str, str]]:
    """Return ``(version, date)`` pairs for every single-version header.

    Parameters
    ----------
    path : Path
        Changelog file to scan.

    Returns
    -------
    list of tuple of str
        Headers in file order, one entry per ``## [X.Y.Z] - DATE`` line.
    """
    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _VERSION_HEADER.match(line)
        if match is not None:
            pairs.append((match.group("version"), match.group("date")))
    return pairs


def _range_headers(path: Path) -> list[tuple[str, str]]:
    """Return ``(low, high)`` version bounds for every range header.

    Parameters
    ----------
    path : Path
        Changelog file to scan.

    Returns
    -------
    list of tuple of str
        Inclusive version ranges declared as ``## [A.B.C–D.E.F] - DATE``.
    """
    ranges: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _RANGE_HEADER.match(line)
        if match is not None:
            ranges.append((match.group("lo"), match.group("hi")))
    return ranges


def test_root_changelog_version_headers_are_unique() -> None:
    """Every version appears exactly once in the canonical changelog."""
    versions = [version for version, _ in _single_version_headers(_ROOT_CHANGELOG)]
    duplicates = sorted({v for v in versions if versions.count(v) > 1})
    assert duplicates == [], f"duplicate version headers in CHANGELOG.md: {duplicates}"


def test_root_changelog_versions_are_strictly_descending() -> None:
    """The canonical changelog lists releases newest-first without reorder drift."""
    keys = [_version_key(v) for v, _ in _single_version_headers(_ROOT_CHANGELOG)]
    assert keys == sorted(keys, reverse=True), "CHANGELOG.md version order is not descending"


def test_docs_summary_dates_match_root() -> None:
    """Every published summary entry carries the canonical release date."""
    root_dates = dict(_single_version_headers(_ROOT_CHANGELOG))
    mismatches: list[str] = []
    for version, date in _single_version_headers(_DOCS_CHANGELOG):
        canonical = root_dates.get(version)
        if canonical is None:
            mismatches.append(f"{version}: absent from CHANGELOG.md")
        elif canonical != date:
            mismatches.append(f"{version}: docs='{date}' root='{canonical}'")
    assert mismatches == [], f"docs/changelog.md drifts from CHANGELOG.md: {mismatches}"


def test_docs_summary_covers_every_root_version() -> None:
    """Every canonical release is present in the published summary.

    A release counts as covered by its own ``## [X.Y.Z]`` header or by an
    explicit range header (the 0.2.x line ships as ``[0.2.0–0.2.7]``).
    """
    docs_versions = {v for v, _ in _single_version_headers(_DOCS_CHANGELOG)}
    ranges = [(_version_key(lo), _version_key(hi)) for lo, hi in _range_headers(_DOCS_CHANGELOG)]
    missing = [
        version
        for version, _ in _single_version_headers(_ROOT_CHANGELOG)
        if version not in docs_versions
        and not any(lo <= _version_key(version) <= hi for lo, hi in ranges)
    ]
    assert missing == [], f"root versions missing from docs/changelog.md: {missing}"


def test_version_header_dates_are_well_formed() -> None:
    """Header dates use ``YYYY-MM-DD`` (optionally a ``/``-joined span)."""
    malformed: list[str] = []
    for path in (_ROOT_CHANGELOG, _DOCS_CHANGELOG):
        for version, date in _single_version_headers(path):
            if _DATE_FORM.match(date) is None:
                malformed.append(f"{path.name} {version}: '{date}'")
    assert malformed == [], f"malformed changelog header dates: {malformed}"
