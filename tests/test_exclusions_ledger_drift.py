# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Coverage-exclusions ledger drift gate
"""Drift gate binding the coverage omit lists to the exclusions ledger.

Every glob in ``[tool.coverage.run].omit`` and ``[tool.coverage.report].omit``
must have exactly one row in ``docs/release_coverage_exclusions.json``, and
every ledger row must match a live omit glob. This is the coverage analogue of
the no-``pragma: no cover``-without-issue rule: an omitted file can never be a
silent hole — it must carry a recorded reason and the lane that exercises it
(or an explicit tracked gap). The test fails the build the moment the two
drift apart in either direction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
LEDGER = REPO_ROOT / "docs" / "release_coverage_exclusions.json"


def _normalise(glob: str) -> str:
    """Reduce a coverage glob or ledger path to a comparable package-relative form.

    Coverage globs are anchored with ``*/`` (``*/scpn_quantum_control/...``);
    ledger rows are repo-relative (``src/scpn_quantum_control/...``). Both
    collapse to ``scpn_quantum_control/...``.
    """
    for prefix in ("*/", "src/"):
        if glob.startswith(prefix):
            return glob[len(prefix) :]
    return glob


def _omit_globs() -> set[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    coverage = data["tool"]["coverage"]
    run = coverage["run"]["omit"]
    report = coverage["report"]["omit"]
    return {_normalise(g) for g in (*run, *report)}


def _ledger_entries() -> list[dict[str, str]]:
    payload = json.loads(LEDGER.read_text(encoding="utf-8"))
    entries = payload["exclusions"]
    assert isinstance(entries, list) and entries, "ledger must hold a non-empty list"
    return entries


def _ledger_globs() -> set[str]:
    keyed = [e.get("path") or e.get("path_glob") for e in _ledger_entries()]
    return {_normalise(g) for g in keyed if g}


def test_every_omit_glob_has_a_ledger_row() -> None:
    """No coverage-omitted file may exist without a recorded justification."""
    missing = _omit_globs() - _ledger_globs()
    assert not missing, (
        "coverage omit globs with no ledger row (add a row to "
        f"docs/release_coverage_exclusions.json): {sorted(missing)}"
    )


def test_every_ledger_row_matches_a_live_omit_glob() -> None:
    """A ledger row for a file that is no longer omitted is stale and must go."""
    stale = _ledger_globs() - _omit_globs()
    assert not stale, (
        "ledger rows that match no coverage omit glob (remove them from "
        f"docs/release_coverage_exclusions.json): {sorted(stale)}"
    )


def test_ledger_rows_are_one_to_one_with_omit_globs() -> None:
    """The ledger holds exactly one row per omit glob — no duplicates, no extras."""
    keyed = [_normalise(e.get("path") or e.get("path_glob", "")) for e in _ledger_entries()]
    duplicates = {g for g in keyed if keyed.count(g) > 1}
    assert not duplicates, f"duplicate ledger rows: {sorted(duplicates)}"
    assert set(keyed) == _omit_globs()


def test_every_ledger_row_states_a_reason_and_an_exercised_by() -> None:
    """Each row must record WHY it is omitted and WHERE it is exercised (or the gap)."""
    for index, entry in enumerate(_ledger_entries()):
        reason = entry.get("reason")
        exercised_by = entry.get("exercised_by")
        assert isinstance(reason, str) and reason.strip(), f"row {index} needs a reason"
        assert isinstance(exercised_by, str) and exercised_by.strip(), (
            f"row {index} needs an 'exercised_by' (name the CI lane or the tracked gap)"
        )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
