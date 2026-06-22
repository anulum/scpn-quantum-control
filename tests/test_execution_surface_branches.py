# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the execution-surface manifest
"""Branch and guard tests for the execution-surface manifest evaluator.

Covers the violation serialiser, the unknown-classification guard, the
non-blocking skip and missing-path violation, the absolute-path resolution
guard and the out-of-range line lookup.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scpn_quantum_control.execution_surface import (
    ExecutionSurfaceViolation,
    _line_at,
    _resolve_repo_path,
    evaluate_execution_surface_manifest,
    load_execution_surface_manifest,
)


def test_violation_serialises_to_dict() -> None:
    """A violation record serialises all fields."""
    violation = ExecutionSurfaceViolation(path="scripts/x.py", reason="missing")
    payload = violation.to_dict()
    assert payload["path"] == "scripts/x.py"
    assert payload["reason"] == "missing"
    assert payload["rule"] is None


def test_load_manifest_rejects_unknown_classification(tmp_path: Path) -> None:
    """An unknown surface classification is rejected."""
    manifest = tmp_path / "surface.toml"
    manifest.write_text(
        '[[surface]]\npath = "scripts/a.py"\nclassification = "bogus_class"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown execution-surface classification"):
        load_execution_surface_manifest(manifest)


def test_evaluate_skips_non_blocking_and_flags_missing_path(tmp_path: Path) -> None:
    """Non-blocking entries are skipped; blocking entries with no path are flagged."""
    manifest = tmp_path / "surface.toml"
    manifest.write_text(
        "\n".join(
            [
                "[[surface]]",
                'path = "missing_surface.py"',
                'classification = "trusted_static"',
                "ci_blocking = true",
                "",
                "[[surface]]",
                'path = "ignored_surface.py"',
                'classification = "trusted_static"',
                "ci_blocking = false",
            ]
        ),
        encoding="utf-8",
    )
    violations = evaluate_execution_surface_manifest(tmp_path, manifest)
    flagged = {v.path: v.reason for v in violations}
    assert flagged == {"missing_surface.py": "manifest path is missing"}


def test_resolve_repo_path_rejects_absolute(tmp_path: Path) -> None:
    """An absolute manifest path does not resolve inside the repository."""
    assert _resolve_repo_path(tmp_path, "/etc/passwd") is None


def test_line_at_returns_empty_for_out_of_range() -> None:
    """A line index outside the text returns an empty string."""
    assert _line_at("alpha\nbeta", 99) == ""
