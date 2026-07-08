# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the p_h1 Open-Claim Guard
"""Tests for the public p_h1 open-claim guard."""

from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.analysis import (
    P_H1_OPEN_CLAIM_BOUNDARY,
    P_H1_OPEN_GUARD_SCHEMA,
    P_H1OpenGuardReport,
    public_markdown_paths,
    run_p_h1_open_guard,
    validate_p_h1_open_claim_text,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT_PATH = REPO_ROOT / "scripts" / "check_p_h1_open_claim_guard.py"


def _load_export_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_check_p_h1_open_claim_guard", EXPORT_SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


export_script = _load_export_script()


def test_guard_rejects_closed_derivation_wording() -> None:
    """Closed derivation phrasing for p_h1=0.72 is rejected."""
    violations = validate_p_h1_open_claim_text(
        path="synthetic.md",
        text="The derivation closing the p_h1 gap proves p_h1 = 0.72.",
    )

    assert len(violations) == 2
    assert violations[0].path == "synthetic.md"
    assert "closed-claim pattern" in violations[0].reason
    assert violations[0].to_dict()["path"] == "synthetic.md"


def test_guard_requires_open_marker_for_072_paragraph() -> None:
    """Bare p_h1=0.72 prose must carry an open-question marker."""
    violations = validate_p_h1_open_claim_text(
        path="synthetic.md",
        text="At the operating point, p_h1 = 0.72 controls the transition.",
    )

    assert len(violations) == 1
    assert "lacks an open-question marker" in violations[0].reason


def test_guard_accepts_open_parameter_wording() -> None:
    """Open-parameter wording is accepted for the same numeric threshold."""
    violations = validate_p_h1_open_claim_text(
        path="synthetic.md",
        text=(
            "p_h1 = 0.72 remains an open empirical/theoretical parameter; "
            "the square-lattice expression is a negative control."
        ),
    )

    assert violations == ()


def test_guard_scans_current_public_docs() -> None:
    """The current outward Markdown surface keeps p_h1 open."""
    report = run_p_h1_open_guard(REPO_ROOT)

    assert isinstance(report, P_H1OpenGuardReport)
    assert report.schema == P_H1_OPEN_GUARD_SCHEMA
    assert report.claim_boundary == P_H1_OPEN_CLAIM_BOUNDARY
    assert report.passed
    assert report.violations == ()
    assert "docs/falsification.md" in report.checked_paths
    assert "docs/research_gems.md" in report.marker_hits


def test_public_markdown_discovery_excludes_internal_docs() -> None:
    """Public discovery covers root/docs Markdown and excludes internal plans."""
    paths = public_markdown_paths(REPO_ROOT)
    displays = {str(path.relative_to(REPO_ROOT)) for path in paths}

    assert "README.md" in displays
    assert "docs/falsification.md" in displays
    assert not any("docs/internal" in display for display in displays)


def test_guard_reports_explicit_out_of_repo_paths(tmp_path: Path) -> None:
    """Explicit scan paths outside the repo are reported as absolute paths."""
    external = tmp_path / "external.md"
    external.write_text(
        "p_h1 = 0.72 remains an open empirical/theoretical parameter.\n",
        encoding="utf-8",
    )

    report = run_p_h1_open_guard(REPO_ROOT, paths=(external,))

    assert report.passed
    assert report.checked_paths == (str(external),)
    assert report.marker_hits == (str(external),)


def test_export_script_writes_guard_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard report regenerates through the export script."""
    output = tmp_path / "p_h1_open_guard.json"
    monkeypatch.setattr(export_script, "parse_args", lambda: Namespace(output=output))

    assert export_script.main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == P_H1_OPEN_GUARD_SCHEMA
    assert payload["passed"] is True
    assert payload["violations"] == []
