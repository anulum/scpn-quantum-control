# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- CI documentation surface gate contract
"""Static contract for the CI documentation-surface gate."""

from __future__ import annotations

from pathlib import Path


def test_ci_lint_job_gates_documentation_surface() -> None:
    """CI must fail if repository documentation-surface findings reappear."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "Audit documentation surface" in workflow
    assert "python tools/audit_documentation_surface.py" in workflow
    assert "--allowlist tools/documentation_surface_allowlist.json" in workflow
    assert "--fail-on-findings" in workflow
