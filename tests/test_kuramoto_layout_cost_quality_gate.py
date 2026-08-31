# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Kuramoto layout-cost quality-gate tests
"""Lock layout-cost quality gates into preflight and CI."""

from tools import kuramoto_layout_cost_quality_gates as quality_gates
from tools import preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gates_cover_typing_and_docs() -> None:
    """Require strict typing and NumPy docstrings for the owned cohort."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-kuramoto-layout-cost-quality"][5:]
        == quality_gates.KURAMOTO_LAYOUT_COST_QUALITY_RATCHET
    )
    assert "D,D413" in gates["ruff D kuramoto-layout-cost quality ratchet"]


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and exact source-only coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    assert "--branch" in gates["kuramoto-layout-cost focused coverage"]
    threshold = gates["kuramoto-layout-cost exact coverage threshold"]
    assert "--fail-under=100" in threshold
    assert "--include=*/hardware/kuramoto_layout_cost.py" in threshold


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.KURAMOTO_LAYOUT_COST_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command


def test_ci_runs_and_aggregates_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  kuramoto-layout-cost-quality:")
    end = workflow.index("\n\n  layout-method-comparison-quality:", start)
    block = workflow[start:end]
    assert all(path in block for path in quality_gates.KURAMOTO_LAYOUT_COST_QUALITY_RATCHET)
    assert "kuramoto-layout-cost-quality" in workflow[workflow.index("  ci-gate:") :]
