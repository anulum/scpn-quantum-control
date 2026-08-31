# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable-notebook-curriculum quality-gate tests
"""Lock the differentiable notebook curriculum gate into preflight and CI."""

from tools import differentiable_notebook_curriculum_quality_gates as quality_gates
from tools import preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and isolated NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-differentiable-notebook-curriculum"][5:]
        == quality_gates.DIFFERENTIABLE_NOTEBOOK_CURRICULUM_QUALITY_RATCHET
    )
    assert "D,D413" in gates["ruff D differentiable-notebook-curriculum ratchet"]


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and exact source-only coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    assert "--branch" in gates["differentiable-notebook-curriculum focused coverage"]
    assert (
        "--fail-under=100" in gates["differentiable-notebook-curriculum exact coverage threshold"]
    )


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.DIFFERENTIABLE_NOTEBOOK_CURRICULUM_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command


def test_ci_runs_and_aggregates_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  differentiable-notebook-curriculum-quality:")
    end = workflow.index("\n\n  decisive-advantage-quality:", start)
    block = workflow[start:end]
    assert all(
        path in block for path in quality_gates.DIFFERENTIABLE_NOTEBOOK_CURRICULUM_QUALITY_RATCHET
    )
    assert "differentiable-notebook-curriculum-quality" in workflow[workflow.index("  ci-gate:") :]
