# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — resource-budget quality-gate tests
"""Lock the resource-budget gate into preflight and CI."""

from pathlib import Path

from tools import preflight
from tools import resource_budget_gate_quality_gates as quality_gates


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and isolated NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-resource-budget-gate-quality"][5:]
        == quality_gates.RESOURCE_BUDGET_GATE_QUALITY_RATCHET
    )
    ruff = gates["ruff D resource-budget-gate quality ratchet"]
    assert "--isolated" in ruff and "--preview" in ruff
    assert "D,D413,D417,D420" in ruff


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and exact source-only coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["resource-budget-gate focused coverage"]
    report = gates["resource-budget-gate exact coverage threshold"]
    assert "--branch" in run
    assert run[-len(quality_gates.RESOURCE_BUDGET_GATE_COVERAGE_COHORT) :] == (
        quality_gates.RESOURCE_BUDGET_GATE_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert "--include=*/compile_budget.py,*/resource_budget_gate.py" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.RESOURCE_BUDGET_GATE_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command


def test_ci_runs_and_aggregates_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text()
    start = workflow.index("  resource-budget-gate-quality:")
    end = workflow.index("\n\n  decisive-advantage-quality:", start)
    block = workflow[start:end]
    assert all(path in block for path in quality_gates.RESOURCE_BUDGET_GATE_QUALITY_RATCHET)
    assert all(path in block for path in quality_gates.RESOURCE_BUDGET_GATE_COVERAGE_COHORT)
    assert "--include=*/compile_budget.py,*/resource_budget_gate.py" in block
    assert "resource-budget-gate-quality" in workflow[workflow.index("  ci-gate:") :]
