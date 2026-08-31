# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Application honesty quality-gate tests
"""Lock application honesty quality gates into preflight and CI."""

from tools import application_honesty_quality_gates as quality_gates
from tools import preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gates_cover_typing_docs_and_evidence() -> None:
    """Require strict typing, NumPy docstrings, and evidence freshness."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-application-honesty-quality"][5:]
        == quality_gates.APPLICATION_HONESTY_QUALITY_RATCHET
    )
    docstring = gates["ruff D application-honesty quality ratchet"]
    assert "--preview" in docstring
    assert "D,D107,D413,D417,D420" in docstring
    assert gates["application-honesty evidence drift"][-1] == "--check"


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and exact source-only coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["application-honesty focused coverage"]
    assert "--branch" in run
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert (
        run[-len(quality_gates.APPLICATION_HONESTY_COVERAGE_COHORT) :]
        == quality_gates.APPLICATION_HONESTY_COVERAGE_COHORT
    )
    threshold = gates["application-honesty exact coverage threshold"]
    assert any(argument.startswith("--data-file=/tmp/") for argument in threshold)
    assert "--fail-under=100" in threshold
    assert f"--include={quality_gates.APPLICATION_HONESTY_COVERAGE_INCLUDE}" in threshold


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.APPLICATION_HONESTY_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command


def test_ci_runs_and_aggregates_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  application-honesty-quality:")
    end = workflow.index("\n\n  layout-method-comparison-quality:", start)
    block = workflow[start:end]
    assert all(path in block for path in quality_gates.APPLICATION_HONESTY_QUALITY_RATCHET)
    assert all(path in block for path in quality_gates.APPLICATION_HONESTY_COVERAGE_COHORT)
    assert quality_gates.APPLICATION_HONESTY_COVERAGE_INCLUDE in block
    assert "application-honesty-quality" in workflow[workflow.index("  ci-gate:") :]
