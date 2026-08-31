# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — adaptive-branching quality-gate tests
"""Lock the adaptive-branching owner into preflight and CI."""

from pathlib import Path

from tools import adaptive_branching_quality_gates as quality_gates
from tools import preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and isolated complete NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-adaptive-branching-quality"][5:]
        == quality_gates.ADAPTIVE_BRANCHING_TYPING_RATCHET
    )
    docs = gates["ruff D adaptive-branching quality ratchet"]
    assert "D,D413,D417,D420" in docs
    assert "--preview" in docs
    assert "lint.explicit-preview-rules = true" in docs
    assert docs[-len(quality_gates.ADAPTIVE_BRANCHING_DOCSTRING_RATCHET) :] == (
        quality_gates.ADAPTIVE_BRANCHING_DOCSTRING_RATCHET
    )


def test_coverage_gate_executes_export_and_is_exact() -> None:
    """Require real export execution and exact source branch coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["adaptive-branching focused coverage"]
    report = gates["adaptive-branching exact coverage threshold"]
    assert "--branch" in run
    assert run[-len(quality_gates.ADAPTIVE_BRANCHING_COVERAGE_COHORT) :] == (
        quality_gates.ADAPTIVE_BRANCHING_COVERAGE_COHORT
    )
    assert "tests/test_export_s8_adaptive_branching_readiness.py" in run
    assert quality_gates.ADAPTIVE_BRANCHING_COVERAGE_DATA_FILE.startswith("/tmp/")
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.ADAPTIVE_BRANCHING_COVERAGE_INCLUDE}" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper-defined static and coverage commands verbatim in preflight."""
    assert dict(preflight.ADAPTIVE_BRANCHING_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text(encoding="utf-8")
    assert "gates.extend(ADAPTIVE_BRANCHING_COVERAGE_GATES)" in source


def test_ci_runs_and_aggregates_adaptive_branching_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  adaptive-branching-quality:")
    end = workflow.index("\n\n  quantum-sync-oracle-quality:", start)
    block = workflow[start:end]
    assert all(path in block for path in quality_gates.ADAPTIVE_BRANCHING_TYPING_RATCHET)
    assert all(path in block for path in quality_gates.ADAPTIVE_BRANCHING_DOCSTRING_RATCHET)
    assert all(path in block for path in quality_gates.ADAPTIVE_BRANCHING_COVERAGE_COHORT)
    assert quality_gates.ADAPTIVE_BRANCHING_COVERAGE_INCLUDE in block
    assert "adaptive-branching-quality" in workflow[workflow.index("  ci-gate:") :]
