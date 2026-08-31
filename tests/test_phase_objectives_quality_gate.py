# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — phase-objectives quality-gate tests
"""Lock the phase-objectives owner into preflight and required CI."""

from pathlib import Path

from tools import phase_objectives_quality_gates as quality_gates
from tools import preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and isolated complete NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-phase-objectives-quality"][5:]
        == quality_gates.PHASE_OBJECTIVES_TYPING_RATCHET
    )
    docs = gates["ruff D phase-objectives quality ratchet"]
    assert "D,D413,D417,D420" in docs
    assert "--preview" in docs
    assert "lint.explicit-preview-rules = true" in docs
    assert docs[-len(quality_gates.PHASE_OBJECTIVES_DOCSTRING_RATCHET) :] == (
        quality_gates.PHASE_OBJECTIVES_DOCSTRING_RATCHET
    )


def test_coverage_gate_runs_real_objective_suite_and_is_exact() -> None:
    """Require public objective execution and exact source branch coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["phase-objectives focused coverage"]
    report = gates["phase-objectives exact coverage threshold"]
    assert "--branch" in run
    assert run[-1:] == quality_gates.PHASE_OBJECTIVES_COVERAGE_COHORT
    assert quality_gates.PHASE_OBJECTIVES_COVERAGE_DATA_FILE.startswith("/tmp/")
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.PHASE_OBJECTIVES_COVERAGE_INCLUDE}" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper-defined static and coverage commands verbatim in preflight."""
    assert dict(preflight.PHASE_OBJECTIVES_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text(encoding="utf-8")
    assert "gates.extend(PHASE_OBJECTIVES_COVERAGE_GATES)" in source


def test_ci_runs_and_aggregates_phase_objectives_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  phase-objectives-quality:")
    end = workflow.index("\n\n  quantum-sync-oracle-quality:", start)
    block = workflow[start:end]
    assert all(path in block for path in quality_gates.PHASE_OBJECTIVES_TYPING_RATCHET)
    assert all(path in block for path in quality_gates.PHASE_OBJECTIVES_DOCSTRING_RATCHET)
    assert quality_gates.PHASE_OBJECTIVES_TEST in block
    assert quality_gates.PHASE_OBJECTIVES_COVERAGE_INCLUDE in block
    assert "phase-objectives-quality" in workflow[workflow.index("  ci-gate:") :]
