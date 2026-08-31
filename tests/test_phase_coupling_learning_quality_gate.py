# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — phase coupling-learning quality-gate tests
"""Lock the phase coupling-learning owner into preflight and CI."""

from pathlib import Path

from tools import phase_coupling_learning_quality_gates as quality_gates
from tools import preflight


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and isolated complete NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-phase-coupling-learning-quality"][5:]
        == quality_gates.PHASE_COUPLING_LEARNING_TYPING_RATCHET
    )
    docs = gates["ruff D phase-coupling-learning quality ratchet"]
    assert "D,D413,D417,D420" in docs
    assert "--preview" in docs
    assert "lint.explicit-preview-rules = true" in docs
    assert docs[-len(quality_gates.PHASE_COUPLING_LEARNING_DOCSTRING_RATCHET) :] == (
        quality_gates.PHASE_COUPLING_LEARNING_DOCSTRING_RATCHET
    )


def test_coverage_gate_runs_real_simulator_suite_and_is_exact() -> None:
    """Require public simulator execution and exact source branch coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["phase-coupling-learning focused coverage"]
    report = gates["phase-coupling-learning exact coverage threshold"]
    assert "--branch" in run
    assert run[-1:] == quality_gates.PHASE_COUPLING_LEARNING_COVERAGE_COHORT
    assert quality_gates.PHASE_COUPLING_LEARNING_COVERAGE_DATA_FILE.startswith("/tmp/")
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.PHASE_COUPLING_LEARNING_COVERAGE_INCLUDE}" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper-defined static and coverage commands verbatim in preflight."""
    assert dict(preflight.PHASE_COUPLING_LEARNING_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text(encoding="utf-8")
    assert "gates.extend(PHASE_COUPLING_LEARNING_COVERAGE_GATES)" in source


def test_ci_runs_and_aggregates_phase_coupling_learning_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("  phase-coupling-learning-quality:")
    end = workflow.index("\n\n  quantum-sync-oracle-quality:", start)
    block = workflow[start:end]
    assert all(path in block for path in quality_gates.PHASE_COUPLING_LEARNING_TYPING_RATCHET)
    assert all(path in block for path in quality_gates.PHASE_COUPLING_LEARNING_DOCSTRING_RATCHET)
    assert quality_gates.PHASE_COUPLING_LEARNING_TEST in block
    assert quality_gates.PHASE_COUPLING_LEARNING_COVERAGE_INCLUDE in block
    assert "phase-coupling-learning-quality" in workflow[workflow.index("  ci-gate:") :]
