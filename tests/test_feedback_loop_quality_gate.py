# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — feedback-loop quality-gate tests
"""Lock the feedback-loop gate into preflight and CI."""

from pathlib import Path

from tools import feedback_loop_quality_gates as quality_gates
from tools import preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and complete owner NumPy docstrings."""
    realtime_tests = {
        "tests/test_realtime_feedback.py",
        "tests/test_realtime_feedback_fallback_branches.py",
    }
    assert quality_gates.REALTIME_FEEDBACK_SOURCE in quality_gates.FEEDBACK_LOOP_TYPING_RATCHET
    assert quality_gates.REALTIME_FEEDBACK_SOURCE in quality_gates.FEEDBACK_LOOP_DOCSTRING_RATCHET
    assert realtime_tests.issubset(quality_gates.FEEDBACK_LOOP_DOCSTRING_RATCHET)
    assert realtime_tests.issubset(quality_gates.FEEDBACK_LOOP_COVERAGE_COHORT)
    assert "*/control/realtime_feedback.py" in quality_gates.FEEDBACK_LOOP_COVERAGE_INCLUDE
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-feedback-loop-quality"][5:]
        == quality_gates.FEEDBACK_LOOP_TYPING_RATCHET
    )
    ruff = gates["ruff D feedback-loop quality ratchet"]
    assert (
        ruff[-len(quality_gates.FEEDBACK_LOOP_DOCSTRING_RATCHET) :]
        == quality_gates.FEEDBACK_LOOP_DOCSTRING_RATCHET
    )
    assert "--isolated" in ruff and "--preview" in ruff
    assert "D,D413,D417,D420" in ruff
    assert "lint.explicit-preview-rules = true" in ruff


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and exact source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["feedback-loop focused coverage"]
    report = gates["feedback-loop exact coverage threshold"]
    assert "--branch" in run
    assert (
        run[-len(quality_gates.FEEDBACK_LOOP_COVERAGE_COHORT) :]
        == quality_gates.FEEDBACK_LOOP_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.FEEDBACK_LOOP_COVERAGE_INCLUDE}" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.FEEDBACK_LOOP_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text(encoding="utf-8")
    assert "gates.extend(FEEDBACK_LOOP_COVERAGE_GATES)" in source


def test_ci_runs_and_aggregates_feedback_loop_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  feedback-loop-quality:")
    end = workflow.index("\n\n  tn-mps-baseline-design-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.FEEDBACK_LOOP_DOCSTRING_RATCHET:
        assert path in block
    assert "--fail-under=100" in block
    assert quality_gates.FEEDBACK_LOOP_COVERAGE_INCLUDE in block
    assert "feedback-loop-quality" in workflow[workflow.index("  ci-gate:") :]
