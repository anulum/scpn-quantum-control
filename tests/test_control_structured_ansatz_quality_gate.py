# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — control StructuredAnsatz quality-gate tests
"""Lock the control StructuredAnsatz quality gate into preflight and CI."""

from pathlib import Path

from tools import control_structured_ansatz_quality_gates as quality_gates
from tools import preflight


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete direct-owner docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-control-structured-ansatz-quality"][5:]
        == quality_gates.CONTROL_STRUCTURED_ANSATZ_TYPING_RATCHET
    )
    ruff = gates["ruff D control-StructuredAnsatz quality ratchet"]
    assert (
        ruff[-len(quality_gates.CONTROL_STRUCTURED_ANSATZ_DOCSTRING_RATCHET) :]
        == quality_gates.CONTROL_STRUCTURED_ANSATZ_DOCSTRING_RATCHET
    )
    assert "--preview" in ruff and "D,D107,D413,D417,D420" in ruff


def test_coverage_gate_is_isolated_connected_and_exact() -> None:
    """Require real Qiskit execution and exact source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["control-StructuredAnsatz focused coverage"]
    report = gates["control-StructuredAnsatz exact coverage threshold"]
    assert "--branch" in run
    assert (
        run[-len(quality_gates.CONTROL_STRUCTURED_ANSATZ_COVERAGE_COHORT) :]
        == quality_gates.CONTROL_STRUCTURED_ANSATZ_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert "--include=*/control/structured_ansatz.py,*/phase/structured_ansatz.py" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.CONTROL_STRUCTURED_ANSATZ_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text(encoding="utf-8")
    assert "gates.extend(CONTROL_STRUCTURED_ANSATZ_COVERAGE_GATES)" in source


def test_ci_runs_and_aggregates_control_structured_ansatz_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("  control-structured-ansatz-quality:")
    end = workflow.index("\n\n  phase-results-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.CONTROL_STRUCTURED_ANSATZ_DOCSTRING_RATCHET:
        assert path in block
    for path in quality_gates.CONTROL_STRUCTURED_ANSATZ_COVERAGE_COHORT:
        assert path in block
    assert "--fail-under=100" in block
    for path in quality_gates.STRUCTURED_ANSATZ_SOURCES:
        assert path in block
    aggregate = workflow[workflow.index("  ci-gate:") :]
    assert "control-structured-ansatz-quality" in aggregate
