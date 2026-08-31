# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — gradient-backend quality-gate tests
"""Lock the gradient-backend quality gate into preflight and CI."""

from pathlib import Path

from tools import gradient_backend_quality_gates as quality_gates
from tools import preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete direct-owner docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-gradient-backend-quality"][5:]
        == quality_gates.GRADIENT_BACKEND_TYPING_RATCHET
    )
    ruff = gates["ruff D gradient-backend quality ratchet"]
    assert (
        ruff[-len(quality_gates.GRADIENT_BACKEND_DOCSTRING_RATCHET) :]
        == quality_gates.GRADIENT_BACKEND_DOCSTRING_RATCHET
    )
    assert "--preview" in ruff and "D,D413,D417,D420" in ruff


def test_coverage_gate_is_isolated_connected_and_exact() -> None:
    """Require connected planner execution and exact source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["gradient-backend focused coverage"]
    report = gates["gradient-backend exact coverage threshold"]
    assert "--branch" in run
    assert run[-len(quality_gates.GRADIENT_BACKEND_COVERAGE_COHORT) :] == (
        quality_gates.GRADIENT_BACKEND_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    include = next(argument for argument in report if argument.startswith("--include="))
    assert "phase/gradient_backend.py" in include
    assert "phase/provider_gradient.py" in include
    assert "phase/qiskit_runtime.py" in include


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.GRADIENT_BACKEND_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    assert "gates.extend(GRADIENT_BACKEND_COVERAGE_GATES)" in Path("tools/preflight.py").read_text(
        encoding="utf-8"
    )


def test_ci_runs_and_aggregates_gradient_backend_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  gradient-backend-quality:")
    end = workflow.index("\n\n  tn-mps-baseline-design-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.GRADIENT_BACKEND_DOCSTRING_RATCHET:
        assert path in block
    assert "--fail-under=100" in block
    assert "phase/gradient_backend.py" in block
    assert "phase/provider_gradient.py" in block
    assert "phase/qiskit_runtime.py" in block
    assert "tests/test_phase_provider_gradient.py" in block
    assert "tests/test_phase_provider_gradient_branches.py" in block
    assert "tests/test_phase_qiskit_runtime.py" in block
    assert "gradient-backend-quality" in workflow[workflow.index("  ci-gate:") :]
