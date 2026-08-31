# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — control-stack compose quality-gate tests
"""Lock the control-stack compose gate into preflight and CI."""

from __future__ import annotations

from tools import control_stack_compose_product_quality_gates as quality_gates
from tools import preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and isolated NumPy docstrings for the cohort."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    mypy = gates["mypy-strict-control-stack-compose-quality"]
    ruff = gates["ruff D control-stack-compose quality ratchet"]
    assert mypy[5:] == quality_gates.CONTROL_STACK_COMPOSE_QUALITY_RATCHET
    assert (
        ruff[-len(quality_gates.CONTROL_STACK_COMPOSE_QUALITY_RATCHET) :]
        == quality_gates.CONTROL_STACK_COMPOSE_QUALITY_RATCHET
    )
    assert "--isolated" in ruff and "D,D413" in ruff


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and a 100 percent source-only report."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["control-stack-compose focused coverage"]
    report = gates["control-stack-compose exact coverage threshold"]
    assert f"--data-file={quality_gates.CONTROL_STACK_COMPOSE_COVERAGE_DATA_FILE}" in run
    assert "--branch" in run
    assert run[-len(quality_gates.CONTROL_STACK_COMPOSE_COVERAGE_COHORT) :] == (
        quality_gates.CONTROL_STACK_COMPOSE_COVERAGE_COHORT
    )
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.CONTROL_STACK_COMPOSE_COVERAGE_INCLUDE}" in report


def test_preflight_uses_the_helper_defined_gates() -> None:
    """Keep helper-defined static and coverage commands verbatim in preflight."""
    static = dict(preflight.STATIC_GATES)
    coverage = dict(preflight.CONTROL_STACK_COMPOSE_COVERAGE_GATES)
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert static[name] == command
    assert coverage == dict(quality_gates.build_coverage_gates(preflight._PY))


def test_ci_runs_and_aggregates_the_product_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  control-stack-compose-quality:")
    end = workflow.index("\n\n  public-api-stability-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.CONTROL_STACK_COMPOSE_QUALITY_RATCHET:
        assert path in block
    assert "--fail-under=100" in block
    assert "tests/test_control_stack_runtime_adapters.py" in block
    assert quality_gates.CONTROL_STACK_COMPOSE_COVERAGE_INCLUDE in block
    assert "control-stack-compose-quality" in workflow[workflow.index("  ci-gate:") :]
