# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — open-system-objective quality-gate tests
"""Lock open-system-objective quality gates into preflight and CI."""

from tools import open_system_objective_quality_gates as quality_gates
from tools import preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and isolated NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert gates["mypy-strict-open-system-objective-quality"][5:] == (
        quality_gates.OPEN_SYSTEM_OBJECTIVE_QUALITY_RATCHET
    )
    assert "D,D413" in gates["ruff D open-system-objective quality ratchet"]


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and exact two-source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["open-system-objective focused coverage"]
    report = gates["open-system-objective exact coverage threshold"]
    assert "--branch" in run
    assert run[-2:] == quality_gates.OPEN_SYSTEM_OBJECTIVE_COVERAGE_COHORT
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.OPEN_SYSTEM_OBJECTIVE_COVERAGE_INCLUDE}" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.OPEN_SYSTEM_OBJECTIVE_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command


def test_ci_runs_and_aggregates_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  open-system-objective-quality:")
    end = workflow.index("\n\n  external-validation-manifest-quality:", start)
    block = workflow[start:end]
    assert all(path in block for path in quality_gates.OPEN_SYSTEM_OBJECTIVE_QUALITY_RATCHET)
    assert quality_gates.OPEN_SYSTEM_OBJECTIVE_COVERAGE_DATA_FILE in block
    assert "open-system-objective-quality" in workflow[workflow.index("  ci-gate:") :]
