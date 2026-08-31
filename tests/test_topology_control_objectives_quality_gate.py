# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — persistent-H1 topology-control quality-gate tests
"""Lock the topology-control owner into preflight and required CI."""

from pathlib import Path

from tools import preflight
from tools import topology_control_objectives_quality_gates as quality_gates
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete preview NumPy documentation."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-topology-control-quality"][5:]
        == quality_gates.TOPOLOGY_CONTROL_TYPING_RATCHET
    )
    docs = gates["ruff D topology-control quality ratchet"]
    assert "D,D413,D417,D420" in docs
    assert "--preview" in docs
    assert docs[-len(quality_gates.TOPOLOGY_CONTROL_DOCSTRING_RATCHET) :] == (
        quality_gates.TOPOLOGY_CONTROL_DOCSTRING_RATCHET
    )


def test_coverage_gate_runs_real_owner_and_is_exact() -> None:
    """Require the real topology suites and exact combined source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["topology-control focused coverage"]
    report = gates["topology-control exact coverage threshold"]
    assert "--branch" in run
    assert (
        run[-len(quality_gates.TOPOLOGY_CONTROL_TESTS) :] == quality_gates.TOPOLOGY_CONTROL_TESTS
    )
    assert quality_gates.TOPOLOGY_CONTROL_COVERAGE_DATA_FILE.startswith("/tmp/")
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.TOPOLOGY_CONTROL_COVERAGE_INCLUDE}" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper-defined static and coverage commands verbatim in preflight."""
    assert dict(preflight.TOPOLOGY_CONTROL_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text(encoding="utf-8")
    assert "gates.extend(TOPOLOGY_CONTROL_COVERAGE_GATES)" in source


def test_ci_runs_and_aggregates_topology_control_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  topology-control-quality:")
    end = workflow.index("\n\n  quantum-sync-oracle-quality:", start)
    block = workflow[start:end]
    assert all(path in block for path in quality_gates.TOPOLOGY_CONTROL_TYPING_RATCHET)
    assert all(path in block for path in quality_gates.TOPOLOGY_CONTROL_DOCSTRING_RATCHET)
    assert all(path in block for path in quality_gates.TOPOLOGY_CONTROL_TESTS)
    assert quality_gates.TOPOLOGY_CONTROL_COVERAGE_INCLUDE in block
    assert "topology-control-quality" in workflow[workflow.index("  ci-gate:") :]
