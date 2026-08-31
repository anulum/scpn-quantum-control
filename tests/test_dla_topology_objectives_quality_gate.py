# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — DLA topology-objective quality-gate tests
"""Lock the DLA topology-objective quality gate into preflight and CI."""

from pathlib import Path

from tools import dla_topology_objectives_quality_gates as quality_gates
from tools import preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete connected docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-dla-topology-objectives-quality"][5:]
        == quality_gates.DLA_TOPOLOGY_OBJECTIVES_TYPING_RATCHET
    )
    ruff = gates["ruff D DLA topology-objectives quality ratchet"]
    assert (
        ruff[-len(quality_gates.DLA_TOPOLOGY_OBJECTIVES_DOCSTRING_RATCHET) :]
        == quality_gates.DLA_TOPOLOGY_OBJECTIVES_DOCSTRING_RATCHET
    )
    assert "--isolated" in ruff and "D,D413" in ruff


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require offline objective execution and exact source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["DLA topology-objectives focused coverage"]
    report = gates["DLA topology-objectives exact coverage threshold"]
    assert "--branch" in run
    assert (
        run[-len(quality_gates.DLA_TOPOLOGY_OBJECTIVES_COVERAGE_COHORT) :]
        == quality_gates.DLA_TOPOLOGY_OBJECTIVES_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert "--include=*/dla_topology_control/objectives.py" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.DLA_TOPOLOGY_OBJECTIVES_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text(encoding="utf-8")
    assert "gates.extend(DLA_TOPOLOGY_OBJECTIVES_COVERAGE_GATES)" in source


def test_ci_runs_and_aggregates_dla_topology_objectives_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  dla-topology-objectives-quality:")
    end = workflow.index("\n\n  tn-mps-baseline-design-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.DLA_TOPOLOGY_OBJECTIVES_DOCSTRING_RATCHET:
        assert path in block
    assert "--fail-under=100" in block
    assert "dla_topology_control/objectives.py" in block
    aggregate = workflow[workflow.index("  ci-gate:") :]
    assert "dla-topology-objectives-quality" in aggregate
