# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — DynQ layout quality-gate tests
"""Lock the connected DynQ layout quality gate into preflight and CI."""

from pathlib import Path

from tools import dynq_layout_quality_gates as quality_gates
from tools import preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete direct-owner docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert gates["mypy-strict-dynq-layout-quality"][5:] == quality_gates.DYNQ_LAYOUT_TYPING_RATCHET
    ruff = gates["ruff D DynQ layout quality ratchet"]
    assert (
        ruff[-len(quality_gates.DYNQ_LAYOUT_DOCSTRING_RATCHET) :]
        == quality_gates.DYNQ_LAYOUT_DOCSTRING_RATCHET
    )
    assert "--preview" in ruff and "D,D413,D417,D420" in ruff


def test_coverage_gate_is_isolated_connected_and_exact() -> None:
    """Require real mapper/pass execution and exact source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["DynQ layout focused coverage"]
    report = gates["DynQ layout exact coverage threshold"]
    assert "--branch" in run
    assert (
        run[-len(quality_gates.DYNQ_LAYOUT_COVERAGE_COHORT) :]
        == quality_gates.DYNQ_LAYOUT_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert "--include=*/hardware/qubit_mapper.py,*/hardware/dynq_layout_pass.py" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.DYNQ_LAYOUT_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    assert "gates.extend(DYNQ_LAYOUT_COVERAGE_GATES)" in Path("tools/preflight.py").read_text(
        encoding="utf-8"
    )


def test_ci_runs_and_aggregates_dynq_layout_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  dynq-layout-quality:")
    end = workflow.index("\n\n  topology-kernel-classifier-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.DYNQ_LAYOUT_DOCSTRING_RATCHET:
        assert path in block
    assert "--fail-under=100" in block
    for source in (
        quality_gates.QUBIT_MAPPER_SOURCE,
        quality_gates.DYNQ_LAYOUT_PASS_SOURCE,
    ):
        assert source in block
    aggregate = workflow[workflow.index("  ci-gate:") :]
    assert "dynq-layout-quality" in aggregate
