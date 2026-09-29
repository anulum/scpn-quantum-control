# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — qualification projection quality-gate lock
"""Lock the qualification projection quality gate into preflight and CI."""

from pathlib import Path

from tools import preflight
from tools import qualification_projection_quality_gates as quality_gates
from tools.ci_workflow_inventory import read_ci_workflow_source

ROOT = Path(__file__).resolve().parents[1]


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and isolated NumPy docstrings over the whole ratchet."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-qualification-projection-quality"][5:]
        == quality_gates.QUALIFICATION_PROJECTION_QUALITY_RATCHET
    )
    ruff = gates["ruff D qualification-projection quality ratchet"]
    assert "--isolated" in ruff and "--preview" in ruff
    assert "D,D413,D417,D420" in ruff
    assert all(
        (ROOT / path).is_file() for path in quality_gates.QUALIFICATION_PROJECTION_QUALITY_RATCHET
    )


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and exact source-only coverage of all four owners."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["qualification-projection focused coverage"]
    report = gates["qualification-projection exact coverage threshold"]
    assert "--branch" in run
    assert run[-len(quality_gates.QUALIFICATION_PROJECTION_COVERAGE_COHORT) :] == (
        quality_gates.QUALIFICATION_PROJECTION_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.QUALIFICATION_PROJECTION_COVERAGE_INCLUDE}" in report
    assert set(quality_gates.QUALIFICATION_PROJECTION_COVERAGE_COHORT) <= set(
        quality_gates.QUALIFICATION_PROJECTION_QUALITY_RATCHET
    )


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.QUALIFICATION_PROJECTION_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command


def test_ci_runs_and_aggregates_gate() -> None:
    """Keep the focused CI job, its exact cohort and the aggregate dependency."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  qualification-projection-quality:")
    end = workflow.find("\n\n  ", start)
    block = workflow[start : len(workflow) if end == -1 else end]
    assert all(path in block for path in quality_gates.QUALIFICATION_PROJECTION_QUALITY_RATCHET)
    assert all(path in block for path in quality_gates.QUALIFICATION_PROJECTION_COVERAGE_COHORT)
    assert f"--include={quality_gates.QUALIFICATION_PROJECTION_COVERAGE_INCLUDE}" in block
    assert "--fail-under=100" in block
    assert "qualification-projection-quality" in workflow[workflow.index("  ci-gate:") :]
