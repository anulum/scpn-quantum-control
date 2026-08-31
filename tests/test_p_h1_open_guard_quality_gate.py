# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — p_h1 open-claim guard quality-gate tests
"""Lock the p_h1 open-claim guard quality gate into preflight and CI."""

from pathlib import Path

from tools import p_h1_open_guard_quality_gates as quality_gates
from tools import preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_scopes_shared_cli_debt() -> None:
    """Require strict typing while keeping broad CLI docs out of this owner."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-p-h1-open-guard-quality"][5:]
        == quality_gates.P_H1_OPEN_GUARD_TYPING_RATCHET
    )
    ruff = gates["ruff D p_h1 open-guard quality ratchet"]
    assert (
        ruff[-len(quality_gates.P_H1_OPEN_GUARD_DOCSTRING_RATCHET) :]
        == quality_gates.P_H1_OPEN_GUARD_DOCSTRING_RATCHET
    )
    assert quality_gates.P_H1_OPEN_GUARD_SHARED_CLI_TEST not in ruff
    assert "--preview" in ruff and "D,D413,D417,D420" in ruff


def test_coverage_gate_is_isolated_connected_and_exact() -> None:
    """Require real guard/CLI execution and exact source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["p_h1 open-guard focused coverage"]
    report = gates["p_h1 open-guard exact coverage threshold"]
    assert "--branch" in run
    assert (
        run[-len(quality_gates.P_H1_OPEN_GUARD_COVERAGE_COHORT) :]
        == quality_gates.P_H1_OPEN_GUARD_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.P_H1_OPEN_GUARD_COVERAGE_INCLUDE}" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.P_H1_OPEN_GUARD_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    assert "gates.extend(P_H1_OPEN_GUARD_COVERAGE_GATES)" in Path("tools/preflight.py").read_text(
        encoding="utf-8"
    )


def test_ci_runs_and_aggregates_p_h1_open_guard_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  p-h1-open-guard-quality:")
    end = workflow.index("\n\n  studio-scorecard-bundle-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.P_H1_OPEN_GUARD_DOCSTRING_RATCHET:
        assert path in block
    assert quality_gates.P_H1_OPEN_GUARD_SHARED_CLI_TEST in block
    assert quality_gates.PERSISTENT_HOMOLOGY_BRANCH_TEST in block
    assert quality_gates.PERSISTENT_HOMOLOGY_CONNECTED_TEST in block
    assert "--fail-under=100" in block
    assert quality_gates.P_H1_OPEN_GUARD_COVERAGE_INCLUDE in block
    aggregate = workflow[workflow.index("  ci-gate:") :]
    assert "p-h1-open-guard-quality" in aggregate
