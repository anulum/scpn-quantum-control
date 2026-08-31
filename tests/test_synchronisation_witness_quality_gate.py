# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — synchronisation witness quality-gate tests
"""Lock synchronisation-witness quality gates into preflight and CI."""

from tools import preflight
from tools import synchronisation_witness_quality_gates as quality_gates
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gates_cover_typing_and_docs() -> None:
    """Require strict typing and complete NumPy docs for the owned cohorts."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-synchronisation-witness-quality"][5:]
        == quality_gates.SYNCHRONISATION_WITNESS_TYPING_RATCHET
    )
    ruff = gates["ruff D synchronisation-witness quality ratchet"]
    assert (
        ruff[-len(quality_gates.SYNCHRONISATION_WITNESS_DOCSTRING_RATCHET) :]
        == quality_gates.SYNCHRONISATION_WITNESS_DOCSTRING_RATCHET
    )
    assert "--preview" in ruff and "D,D413,D417,D420" in ruff


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require connected branch execution and exact source-only coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["synchronisation-witness focused coverage"]
    assert "--branch" in run
    assert (
        run[-len(quality_gates.SYNCHRONISATION_WITNESS_COVERAGE_COHORT) :]
        == quality_gates.SYNCHRONISATION_WITNESS_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    threshold = gates["synchronisation-witness exact coverage threshold"]
    assert "--fail-under=100" in threshold
    assert any("executive_analyse.py" in argument for argument in threshold)


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.SYNCHRONISATION_WITNESS_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command


def test_ci_runs_and_aggregates_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  synchronisation-witness-quality:")
    end = workflow.index("\n\n  experiment-mitigation-quality:", start)
    block = workflow[start:end]
    assert all(path in block for path in quality_gates.SYNCHRONISATION_WITNESS_TYPING_RATCHET)
    assert all(path in block for path in quality_gates.SYNCHRONISATION_WITNESS_COVERAGE_COHORT)
    assert "--fail-under=100" in block
    assert "synchronisation-witness-quality" in workflow[workflow.index("  ci-gate:") :]
