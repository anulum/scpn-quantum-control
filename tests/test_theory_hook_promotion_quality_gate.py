# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — theory hook promotion quality-gate tests
"""Lock the theory-hook-promotion gate into preflight and CI."""

from tools import preflight
from tools import theory_hook_promotion_quality_gates as quality_gates
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gates_cover_typing_docs_and_evidence_drift() -> None:
    """Require strict typing, NumPy docstrings, and committed evidence parity."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-theory-hook-promotion-quality"][5:]
        == quality_gates.THEORY_HOOK_PROMOTION_QUALITY_RATCHET
    )
    ruff = gates["ruff D theory-hook-promotion quality ratchet"]
    assert "--preview" in ruff and "D,D413,D417" in ruff
    assert gates["theory-hook-promotion evidence drift"][-1] == "--check"


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and exact source-only coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["theory-hook-promotion focused coverage"]
    report = gates["theory-hook-promotion exact coverage threshold"]
    assert "--branch" in run
    assert run[-len(quality_gates.THEORY_HOOK_PROMOTION_COVERAGE_COHORT) :] == (
        quality_gates.THEORY_HOOK_PROMOTION_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.THEORY_HOOK_PROMOTION_COVERAGE_INCLUDE}" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.THEORY_HOOK_PROMOTION_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command


def test_ci_runs_and_aggregates_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  theory-hook-promotion-quality:")
    end = workflow.index("\n\n  scorecard-acceptance-engine-quality:", start)
    block = workflow[start:end]
    assert all(path in block for path in quality_gates.THEORY_HOOK_PROMOTION_QUALITY_RATCHET)
    assert "scripts/run_theory_hook_promotion_evidence.py --check" in block
    assert quality_gates.THEORY_HOOK_PROMOTION_COVERAGE_INCLUDE in block
    assert "theory-hook-promotion-quality" in workflow[workflow.index("  ci-gate:") :]
