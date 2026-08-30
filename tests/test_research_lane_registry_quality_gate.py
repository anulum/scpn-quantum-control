# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — research-lane registry quality-gate tests
"""Lock the research-lane registry gate into preflight and CI."""

from pathlib import Path

from tools import preflight
from tools import research_lane_registry_quality_gates as quality_gates


def test_static_gates_cover_typing_docs_and_evidence_drift() -> None:
    """Require strict typing, NumPy docstrings, and committed evidence parity."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-research-lane-registry-quality"][5:]
        == quality_gates.RESEARCH_LANE_REGISTRY_QUALITY_RATCHET
    )
    ruff = gates["ruff D research-lane-registry quality ratchet"]
    assert "--preview" in ruff and "D,D413,D417,D420" in ruff
    assert gates["research-lane-registry evidence drift"][-1] == "--check"
    assert gates["RL research-governance evidence drift"][-1] == "--check"


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and exact source-only coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["research-lane-registry focused coverage"]
    report = gates["research-lane-registry exact coverage threshold"]
    assert "--branch" in run
    assert run[-len(quality_gates.RESEARCH_LANE_REGISTRY_COVERAGE_COHORT) :] == (
        quality_gates.RESEARCH_LANE_REGISTRY_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.RESEARCH_LANE_REGISTRY_COVERAGE_INCLUDE}" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.RESEARCH_LANE_REGISTRY_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command


def test_ci_runs_and_aggregates_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text()
    start = workflow.index("  research-lane-registry-quality:")
    end = workflow.index("\n\n  theory-hook-promotion-quality:", start)
    block = workflow[start:end]
    assert all(path in block for path in quality_gates.RESEARCH_LANE_REGISTRY_QUALITY_RATCHET)
    assert "scripts/run_research_lane_registry.py --check" in block
    assert "scripts/run_rl_research_governance_evidence.py --check" in block
    assert quality_gates.RESEARCH_LANE_REGISTRY_COVERAGE_INCLUDE in block
    assert "tests/test_frontier_interface_guards.py" in block
    assert "research-lane-registry-quality" in workflow[workflow.index("  ci-gate:") :]
