# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — competitive-baseline-watch quality-gate tests
"""Lock the competitive-baseline-watch gate into preflight and CI."""

from pathlib import Path

from tools import competitive_baseline_watch_quality_gates as quality_gates
from tools import preflight


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and isolated NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-competitive-baseline-watch-quality"][5:]
        == quality_gates.COMPETITIVE_BASELINE_WATCH_QUALITY_RATCHET
    )
    assert "D,D413" in gates["ruff D competitive-baseline-watch quality ratchet"]


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and exact joint source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["competitive-baseline-watch focused coverage"]
    report = gates["competitive-baseline-watch exact coverage threshold"]
    assert "--branch" in run
    assert run[-len(quality_gates.COMPETITIVE_BASELINE_WATCH_COVERAGE_COHORT) :] == (
        quality_gates.COMPETITIVE_BASELINE_WATCH_COVERAGE_COHORT
    )
    assert "--fail-under=100" in report
    assert (
        "--include=*/competitive_baseline_watch.py,*/benchmarks/reproducible_comparison.py,*/benchmarks/kuramoto_competitive_types.py"
        in report
    )


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.COMPETITIVE_BASELINE_WATCH_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command


def test_ci_runs_and_aggregates_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text()
    start = workflow.index("  competitive-baseline-watch-quality:")
    end = workflow.index("\n\n  decisive-advantage-quality:", start)
    block = workflow[start:end]
    assert all(path in block for path in quality_gates.COMPETITIVE_BASELINE_WATCH_QUALITY_RATCHET)
    assert all(path in block for path in quality_gates.COMPETITIVE_BASELINE_WATCH_COVERAGE_COHORT)
    assert "competitive-baseline-watch-quality" in workflow[workflow.index("  ci-gate:") :]
