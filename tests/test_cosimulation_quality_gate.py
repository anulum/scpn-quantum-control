# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — quantum/classical cosimulation quality-gate tests
"""Lock the quantum/classical cosimulation gate into preflight and CI."""

from pathlib import Path

from tools import cosimulation_quality_gates as quality_gates
from tools import preflight


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete admitted-owner docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-cosimulation-quality"][5:] == quality_gates.COSIMULATION_TYPING_RATCHET
    )
    ruff = gates["ruff D cosimulation quality ratchet"]
    assert (
        ruff[-len(quality_gates.COSIMULATION_DOCSTRING_RATCHET) :]
        == quality_gates.COSIMULATION_DOCSTRING_RATCHET
    )
    assert "--isolated" in ruff and "D,D413" in ruff


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require real cosimulation execution and exact partition coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["cosimulation focused coverage"]
    report = gates["cosimulation exact coverage threshold"]
    assert "--branch" in run
    assert (
        run[-len(quality_gates.COSIMULATION_COVERAGE_COHORT) :]
        == quality_gates.COSIMULATION_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.COSIMULATION_COVERAGE_INCLUDE}" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.COSIMULATION_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text(encoding="utf-8")
    assert "gates.extend(COSIMULATION_COVERAGE_GATES)" in source


def test_ci_runs_and_aggregates_cosimulation_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("  cosimulation-quality:")
    end = workflow.index("\n\n  dla-topology-objectives-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.COSIMULATION_DOCSTRING_RATCHET:
        assert path in block
    for path in quality_gates.COSIMULATION_COVERAGE_COHORT:
        assert path in block
    assert quality_gates.COSIMULATION_COVERAGE_INCLUDE in block
    aggregate = workflow[workflow.index("  ci-gate:") :]
    assert "cosimulation-quality" in aggregate
