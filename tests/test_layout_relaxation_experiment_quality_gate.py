# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — layout-relaxation-experiment quality-gate tests
"""Lock the layout-relaxation-experiment quality gate into preflight and CI."""

from pathlib import Path

from tools import layout_relaxation_experiment_quality_gates as quality_gates
from tools import preflight


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete experiment-owner docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-layout-relaxation-experiment-quality"][5:]
        == quality_gates.LAYOUT_RELAXATION_EXPERIMENT_TYPING_RATCHET
    )
    ruff = gates["ruff D layout-relaxation-experiment quality ratchet"]
    assert (
        ruff[-len(quality_gates.LAYOUT_RELAXATION_EXPERIMENT_DOCSTRING_RATCHET) :]
        == quality_gates.LAYOUT_RELAXATION_EXPERIMENT_DOCSTRING_RATCHET
    )
    assert "--preview" in ruff and "D,D413,D417,D420" in ruff


def test_coverage_gate_is_stubbed_and_exact() -> None:
    """Require stubbed experiment execution and exact source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["layout-relaxation-experiment focused coverage"]
    report = gates["layout-relaxation-experiment exact coverage threshold"]
    assert "--branch" in run
    assert run[-1:] == quality_gates.LAYOUT_RELAXATION_EXPERIMENT_COVERAGE_COHORT
    assert "tests/test_run_layout_relaxation_experiment.py" not in run
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert "--include=*/benchmarks/layout_relaxation_experiment.py" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.LAYOUT_RELAXATION_EXPERIMENT_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    assert "gates.extend(LAYOUT_RELAXATION_EXPERIMENT_COVERAGE_GATES)" in Path(
        "tools/preflight.py"
    ).read_text(encoding="utf-8")


def test_ci_runs_and_aggregates_layout_relaxation_experiment_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("  layout-relaxation-experiment-quality:")
    end = workflow.index("\n\n  phase-results-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.LAYOUT_RELAXATION_EXPERIMENT_DOCSTRING_RATCHET:
        assert path in block
    assert "--fail-under=100" in block
    assert "benchmarks/layout_relaxation_experiment.py" in block
    assert "layout-relaxation-experiment-quality" in workflow[workflow.index("  ci-gate:") :]
