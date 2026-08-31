# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — XY compiler quality-gate tests
"""Lock the XY compiler quality gate into preflight and CI."""

from pathlib import Path

from tools import preflight
from tools import xy_compiler_quality_gates as quality_gates
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete connected docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert gates["mypy-strict-XY-compiler-quality"][5:] == (
        quality_gates.XY_COMPILER_TYPING_RATCHET
    )
    ruff = gates["ruff D XY-compiler quality ratchet"]
    assert (
        ruff[-len(quality_gates.XY_COMPILER_DOCSTRING_RATCHET) :]
        == quality_gates.XY_COMPILER_DOCSTRING_RATCHET
    )
    assert "--isolated" in ruff and "D,D413,D417,D420" in ruff


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require real compiler execution and exact source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["XY-compiler focused coverage"]
    report = gates["XY-compiler exact coverage threshold"]
    assert "--branch" in run
    assert run[-len(quality_gates.XY_COMPILER_COVERAGE_COHORT) :] == (
        quality_gates.XY_COMPILER_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert "--include=*/phase/xy_compiler.py" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.XY_COMPILER_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text(encoding="utf-8")
    assert "gates.extend(XY_COMPILER_COVERAGE_GATES)" in source


def test_ci_runs_and_aggregates_xy_compiler_gate() -> None:
    """Keep the focused CI job and transitive aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  xy-compiler-quality:")
    end = workflow.index("\n\n  differentiable-sparse-derivatives-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.XY_COMPILER_DOCSTRING_RATCHET:
        assert path in block
    for path in quality_gates.XY_COMPILER_COVERAGE_COHORT:
        assert path in block
    assert "--fail-under=100" in block
    downstream_start = workflow.index("  differentiable-sparse-derivatives-quality:")
    downstream_end = workflow.index("\n\n  tn-mps-baseline-design-quality:", downstream_start)
    downstream = workflow[downstream_start:downstream_end]
    assert "needs: [lint, varqite-quality, xy-compiler-quality]" in downstream
    aggregate = workflow[workflow.index("  ci-gate:") :]
    assert "differentiable-sparse-derivatives-quality" in aggregate
