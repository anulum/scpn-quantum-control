# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable sparse-derivative quality-gate tests
"""Lock the differentiable sparse-derivative quality gate into preflight and CI."""

from pathlib import Path

from tools import differentiable_sparse_derivatives_quality_gates as quality_gates
from tools import preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete connected docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-differentiable-sparse-derivatives-quality"][5:]
        == quality_gates.DIFFERENTIABLE_SPARSE_DERIVATIVES_TYPING_RATCHET
    )
    ruff = gates["ruff D differentiable-sparse-derivatives quality ratchet"]
    assert (
        ruff[-len(quality_gates.DIFFERENTIABLE_SPARSE_DERIVATIVES_DOCSTRING_RATCHET) :]
        == quality_gates.DIFFERENTIABLE_SPARSE_DERIVATIVES_DOCSTRING_RATCHET
    )
    assert "--isolated" in ruff and "D,D413" in ruff


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require offline sparse-derivative and Fisher exact source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["differentiable-sparse-derivatives focused coverage"]
    report = gates["differentiable-sparse-derivatives exact coverage threshold"]
    assert "--branch" in run
    assert (
        run[-len(quality_gates.DIFFERENTIABLE_SPARSE_DERIVATIVES_COVERAGE_COHORT) :]
        == quality_gates.DIFFERENTIABLE_SPARSE_DERIVATIVES_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert "--include=*/differentiable_sparse_derivatives.py,*/differentiable_fisher.py" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.DIFFERENTIABLE_SPARSE_DERIVATIVES_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text(encoding="utf-8")
    assert "gates.extend(DIFFERENTIABLE_SPARSE_DERIVATIVES_COVERAGE_GATES)" in source


def test_ci_runs_and_aggregates_sparse_derivatives_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  differentiable-sparse-derivatives-quality:")
    end = workflow.index("\n\n  tn-mps-baseline-design-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.DIFFERENTIABLE_SPARSE_DERIVATIVES_DOCSTRING_RATCHET:
        assert path in block
    assert "--fail-under=100" in block
    assert "differentiable_sparse_derivatives.py" in block
    assert "differentiable_fisher.py" in block
    aggregate = workflow[workflow.index("  ci-gate:") :]
    assert "differentiable-sparse-derivatives-quality" in aggregate
