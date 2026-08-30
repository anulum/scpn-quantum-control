# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable implicit-sensitivity quality-gate tests
"""Lock the differentiable implicit-sensitivity gate into preflight and CI."""

from pathlib import Path

from tools import differentiable_implicit_sensitivity_quality_gates as quality_gates
from tools import preflight


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete direct-owner docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-differentiable-implicit-sensitivity-quality"][5:]
        == quality_gates.DIFFERENTIABLE_IMPLICIT_SENSITIVITY_TYPING_RATCHET
    )
    ruff = gates["ruff D differentiable implicit-sensitivity quality ratchet"]
    assert (
        ruff[-len(quality_gates.DIFFERENTIABLE_IMPLICIT_SENSITIVITY_DOCSTRING_RATCHET) :]
        == quality_gates.DIFFERENTIABLE_IMPLICIT_SENSITIVITY_DOCSTRING_RATCHET
    )
    assert "--preview" in ruff and "D,D413,D417,D420" in ruff


def test_coverage_gate_is_isolated_local_and_exact() -> None:
    """Require local implicit-sensitivity execution and exact coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["differentiable implicit-sensitivity focused coverage"]
    report = gates["differentiable implicit-sensitivity exact coverage threshold"]
    assert "--branch" in run
    assert (
        run[-len(quality_gates.DIFFERENTIABLE_IMPLICIT_SENSITIVITY_COVERAGE_COHORT) :]
        == quality_gates.DIFFERENTIABLE_IMPLICIT_SENSITIVITY_COVERAGE_COHORT
    )
    assert quality_gates.DIFFERENTIABLE_IMPLICIT_SENSITIVITY_COVERAGE_DATA_FILE.startswith("/tmp/")
    assert "--fail-under=100" in report
    assert "--include=*/differentiable_implicit_sensitivity.py" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.DIFFERENTIABLE_IMPLICIT_SENSITIVITY_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    assert "gates.extend(DIFFERENTIABLE_IMPLICIT_SENSITIVITY_COVERAGE_GATES)" in Path(
        "tools/preflight.py"
    ).read_text(encoding="utf-8")


def test_ci_runs_and_aggregates_differentiable_implicit_sensitivity_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("  differentiable-implicit-sensitivity-quality:")
    end = workflow.index("\n\n  differentiable-residual-weights-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.DIFFERENTIABLE_IMPLICIT_SENSITIVITY_DOCSTRING_RATCHET:
        assert path in block
    assert "--fail-under=100" in block
    assert quality_gates.DIFFERENTIABLE_IMPLICIT_SENSITIVITY_SOURCE in block
    assert (
        "differentiable-implicit-sensitivity-quality" in workflow[workflow.index("  ci-gate:") :]
    )
