# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable audit-contract quality-gate tests
"""Lock the differentiable audit-contract gate into preflight and CI."""

from pathlib import Path

from tools import differentiable_audit_contracts_quality_gates as quality_gates
from tools import preflight


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and complete connected NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert gates["mypy-strict-differentiable-audit-contracts-quality"][5:] == (
        quality_gates.DIFFERENTIABLE_AUDIT_CONTRACTS_TYPING_RATCHET
    )
    ruff = gates["ruff D differentiable-audit-contracts quality ratchet"]
    assert ruff[-len(quality_gates.DIFFERENTIABLE_AUDIT_CONTRACTS_DOCSTRING_RATCHET) :] == (
        quality_gates.DIFFERENTIABLE_AUDIT_CONTRACTS_DOCSTRING_RATCHET
    )
    assert "--isolated" in ruff and "D,D413" in ruff


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require connected audit execution and exact source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["differentiable-audit-contracts focused coverage"]
    report = gates["differentiable-audit-contracts exact coverage threshold"]
    assert "--branch" in run
    assert run[-len(quality_gates.DIFFERENTIABLE_AUDIT_CONTRACTS_COVERAGE_COHORT) :] == (
        quality_gates.DIFFERENTIABLE_AUDIT_CONTRACTS_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert "--include=*/phase/differentiable_audit_contracts.py" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.DIFFERENTIABLE_AUDIT_CONTRACTS_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text(encoding="utf-8")
    assert "gates.extend(DIFFERENTIABLE_AUDIT_CONTRACTS_COVERAGE_GATES)" in source


def test_ci_runs_and_aggregates_differentiable_audit_contracts_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("  differentiable-audit-contracts-quality:")
    end = workflow.index("\n\n  tn-mps-baseline-design-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.DIFFERENTIABLE_AUDIT_CONTRACTS_DOCSTRING_RATCHET:
        assert path in block
    for path in quality_gates.DIFFERENTIABLE_AUDIT_CONTRACTS_COVERAGE_COHORT:
        assert path in block
    assert "--fail-under=100" in block
    assert "phase/differentiable_audit_contracts.py" in block
    aggregate = workflow[workflow.index("  ci-gate:") :]
    assert "differentiable-audit-contracts-quality" in aggregate
