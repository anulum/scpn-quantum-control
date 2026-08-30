# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — whole-program frontend-contract quality-gate tests
"""Lock the whole-program frontend-contract gate into preflight and CI."""

from pathlib import Path

from tools import preflight
from tools import whole_program_frontend_contracts_quality_gates as quality_gates


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and complete owner NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-whole-program-frontend-contracts-quality"][5:]
        == quality_gates.WHOLE_PROGRAM_FRONTEND_CONTRACTS_TYPING_RATCHET
    )
    ruff = gates["ruff D whole-program-frontend-contracts quality ratchet"]
    assert ruff[-len(quality_gates.WHOLE_PROGRAM_FRONTEND_CONTRACTS_DOCSTRING_RATCHET) :] == (
        quality_gates.WHOLE_PROGRAM_FRONTEND_CONTRACTS_DOCSTRING_RATCHET
    )
    assert "--isolated" in ruff and "D,D413" in ruff


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and exact source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["whole-program-frontend-contracts focused coverage"]
    report = gates["whole-program-frontend-contracts exact coverage threshold"]
    assert "--branch" in run
    assert run[-len(quality_gates.WHOLE_PROGRAM_FRONTEND_CONTRACTS_COVERAGE_COHORT) :] == (
        quality_gates.WHOLE_PROGRAM_FRONTEND_CONTRACTS_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.WHOLE_PROGRAM_FRONTEND_CONTRACTS_COVERAGE_INCLUDE}" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.WHOLE_PROGRAM_FRONTEND_CONTRACTS_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text(encoding="utf-8")
    assert "gates.extend(WHOLE_PROGRAM_FRONTEND_CONTRACTS_COVERAGE_GATES)" in source


def test_ci_runs_and_aggregates_frontend_contract_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("  whole-program-frontend-contracts-quality:")
    end = workflow.index("\n\n  tn-mps-baseline-design-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.WHOLE_PROGRAM_FRONTEND_CONTRACTS_DOCSTRING_RATCHET:
        assert path in block
    assert "--fail-under=100" in block
    assert quality_gates.WHOLE_PROGRAM_FRONTEND_CONTRACTS_COVERAGE_INCLUDE in block
    assert "whole-program-frontend-contracts-quality" in workflow[workflow.index("  ci-gate:") :]
