# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable-programming contract gate tests
"""Lock the benchmark-contract owner into preflight and required CI."""

from tools import differentiable_programming_contracts_quality_gates as quality_gates
from tools import differentiable_quality_gates, preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete preview NumPy documentation."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    typing = gates["mypy-strict-differentiable-programming-contracts-quality"]
    documentation = gates["ruff D differentiable-programming-contracts quality ratchet"]

    assert typing[5:] == quality_gates.DIFFERENTIABLE_PROGRAMMING_CONTRACTS_QUALITY_RATCHET
    assert "--preview" in documentation
    assert "D,D413,D417,D420" in documentation
    assert "lint.explicit-preview-rules = true" in documentation
    assert documentation[
        -len(quality_gates.DIFFERENTIABLE_PROGRAMMING_CONTRACTS_QUALITY_RATCHET) :
    ] == (quality_gates.DIFFERENTIABLE_PROGRAMMING_CONTRACTS_QUALITY_RATCHET)


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require real branch execution and exact source-only coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["differentiable-programming-contracts focused coverage"]
    report = gates["differentiable-programming-contracts exact coverage threshold"]

    assert "--branch" in run
    assert run[-len(quality_gates.DIFFERENTIABLE_PROGRAMMING_CONTRACTS_COVERAGE_COHORT) :] == (
        quality_gates.DIFFERENTIABLE_PROGRAMMING_CONTRACTS_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert (
        f"--include={quality_gates.DIFFERENTIABLE_PROGRAMMING_CONTRACTS_COVERAGE_INCLUDE}"
        in report
    )
    assert quality_gates.DIFFERENTIABLE_PROGRAMMING_QUANTUM_TEST in run
    assert "*/benchmarks/differentiable_programming_quantum.py" in report


def test_aggregate_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in aggregate differentiable preflight."""
    static = dict(differentiable_quality_gates.build_static_quality_gates(preflight._PY))
    coverage = dict(differentiable_quality_gates.build_coverage_gates(preflight._PY))

    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert static[name] == command
        assert dict(preflight.STATIC_GATES)[name] == command
    for name, command in quality_gates.build_coverage_gates(preflight._PY):
        assert coverage[name] == command
        assert dict(preflight.DIFFERENTIABLE_QUALITY_COVERAGE_GATES)[name] == command


def test_required_ci_runs_the_helper_defined_contract_owner() -> None:
    """Keep the focused owner inside the required whole-program AD job."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  whole-program-ad-product-quality:")
    end = workflow.index("\n\n  neural-operator-cost-model-quality:", start)
    block = workflow[start:end]

    for path in quality_gates.DIFFERENTIABLE_PROGRAMMING_CONTRACTS_QUALITY_RATCHET:
        assert path in block
    for name, _command in (
        *quality_gates.build_static_quality_gates("python"),
        *quality_gates.build_coverage_gates("python"),
    ):
        assert name in block
    assert "--fail-under=100" in block
