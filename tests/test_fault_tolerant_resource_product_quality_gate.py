# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — fault-tolerant resource product quality-gate tests
"""Lock the fault-tolerant resource product gate into preflight and CI."""

from tools import fault_tolerant_resource_product_quality_gates as quality_gates
from tools import preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and isolated NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-fault-tolerant-resource-product-quality"][5:]
        == quality_gates.FAULT_TOLERANT_RESOURCE_PRODUCT_QUALITY_RATCHET
    )
    assert "D,D413,D417,D420" in gates["ruff D fault-tolerant-resource-product quality ratchet"]


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and exact source-only coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["fault-tolerant-resource-product focused coverage"]
    report = gates["fault-tolerant-resource-product exact coverage threshold"]
    assert "--branch" in run
    cohort = quality_gates.FAULT_TOLERANT_RESOURCE_PRODUCT_COVERAGE_COHORT
    assert run[-len(cohort) :] == cohort
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert "--include=*/fault_tolerant_resource_product.py,*/qec/error_budget.py" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.FAULT_TOLERANT_RESOURCE_PRODUCT_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command


def test_ci_runs_and_aggregates_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  fault-tolerant-resource-product-quality:")
    end = workflow.index("\n\n  phase-qnode-product-quality:", start)
    block = workflow[start:end]
    assert all(
        path in block for path in quality_gates.FAULT_TOLERANT_RESOURCE_PRODUCT_QUALITY_RATCHET
    )
    assert quality_gates.FAULT_TOLERANT_RESOURCE_PRODUCT_COVERAGE_DATA_FILE in block
    assert "fault-tolerant-resource-product-quality" in workflow[workflow.index("  ci-gate:") :]
