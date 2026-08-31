# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — whole-program AD product quality-gate tests
"""Lock the whole-program AD product gate into preflight and CI."""

from tools import preflight
from tools import whole_program_ad_product_quality_gates as quality_gates
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and isolated NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-whole-program-ad-product-quality"][5:]
        == quality_gates.WHOLE_PROGRAM_AD_PRODUCT_QUALITY_RATCHET
    )
    ruff = gates["ruff D whole-program-ad-product quality ratchet"]
    assert "--isolated" in ruff and "--preview" in ruff
    assert "D,D413,D417,D420" in ruff


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and exact source-only coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["whole-program-ad-product focused coverage"]
    report = gates["whole-program-ad-product exact coverage threshold"]
    assert "--branch" in run
    assert run[-len(quality_gates.WHOLE_PROGRAM_AD_PRODUCT_COVERAGE_COHORT) :] == (
        quality_gates.WHOLE_PROGRAM_AD_PRODUCT_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert "--include=*/whole_program_ad_product.py,*/whole_program_ad_api.py" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.WHOLE_PROGRAM_AD_PRODUCT_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command


def test_ci_runs_and_aggregates_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  whole-program-ad-product-quality:")
    end = workflow.index("\n\n  decisive-advantage-quality:", start)
    block = workflow[start:end]
    assert all(path in block for path in quality_gates.WHOLE_PROGRAM_AD_PRODUCT_QUALITY_RATCHET)
    assert all(path in block for path in quality_gates.WHOLE_PROGRAM_AD_PRODUCT_COVERAGE_COHORT)
    assert "--include=*/whole_program_ad_product.py,*/whole_program_ad_api.py" in block
    assert "whole-program-ad-product-quality" in workflow[workflow.index("  ci-gate:") :]
