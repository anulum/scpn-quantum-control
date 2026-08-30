# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Wirtinger-implicit quality-gate tests
"""Lock the Wirtinger-implicit product gate into preflight and CI."""

from pathlib import Path

from tools import preflight
from tools import wirtinger_implicit_product_quality_gates as quality_gates


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and isolated NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-wirtinger-implicit-product-quality"][5:]
        == quality_gates.WIRTINGER_IMPLICIT_PRODUCT_QUALITY_RATCHET
    )
    ruff = gates["ruff D wirtinger-implicit-product quality ratchet"]
    assert "D,D413" in ruff
    assert (
        ruff[-len(quality_gates.WIRTINGER_IMPLICIT_PRODUCT_DOCSTRING_RATCHET) :]
        == quality_gates.WIRTINGER_IMPLICIT_PRODUCT_DOCSTRING_RATCHET
    )


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and exact source-only coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["wirtinger-implicit-product focused coverage"]
    report = gates["wirtinger-implicit-product exact coverage threshold"]
    assert "--branch" in run
    assert (
        run[-len(quality_gates.WIRTINGER_IMPLICIT_PRODUCT_COVERAGE_COHORT) :]
        == quality_gates.WIRTINGER_IMPLICIT_PRODUCT_COVERAGE_COHORT
    )
    assert "--fail-under=100" in report
    assert "--include=*/wirtinger_implicit_product.py,*/wirtinger_calculus.py" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.WIRTINGER_IMPLICIT_PRODUCT_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command


def test_ci_runs_and_aggregates_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text()
    start = workflow.index("  wirtinger-implicit-product-quality:")
    end = workflow.index("\n\n  decisive-advantage-quality:", start)
    block = workflow[start:end]
    assert all(
        path in block for path in quality_gates.WIRTINGER_IMPLICIT_PRODUCT_DOCSTRING_RATCHET
    )
    assert "wirtinger-implicit-product-quality" in workflow[workflow.index("  ci-gate:") :]
