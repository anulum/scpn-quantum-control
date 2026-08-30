# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — error-mitigation product quality-gate tests
"""Lock the error-mitigation product gate into preflight and CI."""

from pathlib import Path

from tools import error_mitigation_product_quality_gates as quality_gates
from tools import preflight


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and isolated NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-error-mitigation-product-quality"][5:]
        == quality_gates.ERROR_MITIGATION_PRODUCT_QUALITY_RATCHET
    )
    assert "D,D413" in gates["ruff D error-mitigation-product quality ratchet"]


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and exact source-only coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["error-mitigation-product focused coverage"]
    assert "--branch" in run
    assert (
        run[-len(quality_gates.ERROR_MITIGATION_PRODUCT_COVERAGE_COHORT) :]
        == quality_gates.ERROR_MITIGATION_PRODUCT_COVERAGE_COHORT
    )
    report = gates["error-mitigation-product exact coverage threshold"]
    assert "--fail-under=100" in report
    assert (
        "--include=*/error_mitigation_product.py,*/mitigation/compound_mitigation.py,*/mitigation/cpdr.py"
        in report
    )


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.ERROR_MITIGATION_PRODUCT_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command


def test_ci_runs_and_aggregates_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("  error-mitigation-product-quality:")
    end = workflow.index("\n\n  cloud-native-deployment-quality:", start)
    block = workflow[start:end]
    assert all(path in block for path in quality_gates.ERROR_MITIGATION_PRODUCT_QUALITY_RATCHET)
    assert "*/mitigation/compound_mitigation.py" in block
    assert "*/mitigation/cpdr.py" in block
    assert "error-mitigation-product-quality" in workflow[workflow.index("  ci-gate:") :]
