# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Studio-executive quality-gate tests
"""Lock the Studio-executive product gate into preflight and CI."""

from pathlib import Path

from tools import preflight
from tools import studio_executive_product_quality_gates as quality_gates


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and isolated NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-studio-executive-product-quality"][5:]
        == quality_gates.STUDIO_EXECUTIVE_PRODUCT_QUALITY_RATCHET
    )
    ruff = gates["ruff D studio-executive-product quality ratchet"]
    assert "--isolated" in ruff and "--preview" in ruff
    assert "D,D413,D417,D420" in ruff


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and exact joint source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["studio-executive-product focused coverage"]
    report = gates["studio-executive-product exact coverage threshold"]
    assert "--branch" in run
    assert run[-len(quality_gates.STUDIO_EXECUTIVE_PRODUCT_COVERAGE_COHORT) :] == (
        quality_gates.STUDIO_EXECUTIVE_PRODUCT_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert (
        "--include=*/studio_executive_product.py,*/studio/manifest.py,*/studio/verbs.py" in report
    )


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.STUDIO_EXECUTIVE_PRODUCT_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command


def test_ci_runs_and_aggregates_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text()
    start = workflow.index("  studio-executive-product-quality:")
    end = workflow.index("\n\n  decisive-advantage-quality:", start)
    block = workflow[start:end]
    assert all(path in block for path in quality_gates.STUDIO_EXECUTIVE_PRODUCT_QUALITY_RATCHET)
    assert all(path in block for path in quality_gates.STUDIO_EXECUTIVE_PRODUCT_COVERAGE_COHORT)
    assert "requirements-ci-studio-platform.txt" in block
    assert "--no-deps --require-hashes" in block
    assert (
        "--include=*/studio_executive_product.py,*/studio/manifest.py,*/studio/verbs.py" in block
    )
    assert "studio-executive-product-quality" in workflow[workflow.index("  ci-gate:") :]
