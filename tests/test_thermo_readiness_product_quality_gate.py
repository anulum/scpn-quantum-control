# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — thermo-readiness product quality-gate tests
"""Lock the thermo-readiness quality gate into preflight and CI."""

from __future__ import annotations

from pathlib import Path

from tools import preflight
from tools import thermo_readiness_product_quality_gates as quality_gates


def test_static_gate_owns_product_and_ambient_readiness_surfaces() -> None:
    """Type-check the product and ambient owner with complete preview docs."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    mypy = gates["mypy-strict-thermo-readiness-product-quality"]
    ruff = gates["ruff D thermo-readiness-product quality ratchet"]

    assert mypy[5:] == quality_gates.THERMO_READINESS_PRODUCT_TYPING_RATCHET
    assert "tests/test_thermo_readiness_product.py" in mypy
    assert (
        ruff[-len(quality_gates.THERMO_READINESS_PRODUCT_DOCSTRING_RATCHET) :]
        == quality_gates.THERMO_READINESS_PRODUCT_DOCSTRING_RATCHET
    )
    assert "tests/test_thermo_readiness_product.py" not in ruff
    assert quality_gates.QUANTUM_THERMO_READINESS_SOURCE in ruff
    assert quality_gates.QUANTUM_THERMO_READINESS_EXPORTER in ruff
    assert "--isolated" in ruff and "--preview" in ruff
    assert "D,D413,D417,D420" in ruff


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and a 100 percent source-only report."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["thermo-readiness-product focused coverage"]
    report = gates["thermo-readiness-product exact coverage threshold"]

    assert f"--data-file={quality_gates.THERMO_READINESS_PRODUCT_COVERAGE_DATA_FILE}" in run
    assert "--branch" in run
    assert run[-len(quality_gates.THERMO_READINESS_PRODUCT_COVERAGE_COHORT) :] == (
        quality_gates.THERMO_READINESS_PRODUCT_COVERAGE_COHORT
    )
    assert quality_gates.THERMO_READINESS_PRODUCT_COVERAGE_DATA_FILE.startswith("/tmp/")
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.THERMO_READINESS_PRODUCT_COVERAGE_INCLUDE}" in report


def test_preflight_uses_the_helper_defined_gates() -> None:
    """Keep helper-defined static and coverage commands verbatim in preflight."""
    static = dict(preflight.STATIC_GATES)
    coverage = dict(preflight.THERMO_READINESS_PRODUCT_COVERAGE_GATES)
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert static[name] == command
    assert coverage == dict(quality_gates.build_coverage_gates(preflight._PY))


def test_ci_runs_and_aggregates_the_thermo_readiness_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("  thermo-readiness-product-quality:")
    end = workflow.index("\n\n  adaptive-branching-quality:", start)
    block = workflow[start:end]

    for path in quality_gates.THERMO_READINESS_PRODUCT_TYPING_RATCHET:
        assert path in block
    for path in quality_gates.THERMO_READINESS_PRODUCT_DOCSTRING_RATCHET:
        assert path in block
    for path in quality_gates.THERMO_READINESS_PRODUCT_COVERAGE_COHORT:
        assert path in block
    assert "--fail-under=100" in block
    assert quality_gates.THERMO_READINESS_PRODUCT_COVERAGE_INCLUDE in block
    assert "thermo-readiness-product-quality" in workflow[workflow.index("  ci-gate:") :]
