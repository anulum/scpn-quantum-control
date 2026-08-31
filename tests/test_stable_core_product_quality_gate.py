# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — stable-core product quality-gate tests
"""Lock the stable-core product quality gate into preflight and CI."""

from __future__ import annotations

from tools import preflight
from tools import stable_core_product_quality_gates as quality_gates
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_contract_is_strict_and_numpy_documented() -> None:
    """Require strict typing and isolated NumPy docstrings for the full cohort."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))

    mypy = gates["mypy-strict-stable-core-product-quality"]
    assert mypy[:6] == [
        "/python",
        "-m",
        "mypy",
        "--strict",
        "--explicit-package-bases",
        *quality_gates.STABLE_CORE_PRODUCT_QUALITY_RATCHET[:1],
    ]
    assert mypy[5:] == quality_gates.STABLE_CORE_PRODUCT_QUALITY_RATCHET

    ruff = gates["ruff D stable-core-product quality ratchet"]
    assert "--isolated" in ruff
    assert "D,D413" in ruff
    assert 'lint.pydocstyle.convention = "numpy"' in ruff
    assert ruff[-len(quality_gates.STABLE_CORE_PRODUCT_QUALITY_RATCHET) :] == (
        quality_gates.STABLE_CORE_PRODUCT_QUALITY_RATCHET
    )


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require one isolated owner run and a 100 percent source-only report."""
    gates = dict(quality_gates.build_coverage_gates("/python"))

    run = gates["stable-core-product focused coverage"]
    assert f"--data-file={quality_gates.STABLE_CORE_PRODUCT_COVERAGE_DATA_FILE}" in run
    assert "--branch" in run
    assert run[-len(quality_gates.STABLE_CORE_PRODUCT_COVERAGE_COHORT) :] == (
        quality_gates.STABLE_CORE_PRODUCT_COVERAGE_COHORT
    )

    report = gates["stable-core-product exact coverage threshold"]
    assert "--fail-under=100" in report
    assert "--include=*/stable_core_product.py" in report


def test_preflight_uses_the_same_quality_gate_contract() -> None:
    """Keep the helper-defined gates present verbatim in local preflight."""
    static = dict(preflight.STATIC_GATES)
    coverage = dict(preflight.STABLE_CORE_PRODUCT_COVERAGE_GATES)

    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert static[name] == command
    assert coverage == dict(quality_gates.build_coverage_gates(preflight._PY))


def test_ci_runs_and_aggregates_the_stable_core_product_gate() -> None:
    """Keep CI's focused job and aggregate check wired to the exact cohort."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  stable-core-product-quality:")
    end = workflow.index("\n\n  thermo-readiness-product-quality:", start)
    block = workflow[start:end]

    for path in quality_gates.STABLE_CORE_PRODUCT_QUALITY_RATCHET:
        assert path in block
    for path in quality_gates.STABLE_CORE_PRODUCT_COVERAGE_COHORT:
        assert path in block
    assert "--fail-under=100" in block
    assert "--include=*/stable_core_product.py" in block
    assert "stable-core-product-quality" in workflow[workflow.index("  ci-gate:") :]
