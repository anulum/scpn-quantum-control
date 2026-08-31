# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — open-system completeness quality-gate tests
"""Lock the open-system completeness quality gate into preflight and CI."""

from __future__ import annotations

from tools import open_system_completeness_quality_gates as quality_gates
from tools import preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_separates_public_docs_from_owner_test_typing() -> None:
    """Type-check the owner test while limiting Ruff-D to public/gate surfaces."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    mypy = gates["mypy-strict-open-system-completeness-quality"]
    ruff = gates["ruff D open-system-completeness quality ratchet"]

    assert mypy[5:] == quality_gates.OPEN_SYSTEM_COMPLETENESS_TYPING_RATCHET
    assert "tests/test_open_system_mcwf_product.py" in mypy
    assert (
        ruff[-len(quality_gates.OPEN_SYSTEM_COMPLETENESS_DOCSTRING_RATCHET) :]
        == quality_gates.OPEN_SYSTEM_COMPLETENESS_DOCSTRING_RATCHET
    )
    assert "tests/test_open_system_mcwf_product.py" not in ruff
    assert "--isolated" in ruff and "D,D413" in ruff


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and a 100 percent source-only report."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["open-system-completeness focused coverage"]
    report = gates["open-system-completeness exact coverage threshold"]

    assert f"--data-file={quality_gates.OPEN_SYSTEM_COMPLETENESS_COVERAGE_DATA_FILE}" in run
    assert "--branch" in run
    assert (
        run[-len(quality_gates.OPEN_SYSTEM_COMPLETENESS_COVERAGE_COHORT) :]
        == quality_gates.OPEN_SYSTEM_COMPLETENESS_COVERAGE_COHORT
    )
    assert "--fail-under=100" in report
    assert "--include=*/open_system_mcwf_product.py,*/phase/tensor_jump.py" in report


def test_preflight_uses_the_helper_defined_gates() -> None:
    """Keep helper-defined static and coverage commands verbatim in preflight."""
    static = dict(preflight.STATIC_GATES)
    coverage = dict(preflight.OPEN_SYSTEM_COMPLETENESS_COVERAGE_GATES)
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert static[name] == command
    assert coverage == dict(quality_gates.build_coverage_gates(preflight._PY))


def test_ci_runs_and_aggregates_the_open_system_completeness_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  open-system-completeness-quality:")
    end = workflow.index("\n\n  thermo-readiness-product-quality:", start)
    block = workflow[start:end]

    for path in quality_gates.OPEN_SYSTEM_COMPLETENESS_TYPING_RATCHET:
        assert path in block
    assert "--fail-under=100" in block
    for path in quality_gates.OPEN_SYSTEM_COMPLETENESS_COVERAGE_COHORT:
        assert path in block
    assert "--include=*/open_system_mcwf_product.py,*/phase/tensor_jump.py" in block
    assert "open-system-completeness-quality" in workflow[workflow.index("  ci-gate:") :]
