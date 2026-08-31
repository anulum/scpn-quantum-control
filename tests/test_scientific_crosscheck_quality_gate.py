# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — scientific crosscheck quality-gate tests
"""Lock scientific crosscheck quality commands into preflight and CI."""

from tools import preflight
from tools import scientific_crosscheck_quality_gates as quality_gates
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_helper_builds_strict_preview_and_exact_gates() -> None:
    """Require strict preview docs, real suites, and exact source coverage."""
    static = dict(quality_gates.build_static_quality_gates("/python"))
    docs = static["ruff D scientific-crosscheck quality ratchet"]
    coverage = dict(quality_gates.build_coverage_gates("/python"))
    run = coverage["scientific-crosscheck focused coverage"]
    report = coverage["scientific-crosscheck exact coverage threshold"]
    assert "--preview" in docs
    assert "D,D413,D417,D420" in docs
    assert run[-2:] == quality_gates.SCIENTIFIC_CROSSCHECK_COVERAGE_COHORT
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.SCIENTIFIC_CROSSCHECK_COVERAGE_INCLUDE}" in report


def test_preflight_uses_helper_commands_verbatim() -> None:
    """Keep helper-defined commands exact in preflight."""
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    assert dict(preflight.SCIENTIFIC_CROSSCHECK_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )


def test_ci_runs_and_aggregates_scientific_crosschecks() -> None:
    """Keep the dedicated CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  scientific-crosscheck-quality:")
    end = workflow.index("\n\n  resource-budget-gate-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.SCIENTIFIC_CROSSCHECK_QUALITY_RATCHET:
        assert path in block
    for path in quality_gates.SCIENTIFIC_CROSSCHECK_COVERAGE_COHORT:
        assert path in block
    assert quality_gates.SCIENTIFIC_CROSSCHECK_COVERAGE_INCLUDE in block
    resource_start = workflow.index("  resource-budget-gate-quality:", end)
    resource_block = workflow[
        resource_start : workflow.index("\n\n  advantage-language", resource_start)
    ]
    assert "needs: [lint, scientific-crosscheck-quality]" in resource_block
