# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Program AD alias-analysis quality-gate tests
"""Lock the Program AD alias-analysis quality gate into preflight and CI."""

from pathlib import Path

from tools import preflight
from tools import program_ad_alias_analysis_quality_gates as quality_gates
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete direct-owner docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-program-ad-alias-analysis-quality"][5:]
        == quality_gates.PROGRAM_AD_ALIAS_ANALYSIS_TYPING_RATCHET
    )
    ruff = gates["ruff D program-ad alias-analysis quality ratchet"]
    assert (
        ruff[-len(quality_gates.PROGRAM_AD_ALIAS_ANALYSIS_DOCSTRING_RATCHET) :]
        == quality_gates.PROGRAM_AD_ALIAS_ANALYSIS_DOCSTRING_RATCHET
    )
    assert "--preview" in ruff and "D,D413,D417,D420" in ruff


def test_coverage_gate_is_isolated_public_and_exact() -> None:
    """Require public alias-analysis execution and exact source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["program-ad alias-analysis focused coverage"]
    report = gates["program-ad alias-analysis exact coverage threshold"]
    assert "--branch" in run
    assert (
        run[-len(quality_gates.PROGRAM_AD_ALIAS_ANALYSIS_COVERAGE_COHORT) :]
        == quality_gates.PROGRAM_AD_ALIAS_ANALYSIS_COVERAGE_COHORT
    )
    assert quality_gates.PROGRAM_AD_ALIAS_ANALYSIS_COVERAGE_DATA_FILE.startswith("/tmp/")
    assert "--fail-under=100" in report
    assert "--include=*/program_ad_alias_analysis.py" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.PROGRAM_AD_ALIAS_ANALYSIS_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    assert "gates.extend(PROGRAM_AD_ALIAS_ANALYSIS_COVERAGE_GATES)" in Path(
        "tools/preflight.py"
    ).read_text(encoding="utf-8")


def test_ci_runs_and_aggregates_program_ad_alias_analysis_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  program-ad-alias-analysis-quality:")
    end = workflow.index("\n\n  general-unitary-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.PROGRAM_AD_ALIAS_ANALYSIS_DOCSTRING_RATCHET:
        assert path in block
    assert "--fail-under=100" in block
    assert quality_gates.PROGRAM_AD_ALIAS_ANALYSIS_SOURCE in block
    assert "program-ad-alias-analysis-quality" in workflow[workflow.index("  ci-gate:") :]
