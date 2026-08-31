# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Program AD fuzz assurance quality-gate tests
"""Lock the Program AD fuzz assurance gate into preflight and CI."""

from __future__ import annotations

from tools import preflight
from tools import program_ad_fuzz_assurance_quality_gates as quality_gates
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and isolated NumPy docstrings for the cohort."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    mypy = gates["mypy-strict-program-ad-fuzz-assurance-quality"]
    ruff = gates["ruff D program-ad-fuzz-assurance quality ratchet"]
    assert mypy[5:] == quality_gates.PROGRAM_AD_FUZZ_ASSURANCE_QUALITY_RATCHET
    assert (
        ruff[-len(quality_gates.PROGRAM_AD_FUZZ_ASSURANCE_QUALITY_RATCHET) :]
        == quality_gates.PROGRAM_AD_FUZZ_ASSURANCE_QUALITY_RATCHET
    )
    assert "--isolated" in ruff and "D,D413" in ruff


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and a 100 percent source-only report."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["program-ad-fuzz-assurance focused coverage"]
    report = gates["program-ad-fuzz-assurance exact coverage threshold"]
    data_file = quality_gates.PROGRAM_AD_FUZZ_ASSURANCE_COVERAGE_DATA_FILE
    assert f"--data-file={data_file}" in run
    assert "--branch" in run
    assert run[-1:] == quality_gates.PROGRAM_AD_FUZZ_ASSURANCE_COVERAGE_COHORT
    assert "--fail-under=100" in report
    assert "--include=*/program_ad_fuzz_assurance.py" in report


def test_preflight_uses_the_helper_defined_gates() -> None:
    """Keep helper-defined static and coverage commands verbatim in preflight."""
    static = dict(preflight.STATIC_GATES)
    coverage = dict(preflight.PROGRAM_AD_FUZZ_ASSURANCE_COVERAGE_GATES)
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert static[name] == command
    assert coverage == dict(quality_gates.build_coverage_gates(preflight._PY))


def test_ci_runs_and_aggregates_the_product_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  program-ad-fuzz-assurance-quality:")
    end = workflow.index("\n\n  multi-hal-federation-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.PROGRAM_AD_FUZZ_ASSURANCE_QUALITY_RATCHET:
        assert path in block
    assert "--fail-under=100" in block
    assert "--include=*/program_ad_fuzz_assurance.py" in block
    assert "program-ad-fuzz-assurance-quality" in workflow[workflow.index("  ci-gate:") :]
