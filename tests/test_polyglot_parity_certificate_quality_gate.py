# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — polyglot parity certificate quality-gate tests
"""Lock the polyglot parity certificate gate into preflight and CI."""

from __future__ import annotations

from tools import polyglot_parity_certificate_quality_gates as quality_gates
from tools import preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and isolated NumPy docstrings for the cohort."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    mypy = gates["mypy-strict-polyglot-parity-certificate-quality"]
    ruff = gates["ruff D polyglot-parity-certificate quality ratchet"]
    assert mypy[5:] == quality_gates.POLYGLOT_PARITY_CERTIFICATE_QUALITY_RATCHET
    assert (
        ruff[-len(quality_gates.POLYGLOT_PARITY_CERTIFICATE_QUALITY_RATCHET) :]
        == quality_gates.POLYGLOT_PARITY_CERTIFICATE_QUALITY_RATCHET
    )
    assert "--isolated" in ruff and "D,D413" in ruff


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and a 100 percent source-only report."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["polyglot-parity-certificate focused coverage"]
    report = gates["polyglot-parity-certificate exact coverage threshold"]
    data_file = quality_gates.POLYGLOT_PARITY_CERTIFICATE_COVERAGE_DATA_FILE
    assert f"--data-file={data_file}" in run
    assert "--branch" in run
    assert run[-1:] == quality_gates.POLYGLOT_PARITY_CERTIFICATE_COVERAGE_COHORT
    assert "--fail-under=100" in report
    assert "--include=*/polyglot_parity_certificate.py" in report


def test_preflight_uses_the_helper_defined_gates() -> None:
    """Keep helper-defined static and coverage commands verbatim in preflight."""
    static = dict(preflight.STATIC_GATES)
    coverage = dict(preflight.POLYGLOT_PARITY_CERTIFICATE_COVERAGE_GATES)
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert static[name] == command
    assert coverage == dict(quality_gates.build_coverage_gates(preflight._PY))


def test_ci_runs_and_aggregates_the_product_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  polyglot-parity-certificate-quality:")
    end = workflow.index("\n\n  program-ad-fuzz-assurance-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.POLYGLOT_PARITY_CERTIFICATE_QUALITY_RATCHET:
        assert path in block
    assert "--fail-under=100" in block
    assert "--include=*/polyglot_parity_certificate.py" in block
    assert "polyglot-parity-certificate-quality" in workflow[workflow.index("  ci-gate:") :]
