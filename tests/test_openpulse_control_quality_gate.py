# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — OpenPulse-control quality-gate tests
"""Lock the OpenPulse-control quality gate into preflight and CI."""

from pathlib import Path

from tools import openpulse_control_quality_gates as quality_gates
from tools import preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete connected docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-openpulse-control-quality"][5:]
        == quality_gates.OPENPULSE_CONTROL_TYPING_RATCHET
    )
    ruff = gates["ruff D openpulse-control quality ratchet"]
    assert (
        ruff[-len(quality_gates.OPENPULSE_CONTROL_DOCSTRING_RATCHET) :]
        == quality_gates.OPENPULSE_CONTROL_DOCSTRING_RATCHET
    )
    assert "--isolated" in ruff and "D,D413" in ruff


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require offline OpenPulse execution and exact source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["openpulse-control focused coverage"]
    report = gates["openpulse-control exact coverage threshold"]
    assert "--branch" in run
    assert (
        run[-len(quality_gates.OPENPULSE_CONTROL_COVERAGE_COHORT) :]
        == quality_gates.OPENPULSE_CONTROL_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert "--include=*/hardware/openpulse_control.py" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.OPENPULSE_CONTROL_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text(encoding="utf-8")
    assert "gates.extend(OPENPULSE_CONTROL_COVERAGE_GATES)" in source


def test_ci_runs_and_aggregates_openpulse_control_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  openpulse-control-quality:")
    end = workflow.index("\n\n  tn-mps-baseline-design-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.OPENPULSE_CONTROL_DOCSTRING_RATCHET:
        assert path in block
    assert "--fail-under=100" in block
    assert "hardware/openpulse_control.py" in block
    aggregate = workflow[workflow.index("  ci-gate:") :]
    assert "openpulse-control-quality" in aggregate
