# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Kuramoto layout-optimiser quality-gate tests
"""Lock the Kuramoto layout-optimiser quality gate into preflight and CI."""

from pathlib import Path

from tools import kuramoto_layout_optimiser_quality_gates as quality_gates
from tools import preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete direct-owner docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-kuramoto-layout-optimiser-quality"][5:]
        == quality_gates.KURAMOTO_LAYOUT_OPTIMISER_TYPING_RATCHET
    )
    ruff = gates["ruff D Kuramoto layout-optimiser quality ratchet"]
    assert (
        ruff[-len(quality_gates.KURAMOTO_LAYOUT_OPTIMISER_DOCSTRING_RATCHET) :]
        == quality_gates.KURAMOTO_LAYOUT_OPTIMISER_DOCSTRING_RATCHET
    )
    assert "--preview" in ruff and "D,D413,D417,D420" in ruff


def test_coverage_gate_is_isolated_connected_and_exact() -> None:
    """Require offline optimizer execution and exact source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["Kuramoto layout-optimiser focused coverage"]
    report = gates["Kuramoto layout-optimiser exact coverage threshold"]
    assert "--branch" in run
    assert run[-1:] == quality_gates.KURAMOTO_LAYOUT_OPTIMISER_COVERAGE_COHORT
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert "--include=*/hardware/kuramoto_layout_optimiser.py" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.KURAMOTO_LAYOUT_OPTIMISER_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    assert "gates.extend(KURAMOTO_LAYOUT_OPTIMISER_COVERAGE_GATES)" in Path(
        "tools/preflight.py"
    ).read_text(encoding="utf-8")


def test_ci_runs_and_aggregates_kuramoto_layout_optimiser_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  kuramoto-layout-optimiser-quality:")
    end = workflow.index("\n\n  tn-mps-baseline-design-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.KURAMOTO_LAYOUT_OPTIMISER_DOCSTRING_RATCHET:
        assert path in block
    assert "--fail-under=100" in block
    assert "hardware/kuramoto_layout_optimiser.py" in block
    assert "kuramoto-layout-optimiser-quality" in workflow[workflow.index("  ci-gate:") :]
