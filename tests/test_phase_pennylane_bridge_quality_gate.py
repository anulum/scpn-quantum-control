# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase-QNode PennyLane quality-gate tests
"""Lock the Phase-QNode PennyLane bridge gate into framework-parity CI."""

from pathlib import Path

from tools import phase_pennylane_bridge_quality_gates as quality_gates


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and isolated NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-phase-pennylane-bridge"][5:]
        == quality_gates.PHASE_PENNYLANE_BRIDGE_QUALITY_RATCHET
    )
    docs = gates["ruff D phase-pennylane-bridge quality ratchet"]
    assert "--preview" in docs
    assert "D,D413,D417,D420" in docs


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require real PennyLane execution and exact bridge coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["phase-pennylane-bridge focused coverage"]
    report = gates["phase-pennylane-bridge exact coverage threshold"]
    cohort = quality_gates.PHASE_PENNYLANE_BRIDGE_COVERAGE_COHORT
    assert "--branch" in run
    assert run[-len(cohort) :] == cohort
    assert quality_gates.PHASE_PENNYLANE_BRIDGE_COVERAGE_DATA_FILE.startswith("/tmp/")
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.PHASE_PENNYLANE_BRIDGE_COVERAGE_INCLUDE}" in report


def test_framework_parity_ci_mirrors_the_helper_contract() -> None:
    """Keep exact PennyLane coverage inside the existing framework category."""
    workflow = Path(".github/workflows/ci-framework-parity.yml").read_text(encoding="utf-8")
    for path in quality_gates.PHASE_PENNYLANE_BRIDGE_COVERAGE_COHORT:
        assert path in workflow
    assert quality_gates.PHASE_PENNYLANE_BRIDGE_COVERAGE_DATA_FILE in workflow
    assert quality_gates.PHASE_PENNYLANE_BRIDGE_COVERAGE_INCLUDE in workflow
    assert "Run Phase-QNode PennyLane bridge focused coverage" in workflow
    assert "Enforce Phase-QNode PennyLane bridge exact coverage" in workflow
