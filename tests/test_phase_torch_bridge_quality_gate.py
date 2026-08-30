# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase-QNode Torch quality-gate tests
"""Lock the Phase-QNode Torch facade gate into preflight and CI."""

from pathlib import Path

from tools import phase_torch_bridge_quality_gates as quality_gates
from tools import preflight


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and isolated NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-phase-torch-bridge"][5:]
        == quality_gates.PHASE_TORCH_BRIDGE_QUALITY_RATCHET
    )
    docs = gates["ruff D phase-torch-bridge quality ratchet"]
    assert "--preview" in docs
    assert "D,D413,D417,D420" in docs


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require real Torch execution and exact facade coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["phase-torch-bridge focused coverage"]
    report = gates["phase-torch-bridge exact coverage threshold"]
    cohort = quality_gates.PHASE_TORCH_BRIDGE_COVERAGE_COHORT
    assert "--branch" in run
    assert run[-len(cohort) :] == cohort
    assert quality_gates.PHASE_TORCH_BRIDGE_COVERAGE_DATA_FILE.startswith("/tmp/")
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.PHASE_TORCH_BRIDGE_COVERAGE_INCLUDE}" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.PHASE_TORCH_BRIDGE_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command


def test_ci_runs_torch_gate_inside_differentiable_parity() -> None:
    """Keep the overlay-backed CI steps and aggregate owner required."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    for path in quality_gates.PHASE_TORCH_BRIDGE_QUALITY_RATCHET:
        assert path in workflow
    for path in quality_gates.PHASE_TORCH_BRIDGE_COVERAGE_COHORT:
        assert path in workflow
    assert quality_gates.PHASE_TORCH_BRIDGE_COVERAGE_DATA_FILE in workflow
    assert quality_gates.PHASE_TORCH_BRIDGE_COVERAGE_INCLUDE in workflow
    assert "differentiable-parity" in workflow[workflow.index("  ci-gate:") :]
