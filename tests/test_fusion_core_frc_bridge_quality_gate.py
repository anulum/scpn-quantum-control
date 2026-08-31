# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — fusion-core FRC bridge quality-gate tests
"""Lock the fusion-core FRC bridge gate into preflight and CI."""

from pathlib import Path

from tools import fusion_core_frc_bridge_quality_gates as quality_gates
from tools import preflight


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and complete connected NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert gates["mypy-strict-fusion-core-frc-bridge-quality"][5:] == (
        quality_gates.FUSION_CORE_FRC_BRIDGE_TYPING_RATCHET
    )
    ruff = gates["ruff D fusion-core-frc-bridge quality ratchet"]
    assert ruff[-len(quality_gates.FUSION_CORE_FRC_BRIDGE_DOCSTRING_RATCHET) :] == (
        quality_gates.FUSION_CORE_FRC_BRIDGE_DOCSTRING_RATCHET
    )
    assert "--isolated" in ruff and "D,D413" in ruff


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require offline FRC execution and exact bridge/cost source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["fusion-core-frc-bridge focused coverage"]
    report = gates["fusion-core-frc-bridge exact coverage threshold"]
    assert "--branch" in run
    assert run[-len(quality_gates.FUSION_CORE_FRC_BRIDGE_COVERAGE_COHORT) :] == (
        quality_gates.FUSION_CORE_FRC_BRIDGE_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.FUSION_CORE_FRC_BRIDGE_COVERAGE_INCLUDE}" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.FUSION_CORE_FRC_BRIDGE_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text(encoding="utf-8")
    assert "gates.extend(FUSION_CORE_FRC_BRIDGE_COVERAGE_GATES)" in source


def test_ci_runs_and_aggregates_fusion_core_frc_bridge_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("  fusion-core-frc-bridge-quality:")
    end = workflow.index("\n\n  tn-mps-baseline-design-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.FUSION_CORE_FRC_BRIDGE_DOCSTRING_RATCHET:
        assert path in block
    for path in quality_gates.FUSION_CORE_FRC_BRIDGE_COVERAGE_COHORT:
        assert path in block
    assert "--fail-under=100" in block
    for source in (
        quality_gates.FUSION_CORE_FRC_BRIDGE_SOURCE,
        quality_gates.FRC_PULSED_COST_SOURCE,
    ):
        assert source in block
    assert quality_gates.FUSION_CORE_FRC_BRIDGE_COVERAGE_INCLUDE in block
    aggregate = workflow[workflow.index("  ci-gate:") :]
    assert "fusion-core-frc-bridge-quality" in aggregate
