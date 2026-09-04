# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — QNode framework-parity quality-gate tests
"""Lock the local QNode framework-parity owner into preflight and CI."""

from tools import phase_qnode_framework_parity_quality_gates as quality_gates
from tools import preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and complete isolated NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-phase-qnode-framework-parity"][5:]
        == quality_gates.PHASE_QNODE_FRAMEWORK_PARITY_QUALITY_RATCHET
    )
    docs = gates["ruff D phase-qnode-framework-parity quality ratchet"]
    assert "--preview" in docs
    assert "D,D413,D417,D420" in docs


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require real local-framework execution and exact source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["phase-qnode-framework-parity focused coverage"]
    report = gates["phase-qnode-framework-parity exact coverage threshold"]
    cohort = quality_gates.PHASE_QNODE_FRAMEWORK_PARITY_COVERAGE_COHORT
    assert "--branch" in run
    assert run[-len(cohort) :] == cohort
    assert quality_gates.PHASE_QNODE_FRAMEWORK_PARITY_COVERAGE_DATA_FILE.startswith("/tmp/")
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.PHASE_QNODE_FRAMEWORK_PARITY_COVERAGE_INCLUDE}" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight without executing it."""
    assert dict(preflight.PHASE_QNODE_FRAMEWORK_PARITY_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command


def test_ci_runs_the_exact_helper_owned_surface() -> None:
    """Keep static and exact commands in the distributed CI categories."""
    workflow = read_ci_workflow_source()
    for path in quality_gates.PHASE_QNODE_FRAMEWORK_PARITY_QUALITY_RATCHET:
        assert path in workflow
    for path in quality_gates.PHASE_QNODE_FRAMEWORK_PARITY_COVERAGE_COHORT:
        assert path in workflow
    assert quality_gates.PHASE_QNODE_FRAMEWORK_PARITY_COVERAGE_DATA_FILE in workflow
    assert quality_gates.PHASE_QNODE_FRAMEWORK_PARITY_COVERAGE_INCLUDE in workflow
