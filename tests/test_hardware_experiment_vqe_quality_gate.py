# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — hardware experiment VQE quality-gate tests
"""Lock the hardware experiment VQE gate into preflight and CI."""

from pathlib import Path

from tools import hardware_experiment_vqe_quality_gates as quality_gates
from tools import preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and complete connected NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert gates["mypy-strict-hardware-experiment-vqe-quality"][5:] == (
        quality_gates.HARDWARE_EXPERIMENT_VQE_TYPING_RATCHET
    )
    ruff = gates["ruff D hardware-experiment-vqe quality ratchet"]
    assert ruff[-len(quality_gates.HARDWARE_EXPERIMENT_VQE_DOCSTRING_RATCHET) :] == (
        quality_gates.HARDWARE_EXPERIMENT_VQE_DOCSTRING_RATCHET
    )
    assert "--isolated" in ruff and "D,D413" in ruff


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require offline VQE execution and exact source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["hardware-experiment-vqe focused coverage"]
    report = gates["hardware-experiment-vqe exact coverage threshold"]
    assert "--branch" in run
    assert run[-len(quality_gates.HARDWARE_EXPERIMENT_VQE_COVERAGE_COHORT) :] == (
        quality_gates.HARDWARE_EXPERIMENT_VQE_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert "--include=*/hardware/experiment_vqe.py" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.HARDWARE_EXPERIMENT_VQE_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text(encoding="utf-8")
    assert "gates.extend(HARDWARE_EXPERIMENT_VQE_COVERAGE_GATES)" in source


def test_ci_runs_and_aggregates_hardware_experiment_vqe_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  hardware-experiment-vqe-quality:")
    end = workflow.index("\n\n  tn-mps-baseline-design-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.HARDWARE_EXPERIMENT_VQE_DOCSTRING_RATCHET:
        assert path in block
    for path in quality_gates.HARDWARE_EXPERIMENT_VQE_COVERAGE_COHORT:
        assert path in block
    assert "--fail-under=100" in block
    assert "hardware/experiment_vqe.py" in block
    aggregate = workflow[workflow.index("  ci-gate:") :]
    assert "hardware-experiment-vqe-quality" in aggregate
