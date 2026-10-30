# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase classifier model gate integration tests
"""Keep numerical model checks reachable in preflight and required hosted CI."""

from __future__ import annotations

import shlex
from pathlib import Path

import yaml

from tools import phase_qnn_model_quality_gates as quality
from tools import preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_preflight_registers_the_model_static_and_exact_execution_gates() -> None:
    """Ensure ordinary preflight reaches all model checks without a manual opt-in."""
    static = dict(preflight.STATIC_GATES)
    for name, command in quality.build_static_quality_gates(preflight._PY):
        assert static[name] == command
    assert quality.build_coverage_gates(preflight._PY) == preflight.PHASE_QNN_MODEL_COVERAGE_GATES
    execution, report = dict(quality.build_coverage_gates(preflight._PY)).values()
    assert "--branch" in execution
    assert "--fail-under=100" in report
    assert "--include=" + quality.PHASE_QNN_MODEL_COVERAGE_INCLUDE in report


def test_hosted_job_runs_the_exact_same_model_commands() -> None:
    """Prevent hosted CI drifting away from the model's actual exact-coverage gate."""
    workflow = yaml.safe_load(read_ci_workflow_source())
    steps = workflow["jobs"]["phase-trainability-quality"]["steps"]
    commands = [shlex.split(step["run"]) for step in steps if "run" in step]
    for _name, command in [
        *quality.build_static_quality_gates("python"),
        *quality.build_coverage_gates("python"),
    ]:
        assert command in commands
    assert "phase-trainability-quality" in workflow["jobs"]["ci-gate"]["needs"]


def test_native_model_coverage_is_required_after_actual_sdk_installation() -> None:
    """Keep native model execution, exact coverage and raw evidence in required CI."""
    path = Path(".github/workflows/ci-framework-parity.yml")
    workflow = yaml.safe_load(path.read_text())
    steps = workflow["jobs"]["differentiable-parity"]["steps"]
    named = {step["name"]: step for step in steps if "name" in step}
    names = [step.get("name") for step in steps]
    run = named["Run native phase classifier model coverage"]
    report = named["Enforce native phase classifier model exact coverage"]
    upload = named["Upload exact-commit native phase classifier coverage"]
    assert names.index("Build CPU-only differentiable framework overlay") < names.index(
        run["name"]
    )
    assert "tests/test_phase_qnn_native_model.py" in shlex.split(run["run"])
    assert "--branch" in shlex.split(run["run"])
    include = (
        "--include=*/phase_qnn_native_model.py,*/phase_qnn_native_contracts.py,"
        "*/phase_qnn_framework_gradients.py"
    )
    assert include in shlex.split(run["run"])
    assert include in shlex.split(report["run"])
    assert "--fail-under=100" in shlex.split(report["run"])
    assert upload["with"]["path"] == ".coverage.phase-qnn-native-model"
    assert upload["with"]["if-no-files-found"] == "error"
    assert upload["with"]["include-hidden-files"] is True
    assert not run.get("continue-on-error", False)
    assert not report.get("continue-on-error", False)
    assert not run.get("if")
    assert not report.get("if")
    for source in quality.PHASE_QNN_NATIVE_MODEL_SOURCES:
        assert source in quality.PHASE_QNN_MODEL_QUALITY_RATCHET
    assert "tests/test_phase_qnn_native_model.py" in quality.PHASE_QNN_MODEL_QUALITY_RATCHET
    assert (
        "src/scpn_quantum_control/phase_qnn_framework_gradients.py"
        in quality.PHASE_QNN_MODEL_QUALITY_RATCHET
    )
    assert "tests/test_phase_qnn_framework_gradients.py" in quality.PHASE_QNN_MODEL_QUALITY_RATCHET
