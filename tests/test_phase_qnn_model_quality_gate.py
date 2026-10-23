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
