# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — QNode framework-parity quality-gate tests
"""Lock the local QNode framework-parity owner into preflight and CI."""

import shlex
from pathlib import Path
from tempfile import gettempdir

import pytest
import yaml

from tools import phase_qnode_framework_parity_quality_gates as quality_gates
from tools import preflight
from tools.ci_workflow_inventory import (
    load_ci_workflow_policy,
    read_ci_workflow_source,
    workflow_path_for_job,
)


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
    coverage_path = Path(quality_gates.PHASE_QNODE_FRAMEWORK_PARITY_COVERAGE_DATA_FILE)
    assert coverage_path.parent == Path(gettempdir())
    assert coverage_path.name == "scpn-qc-phase-qnode-framework-parity.coverage"
    assert run.count(f"--data-file={coverage_path}") == 1
    assert report.count(f"--data-file={coverage_path}") == 1
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
    coverage_name = Path(quality_gates.PHASE_QNODE_FRAMEWORK_PARITY_COVERAGE_DATA_FILE).name
    assert workflow.count(f"--data-file=/tmp/{coverage_name}") == 2
    assert quality_gates.PHASE_QNODE_FRAMEWORK_PARITY_COVERAGE_INCLUDE in workflow


@pytest.mark.parametrize(
    ("framework", "source_name"),
    [("Torch", "torch_qnode_transforms.py"), ("PennyLane", "pennylane_import.py")],
)
def test_global_coverage_retains_required_exact_commit_sdk_measurements(
    framework: str, source_name: str
) -> None:
    """Prevent SDK coverage being gated separately but lost from the release report."""
    policy = load_ci_workflow_policy()
    category = next(item for item in policy["categories"] if item["id"] == "native-integration")
    assert "framework-parity" in category["caller_needs"]
    producer = yaml.safe_load(workflow_path_for_job("differentiable-parity").read_text())
    consumer = yaml.safe_load(workflow_path_for_job("test").read_text())
    producer_steps = producer["jobs"]["differentiable-parity"]["steps"]
    consumer_steps = consumer["jobs"]["test"]["steps"]
    upload = next(
        step
        for step in producer_steps
        if step.get("name") == f"Upload exact-commit Phase-QNode {framework} coverage"
    )
    download = next(
        step
        for step in consumer_steps
        if step.get("name")
        == f"Download exact-commit Phase-QNode {framework} coverage (3.12 only)"
    )
    assert upload["uses"].startswith("actions/upload-artifact@")
    assert upload["with"]["name"] == download["with"]["name"]
    assert upload["with"]["name"].endswith("-${{ github.sha }}")
    assert upload["with"]["if-no-files-found"] == "error"
    assert upload.get("continue-on-error", False) is False
    assert "if" not in upload
    assert download["uses"].startswith("actions/download-artifact@")
    assert download["if"] == "matrix.python-version == '3.12'"
    assert download.get("continue-on-error", False) is False
    owner_gate = next(
        step
        for step in producer_steps
        if step.get("name") == f"Enforce Phase-QNode {framework} bridge exact coverage"
    )
    owner_command = shlex.split(owner_gate["run"])
    assert "--fail-under=100" in owner_command
    assert f"--data-file={upload['with']['path']}" in owner_command
    assert any(source_name in arg for arg in owner_command if arg.startswith("--include="))
    assert producer_steps.index(owner_gate) < producer_steps.index(upload)
    coverage_step = next(
        step
        for step in consumer_steps
        if step.get("name") == "Run tests with coverage (3.12 only)"
    )
    destination = download["with"]["path"].replace("${{ runner.temp }}", "$RUNNER_TEMP")
    combine = (
        f'python -m coverage combine --append --keep "{destination}/'
        f'{Path(upload["with"]["path"]).name}"'
    )
    commands = coverage_step["run"]
    assert commands.index(combine) < commands.index("python -m coverage xml")
    exact_report = next(
        shlex.split(line) for line in commands.splitlines() if "coverage report" in line
    )
    assert "--fail-under=100" in exact_report
    assert any(source_name in arg for arg in exact_report if arg.startswith("--include="))
    assert commands.index("coverage report") < commands.index("tools/audit_coverage_policy.py")
    assert consumer_steps.index(download) < consumer_steps.index(coverage_step)
