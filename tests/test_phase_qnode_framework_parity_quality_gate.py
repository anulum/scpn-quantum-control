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


def test_native_operator_gate_requires_actual_runtimes_and_exact_coverage() -> None:
    """Keep the extracted arithmetic in strict checks and required native execution."""
    workflow = yaml.safe_load(workflow_path_for_job("differentiable-parity").read_text())
    steps = workflow["jobs"]["differentiable-parity"]["steps"]
    named = {step["name"]: step for step in steps if "name" in step}
    run = named["Run native QNode arithmetic coverage"]
    report = named["Enforce native QNode arithmetic exact coverage"]
    upload = named["Upload exact-commit native QNode arithmetic coverage"]
    assert steps.index(named["Build CPU-only differentiable framework overlay"]) < steps.index(run)
    assert steps.index(run) < steps.index(report) < steps.index(upload)
    include = (
        "--include=*/phase_qnode_native_operators.py,*/phase_qnode_native_expectations.py,"
        "*/phase_qnode_gate_numerics.py,*/phase_qnode_state_evolution.py,"
        "*/phase_qnode_gate_dispatch.py,*/phase_qnode_native_gates.py,"
        "*/phase_qnode_statevector_engine.py,*/phase_qnode_native_statevector.py"
    )
    assert include in shlex.split(run["run"])
    assert include in shlex.split(report["run"])
    assert "--branch" in shlex.split(run["run"])
    assert "--fail-under=100" in shlex.split(report["run"])
    assert "tests/test_phase_qnode_native_operators.py" in shlex.split(run["run"])
    assert "tests/test_phase_qnode_native_expectations.py" in shlex.split(run["run"])
    assert "tests/test_phase_qnode_gate_numerics.py" in shlex.split(run["run"])
    assert "tests/test_phase_qnode_state_evolution.py" in shlex.split(run["run"])
    assert "tests/test_phase_qnode_gate_dispatch.py" in shlex.split(run["run"])
    assert "tests/test_phase_qnode_native_gates.py" in shlex.split(run["run"])
    assert "tests/test_phase_qnode_native_statevector.py" in shlex.split(run["run"])
    assert run["env"]["JAX_ENABLE_X64"] == "true"
    assert upload["with"]["path"] == ".coverage.phase-qnode-native-arithmetic"
    assert upload["with"]["if-no-files-found"] == "error"
    assert upload["with"]["include-hidden-files"] is True
    assert upload["with"]["name"].endswith("-${{ github.sha }}")
    for step in (run, report, upload):
        assert not step.get("if")
        assert not step.get("continue-on-error", False)
    for path in (
        "src/scpn_quantum_control/phase_qnode_gate_numerics.py",
        "src/scpn_quantum_control/phase_qnode_native_expectations.py",
        "src/scpn_quantum_control/phase_qnode_native_operators.py",
        "src/scpn_quantum_control/phase_qnode_state_evolution.py",
        "src/scpn_quantum_control/phase_qnode_gate_dispatch.py",
        "src/scpn_quantum_control/phase_qnode_native_gates.py",
        "src/scpn_quantum_control/phase_qnode_statevector_engine.py",
        "src/scpn_quantum_control/phase_qnode_native_statevector.py",
        "tests/test_phase_qnode_gate_numerics.py",
        "tests/test_phase_qnode_native_expectations.py",
        "tests/test_phase_qnode_native_operators.py",
        "tests/test_phase_qnode_state_evolution.py",
        "tests/test_phase_qnode_gate_dispatch.py",
        "tests/test_phase_qnode_native_gates.py",
        "tests/test_phase_qnode_native_statevector.py",
    ):
        assert path in quality_gates.PHASE_QNODE_FRAMEWORK_PARITY_QUALITY_RATCHET


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
        if step.get("name") == "Admit complete test coverage (3.12 only)"
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


@pytest.mark.parametrize(
    ("producer_title", "download_title", "sources"),
    [
        (
            "native phase classifier model",
            "native phase classifier",
            (
                "phase_qnn_native_model.py",
                "phase_qnn_native_contracts.py",
                "phase_qnn_framework_gradients.py",
            ),
        ),
        (
            "native QNode arithmetic",
            "native QNode arithmetic",
            (
                "phase_qnode_native_operators.py",
                "phase_qnode_native_expectations.py",
                "phase_qnode_gate_numerics.py",
                "phase_qnode_state_evolution.py",
            ),
        ),
    ],
)
def test_global_coverage_combines_exact_native_model_and_qnode_evidence(
    producer_title: str, download_title: str, sources: tuple[str, ...]
) -> None:
    """Keep real native measurements in the exact global report before release audits.

    Parameters
    ----------
    producer_title
        Existing required native runtime coverage gate's descriptive name.
    download_title
        Descriptive name of the receiving exact-commit artifact step.
    sources
        Numerical sources and contracts whose actual native coverage is required.

    """
    producer = yaml.safe_load(workflow_path_for_job("differentiable-parity").read_text())
    consumer = yaml.safe_load(workflow_path_for_job("test").read_text())
    psteps = producer["jobs"]["differentiable-parity"]["steps"]
    csteps = consumer["jobs"]["test"]["steps"]
    gate = next(
        step for step in psteps if step.get("name") == f"Enforce {producer_title} exact coverage"
    )
    data_file = next(
        arg.split("=", 1)[1] for arg in shlex.split(gate["run"]) if arg.startswith("--data-file=")
    )
    upload = next(step for step in psteps if step.get("with", {}).get("path") == data_file)
    download = next(
        step
        for step in csteps
        if step.get("name") == f"Download exact-commit {download_title} coverage (3.12 only)"
    )
    assert upload["with"]["name"] == download["with"]["name"]
    assert upload["with"]["name"].endswith("-${{ github.sha }}")
    assert upload["with"]["include-hidden-files"] is True
    assert upload["with"]["if-no-files-found"] == "error"
    assert psteps.index(gate) < psteps.index(upload)
    assert download["if"] == "matrix.python-version == '3.12'"
    assert not download.get("continue-on-error", False)
    coverage_step = next(
        step for step in csteps if step.get("name") == "Admit complete test coverage (3.12 only)"
    )
    destination = download["with"]["path"].replace("${{ runner.temp }}", "$RUNNER_TEMP")
    combine = f'python -m coverage combine --append --keep "{destination}/{data_file}"'
    commands = coverage_step["run"]
    assert commands.index(combine) < commands.index("coverage report")
    assert commands.index("coverage report") < commands.index("coverage xml")
    assert commands.index("coverage xml") < commands.index("tools/audit_coverage_policy.py")
    report = next(shlex.split(line) for line in commands.splitlines() if "coverage report" in line)
    assert "--fail-under=100" in report
    include = next(arg for arg in report if arg.startswith("--include="))
    for source in sources:
        assert "*/" + source in include.split("=", 1)[1].split(",")
    assert csteps.index(download) < csteps.index(coverage_step)
    policy = load_ci_workflow_policy()
    category = next(row for row in policy["categories"] if row["id"] == "native-integration")
    assert "framework-parity" in category["caller_needs"]
