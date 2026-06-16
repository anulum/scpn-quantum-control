# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Differentiable Framework CI Workflow
"""Static tests for differentiable framework CI and benchmark workflow policy."""

from __future__ import annotations

from pathlib import Path

WORKFLOW = Path(".github/workflows/differentiable-frameworks.yml")
PYTHON_VERSIONS = ("3.10", "3.11", "3.12", "3.13")


def test_differentiable_framework_workflow_declares_sparse_and_full_matrices() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    for version in PYTHON_VERSIONS:
        assert f'python-version: "{version}"' in text
        assert f"requirements-ci-py{version.replace('.', '')}-linux.txt" in text
    assert "jax" in text
    assert "torch" in text
    assert "tensorflow" in text
    assert "pennylane" in text
    assert "tests/test_phase_qnode_framework_parity.py" in text
    assert "not hardware" in text
    assert "workflow_dispatch" in text
    assert "schedule" in text
    assert "taskset" in text
    assert "upload-artifact" in text
    assert "actions/checkout@df4cb1c069e1874edd31b4311f1884172cec0e10" in text
    assert "actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405" in text
    assert "install-differentiable-framework-overlay --overlay-path" in text
    assert "--install" in text


def test_differentiable_framework_workflow_runs_sparse_and_full_for_each_python() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    rows = _matrix_rows(text)

    expected = {
        (version, profile) for version in PYTHON_VERSIONS for profile in ("sparse", "full")
    }
    observed = {(row["python-version"], row["profile"]) for row in rows}

    assert expected <= observed
    for version, profile in expected:
        row = next(
            item
            for item in rows
            if item["python-version"] == version and item["profile"] == profile
        )
        assert (
            row["requirements-file"] == f"requirements-ci-py{version.replace('.', '')}-linux.txt"
        )


def test_differentiable_framework_workflow_enforces_test_quality_audit() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Run module-specific test audit" in text
    assert "python tools/audit_test_quality.py" in text


def test_differentiable_framework_workflow_declares_optional_gpu_lane() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "optional-gpu-contract" in text
    assert "continue-on-error: true" in text
    assert "workflow_dispatch" in text
    assert "tests/test_gpu_batch_vqe.py" in text
    assert "tests/test_gpu_batch_vqe_contracts.py" in text
    assert "not hardware and not performance" in text
    assert "Upload optional GPU contract evidence" in text


def _matrix_rows(text: str) -> tuple[dict[str, str], ...]:
    rows: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    in_matrix = False
    for raw_line in text.splitlines():
        if in_matrix and raw_line.startswith("    steps:"):
            break
        line = raw_line.strip()
        if line == "include:":
            in_matrix = True
            continue
        if not in_matrix:
            continue
        if line.startswith("- python-version:"):
            current = {"python-version": _workflow_value(line)}
            rows.append(current)
            continue
        if current is not None and ": " in line:
            key, value = line.split(": ", maxsplit=1)
            current[key] = value.strip().strip('"')
    return tuple(rows)


def _workflow_value(line: str) -> str:
    return line.split(": ", maxsplit=1)[1].strip().strip('"')
