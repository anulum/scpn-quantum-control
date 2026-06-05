# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable Benchmark Workflow Tests
"""Static tests for differentiable parity and isolated benchmark CI jobs."""

from __future__ import annotations

from pathlib import Path

WORKFLOW = Path(".github/workflows/ci.yml")


def test_ci_workflow_declares_parity_and_isolated_benchmark_jobs() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "differentiable-parity:" in text
    assert "differentiable-isolated-benchmark:" in text
    assert "isolated-benchmark" in text
    assert "self-hosted" in text
    assert "taskset" in text
    assert "chrt" in text
    assert '"classification": "isolated_affinity"' in text
    assert "actions/upload-artifact" in text


def test_ci_workflow_uses_cpu_framework_wheels_without_cuda_extra() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    parity_block = text.split("differentiable-parity:", maxsplit=1)[1].split(
        "differentiable-isolated-benchmark:",
        maxsplit=1,
    )[0]

    assert "jax[cuda12]" not in parity_block
    assert "tensorflow-cpu" in parity_block
    assert "pennylane" in parity_block
    assert "torch" in parity_block
    assert "install-differentiable-framework-overlay" in parity_block
    assert "--install" in parity_block


def test_ci_workflow_uploads_non_isolated_artifacts_without_production_promotion() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "functional_non_isolated" in text
    assert "RUNNER_ENVIRONMENT" in text
    assert "GITHUB_RUN_ID" in text
    assert "classification" in text
    assert "No provider or QPU execution" in text
