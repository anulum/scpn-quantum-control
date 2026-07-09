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
RISKY_SELF_HOSTED_TRIGGERS = (
    "pull_request_target:",
    "issue_comment:",
    "workflow_run:",
    "repository_dispatch:",
)


def test_ci_workflow_declares_parity_and_isolated_benchmark_jobs() -> None:
    """Assert that CI still declares both differentiable benchmark lanes."""
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
    """Assert that the GitHub-hosted parity lane stays on CPU wheels."""
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
    """Assert that non-isolated CI evidence is not promoted as isolated output."""
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "functional_non_isolated" in text
    assert "RUNNER_ENVIRONMENT" in text
    assert "GITHUB_RUN_ID" in text
    assert "classification" in text
    assert "External comparison evidence must remain functional_non_isolated" in text
    assert "diff-qnode-external-comparison" in text
    assert "No provider or QPU execution" in text


def test_isolated_benchmark_runner_requires_manual_main_opt_in() -> None:
    """Assert that self-hosted benchmark execution is manual and main-bound."""
    text = WORKFLOW.read_text(encoding="utf-8")
    isolated_block = text.split("differentiable-isolated-benchmark:", maxsplit=1)[1].split(
        "  security:",
        maxsplit=1,
    )[0]

    assert "workflow_dispatch:" in text
    assert "run_isolated_benchmark:" in text
    assert "default: false" in text
    assert "type: boolean" in text
    assert all(trigger not in text for trigger in RISKY_SELF_HOSTED_TRIGGERS)
    assert "github.event_name == 'workflow_dispatch'" in isolated_block
    assert "inputs.run_isolated_benchmark == true" in isolated_block
    assert "github.ref == 'refs/heads/main'" in isolated_block
    assert "permissions:" in isolated_block
    assert "contents: read" in isolated_block
    assert "timeout-minutes:" in isolated_block
    assert "ref: refs/heads/main" in isolated_block
    assert "persist-credentials: false" in isolated_block
