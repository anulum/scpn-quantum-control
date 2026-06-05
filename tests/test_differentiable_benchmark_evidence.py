# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable Benchmark Evidence Tests
"""Tests for CI-only differentiable benchmark evidence classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scpn_quantum_control.benchmarks.differentiable_evidence import (
    BenchmarkIsolationMetadata,
    write_differentiable_benchmark_evidence_bundle,
)


def test_github_hosted_benchmark_metadata_downgrades_to_functional_non_isolated() -> None:
    metadata = BenchmarkIsolationMetadata.from_ci_environment(
        {
            "GITHUB_RUN_ID": "12345",
            "GITHUB_SHA": "abc123",
            "RUNNER_NAME": "GitHub Actions 42",
            "RUNNER_ENVIRONMENT": "github-hosted",
            "RUNNER_LABELS": "ubuntu-latest,linux",
        },
        command=("python", "scripts/run_differentiable_benchmark_evidence.py"),
        cpu_affinity=None,
        isolation_method=None,
        load_before=(8.0, 7.0, 6.0),
        load_after=(8.2, 7.1, 6.2),
        governor="performance",
        frequency_mhz=3200.0,
        heavy_jobs_running=True,
    )

    assert metadata.runner_type == "github-hosted"
    assert metadata.classification == "functional_non_isolated"
    assert not metadata.production_eligible
    assert metadata.github_run_id == "12345"
    assert metadata.commit_sha == "abc123"
    assert "self-hosted isolated benchmark runner" in metadata.gap_reason


def test_self_hosted_isolated_metadata_promotes_only_with_all_required_context() -> None:
    metadata = BenchmarkIsolationMetadata.from_ci_environment(
        {
            "GITHUB_RUN_ID": "67890",
            "GITHUB_SHA": "def456",
            "RUNNER_NAME": "isolated-qc-runner",
            "RUNNER_ENVIRONMENT": "self-hosted",
            "RUNNER_LABELS": "self-hosted,linux,isolated-benchmark",
        },
        command=(
            "taskset",
            "-c",
            "2",
            "python",
            "scripts/run_differentiable_benchmark_evidence.py",
        ),
        cpu_affinity="2",
        isolation_method="taskset",
        load_before=(0.05, 0.04, 0.03),
        load_after=(0.06, 0.05, 0.04),
        governor="performance",
        frequency_mhz=3200.0,
        heavy_jobs_running=False,
    )

    assert metadata.runner_type == "self-hosted"
    assert metadata.classification == "isolated_affinity"
    assert metadata.production_eligible
    assert metadata.gap_reason is None


def test_benchmark_evidence_bundle_writes_json_csv_and_markdown_with_artifact_ids(
    tmp_path: Path,
) -> None:
    metadata = BenchmarkIsolationMetadata.from_ci_environment(
        {"GITHUB_RUN_ID": "12345", "GITHUB_SHA": "abc123", "RUNNER_ENVIRONMENT": "github-hosted"},
        command=("python", "scripts/run_differentiable_benchmark_evidence.py"),
        cpu_affinity=None,
        isolation_method=None,
        load_before=None,
        load_after=None,
        governor=None,
        frequency_mhz=None,
        heavy_jobs_running=True,
    )
    bundle = write_differentiable_benchmark_evidence_bundle(
        tmp_path,
        metadata=metadata,
        timing_rows=(
            {
                "case_id": "phase_qnode_vector_grad",
                "backend": "scpn_reference",
                "runtime_seconds": 0.001,
                "memory_peak_bytes": 4096,
                "value_error": 0.0,
                "gradient_error": 0.0,
            },
        ),
    )

    assert bundle.raw_json_path.exists()
    assert bundle.csv_path.exists()
    assert bundle.markdown_path.exists()
    assert bundle.artifact_id.startswith("diff-qnode-")
    assert bundle.classification == "functional_non_isolated"

    raw = bundle.raw_json_path.read_text(encoding="utf-8")
    assert bundle.artifact_id in raw
    assert "functional_non_isolated" in raw
    assert "phase_qnode_vector_grad" in bundle.csv_path.read_text(encoding="utf-8")
    assert "functional_non_isolated" in bundle.markdown_path.read_text(encoding="utf-8")
    assert np.isfinite(bundle.generated_at_epoch)
