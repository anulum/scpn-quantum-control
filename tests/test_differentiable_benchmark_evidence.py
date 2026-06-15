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
    AcceleratorEvidenceMetadata,
    BenchmarkIsolationMetadata,
    capture_accelerator_metadata,
    infer_heavy_jobs_running,
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
    assert metadata.failure_class == "non_isolated_runner"
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
        accelerator_metadata=AcceleratorEvidenceMetadata.cpu_only(),
    )

    assert metadata.runner_type == "self-hosted"
    assert metadata.classification == "isolated_affinity"
    assert metadata.failure_class is None
    assert metadata.production_eligible
    assert metadata.gap_reason is None


def test_self_hosted_isolated_metadata_requires_full_host_context() -> None:
    metadata = BenchmarkIsolationMetadata.from_ci_environment(
        {
            "GITHUB_RUN_ID": "67890",
            "GITHUB_SHA": "def456",
            "RUNNER_NAME": "isolated-qc-runner",
            "RUNNER_ENVIRONMENT": "self-hosted",
            "RUNNER_LABELS": "self-hosted,linux,isolated-benchmark",
        },
        command=("python", "scripts/run_differentiable_benchmark_evidence.py"),
        cpu_affinity="2",
        isolation_method="taskset",
        load_before=None,
        load_after=None,
        governor=None,
        frequency_mhz=None,
        heavy_jobs_running=False,
        accelerator_metadata=AcceleratorEvidenceMetadata.cpu_only(),
    )

    assert metadata.classification == "hard_gap"
    assert metadata.failure_class == "insufficient_isolation_metadata"
    assert not metadata.production_eligible


def test_self_hosted_isolated_metadata_rejects_high_host_load() -> None:
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
        load_after=(1.25, 0.95, 0.70),
        governor="performance",
        frequency_mhz=3200.0,
        heavy_jobs_running=False,
        accelerator_metadata=AcceleratorEvidenceMetadata.cpu_only(),
    )

    assert metadata.classification == "hard_gap"
    assert metadata.failure_class == "host_load_not_isolated"
    assert not metadata.production_eligible


def test_self_hosted_isolated_metadata_rejects_heavy_jobs() -> None:
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
        heavy_jobs_running=True,
        accelerator_metadata=AcceleratorEvidenceMetadata.cpu_only(),
    )

    assert metadata.classification == "hard_gap"
    assert metadata.failure_class == "heavy_concurrent_jobs"
    assert not metadata.production_eligible


def test_requested_cuda_without_visible_device_is_hard_gap() -> None:
    accelerator_metadata = capture_accelerator_metadata(
        {
            "SCPN_BENCH_ACCELERATOR_BACKEND": "cuda",
            "SCPN_BENCH_ACCELERATOR_DISABLE_DISCOVERY": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "SCPN_BENCH_ACCELERATOR_RUNTIME": "cuda=12.4",
        }
    )
    metadata = BenchmarkIsolationMetadata.from_ci_environment(
        {
            "GITHUB_RUN_ID": "67890",
            "GITHUB_SHA": "def456",
            "RUNNER_NAME": "isolated-qc-runner",
            "RUNNER_ENVIRONMENT": "self-hosted",
            "RUNNER_LABELS": "self-hosted,linux,isolated-benchmark",
        },
        command=("python", "scripts/run_differentiable_benchmark_evidence.py"),
        cpu_affinity="2",
        isolation_method="taskset",
        load_before=(0.05, 0.04, 0.03),
        load_after=(0.06, 0.05, 0.04),
        governor="performance",
        frequency_mhz=3200.0,
        heavy_jobs_running=False,
        accelerator_metadata=accelerator_metadata,
    )

    assert accelerator_metadata.requested_backend == "cuda"
    assert accelerator_metadata.detected_backend == "cpu"
    assert accelerator_metadata.cpu_fallback_detected
    assert metadata.classification == "hard_gap"
    assert metadata.failure_class == "silent_accelerator_fallback"
    assert not metadata.production_eligible
    assert "requested cuda" in metadata.gap_reason


def test_requested_cuda_with_visible_device_keeps_isolated_promotion() -> None:
    accelerator_metadata = capture_accelerator_metadata(
        {
            "SCPN_BENCH_ACCELERATOR_BACKEND": "cuda",
            "CUDA_VISIBLE_DEVICES": "0",
            "SCPN_BENCH_ACCELERATOR_DEVICE_NAMES": "NVIDIA L40S",
            "SCPN_BENCH_ACCELERATOR_RUNTIME": "cuda=12.4,cudnn=9.1",
        }
    )
    metadata = BenchmarkIsolationMetadata.from_ci_environment(
        {
            "GITHUB_RUN_ID": "67890",
            "GITHUB_SHA": "def456",
            "RUNNER_NAME": "isolated-qc-runner",
            "RUNNER_ENVIRONMENT": "self-hosted",
            "RUNNER_LABELS": "self-hosted,linux,isolated-benchmark",
        },
        command=("python", "scripts/run_differentiable_benchmark_evidence.py"),
        cpu_affinity="2",
        isolation_method="taskset",
        load_before=(0.05, 0.04, 0.03),
        load_after=(0.06, 0.05, 0.04),
        governor="performance",
        frequency_mhz=3200.0,
        heavy_jobs_running=False,
        accelerator_metadata=accelerator_metadata,
    )

    assert accelerator_metadata.detected_backend == "cuda"
    assert accelerator_metadata.device_ids == ("0",)
    assert accelerator_metadata.device_names == ("NVIDIA L40S",)
    assert accelerator_metadata.runtime_versions["cuda"] == "12.4"
    assert accelerator_metadata.runtime_versions["cudnn"] == "9.1"
    assert metadata.classification == "isolated_affinity"
    assert metadata.failure_class is None


def test_requested_cuda_accepts_explicit_device_metadata() -> None:
    accelerator_metadata = capture_accelerator_metadata(
        {
            "SCPN_BENCH_ACCELERATOR_BACKEND": "cuda",
            "SCPN_BENCH_ACCELERATOR_DEVICE_IDS": "0,1",
            "SCPN_BENCH_ACCELERATOR_DEVICE_NAMES": "NVIDIA L40S,NVIDIA L40S",
        }
    )

    assert accelerator_metadata.requested_backend == "cuda"
    assert accelerator_metadata.detected_backend == "cuda"
    assert accelerator_metadata.device_ids == ("0", "1")
    assert accelerator_metadata.device_names == ("NVIDIA L40S", "NVIDIA L40S")
    assert not accelerator_metadata.cpu_fallback_detected


def test_evidence_bundle_serialises_accelerator_metadata(tmp_path: Path) -> None:
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
        accelerator_metadata=AcceleratorEvidenceMetadata.cpu_only(),
    )
    bundle = write_differentiable_benchmark_evidence_bundle(
        tmp_path,
        metadata=metadata,
        timing_rows=(),
    )

    payload = bundle.raw_json_path.read_text(encoding="utf-8")
    assert '"accelerator"' in payload
    assert '"requested_backend": "cpu"' in payload
    assert '"cpu_fallback_detected": false' in payload


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
    assert "diff-qnode-external-comparison-schema-v1" in raw
    assert "non_isolated_runner" in raw
    assert "phase_qnode_vector_grad" in bundle.csv_path.read_text(encoding="utf-8")
    assert "functional_non_isolated" in bundle.markdown_path.read_text(encoding="utf-8")
    assert np.isfinite(bundle.generated_at_epoch)


def test_heavy_job_inference_uses_cpu_scaled_load(monkeypatch) -> None:
    monkeypatch.setattr("os.cpu_count", lambda: 4)

    assert infer_heavy_jobs_running((4.0, 1.0, 1.0))
    assert not infer_heavy_jobs_running((1.0, 1.0, 1.0))
