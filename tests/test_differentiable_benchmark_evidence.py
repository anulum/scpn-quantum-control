# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable Benchmark Evidence Tests
"""Tests for CI-only differentiable benchmark evidence classification."""

from __future__ import annotations

import builtins
import importlib.metadata as importlib_metadata
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from scpn_quantum_control.benchmarks import differentiable_evidence as _evidence
from scpn_quantum_control.benchmarks.differentiable_evidence import (
    AcceleratorEvidenceMetadata,
    BenchmarkIsolationMetadata,
    capture_accelerator_metadata,
    capture_host_load,
    infer_heavy_jobs_running,
    read_cpu_frequency_mhz,
    read_cpu_governor,
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
    assert metadata.gap_reason is not None
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
    assert metadata.gap_reason is not None
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


def test_requested_rocm_uses_visible_rocm_devices() -> None:
    accelerator_metadata = capture_accelerator_metadata(
        {
            "SCPN_BENCH_ACCELERATOR_BACKEND": "hip",
            "ROCR_VISIBLE_DEVICES": "0,none,1",
            "SCPN_BENCH_ACCELERATOR_RUNTIME": "rocm=6.1,hip",
            "SCPN_BENCH_ACCELERATOR_DISABLE_DISCOVERY": "1",
        }
    )

    assert accelerator_metadata.requested_backend == "rocm"
    assert accelerator_metadata.detected_backend == "rocm"
    assert accelerator_metadata.device_ids == ("0", "1")
    assert accelerator_metadata.runtime_versions == {"rocm": "6.1", "hip": "unknown"}
    assert "rocm accelerator metadata is present" in accelerator_metadata.claim_boundary


def test_detected_backend_uses_visible_accelerators_without_requested_backend() -> None:
    cuda_metadata = capture_accelerator_metadata(
        {
            "SCPN_BENCH_ACCELERATOR_DISABLE_DISCOVERY": "1",
            "CUDA_VISIBLE_DEVICES": "2",
        }
    )
    rocm_metadata = capture_accelerator_metadata(
        {
            "SCPN_BENCH_ACCELERATOR_DISABLE_DISCOVERY": "1",
            "HIP_VISIBLE_DEVICES": "3",
        }
    )

    assert cuda_metadata.detected_backend == "cuda"
    assert not cuda_metadata.cpu_fallback_detected
    assert rocm_metadata.detected_backend == "rocm"
    assert not rocm_metadata.cpu_fallback_detected


def test_cuda_probe_without_jax_returns_empty_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def fake_import(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "jax":
            raise ImportError("jax unavailable")
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert _evidence._probe_requested_accelerator("cuda") == ((), (), {})


def test_cuda_probe_records_runtime_versions_when_device_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_jax = ModuleType("jax")

    def fake_devices(_kind: str) -> tuple[object, ...]:
        raise RuntimeError("gpu probe failed")

    fake_jax.devices = fake_devices  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "jax", fake_jax)
    monkeypatch.setattr(_evidence, "_jax_runtime_versions", lambda: {"jax": "test"})

    assert _evidence._probe_requested_accelerator("cuda") == ((), (), {"jax": "test"})


def test_jax_runtime_versions_skip_missing_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_version(package: str) -> str:
        if package == "jax":
            return "1.2.3"
        raise importlib_metadata.PackageNotFoundError(package)

    monkeypatch.setattr(
        "scpn_quantum_control.benchmarks.differentiable_evidence.importlib_metadata.version",
        fake_version,
    )

    assert _evidence._jax_runtime_versions() == {"jax": "1.2.3"}


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
        external_artifact_ids=("diff-qnode-external-comparison-local",),
    )

    assert bundle.raw_json_path.exists()
    assert bundle.csv_path.exists()
    assert bundle.markdown_path.exists()
    assert bundle.artifact_id.startswith("diff-qnode-")
    assert bundle.classification == "functional_non_isolated"

    raw = bundle.raw_json_path.read_text(encoding="utf-8")
    assert bundle.artifact_id in raw
    assert "functional_non_isolated" in raw
    assert "diff-qnode-external-comparison-local" in raw
    assert "non_isolated_runner" in raw
    assert "phase_qnode_vector_grad" in bundle.csv_path.read_text(encoding="utf-8")
    assert "functional_non_isolated" in bundle.markdown_path.read_text(encoding="utf-8")
    assert np.isfinite(bundle.generated_at_epoch)


def test_heavy_job_inference_uses_cpu_scaled_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("os.cpu_count", lambda: 4)

    assert infer_heavy_jobs_running((4.0, 1.0, 1.0))
    assert not infer_heavy_jobs_running((1.0, 1.0, 1.0))


def test_capture_host_load_returns_float_tuple(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "getloadavg", lambda: (1, 2, 3))

    assert capture_host_load() == (1.0, 2.0, 3.0)


def test_capture_host_load_returns_none_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_load() -> tuple[float, float, float]:
        raise OSError("load unavailable")

    monkeypatch.setattr(os, "getloadavg", missing_load)

    assert capture_host_load() is None


class _FakeSystemPath:
    def __init__(self, value: str, files: dict[str, str]) -> None:
        self._value = value
        self._files = files

    def exists(self) -> bool:
        return self._value in self._files

    def read_text(self, *, encoding: str, errors: str = "strict") -> str:
        return self._files[self._value]


def _fake_system_path_factory(files: dict[str, str]) -> type[_FakeSystemPath]:
    class FakeSystemPath(_FakeSystemPath):
        def __init__(self, value: str) -> None:
            super().__init__(value, files)

    return FakeSystemPath


def test_read_cpu_governor_handles_present_empty_and_missing_sysfs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {"/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor": "performance\n"}
    monkeypatch.setattr(_evidence, "Path", _fake_system_path_factory(files))
    assert read_cpu_governor() == "performance"

    files["/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"] = "\n"
    assert read_cpu_governor() is None

    files.clear()
    assert read_cpu_governor() is None


def test_read_cpu_frequency_prefers_sysfs_and_falls_back_to_cpuinfo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {"/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq": "3200000\n"}
    monkeypatch.setattr(_evidence, "Path", _fake_system_path_factory(files))
    assert read_cpu_frequency_mhz() == 3200.0

    files.clear()
    files["/proc/cpuinfo"] = "processor: 0\ncpu MHz\t\t: 2419.250\n"
    assert read_cpu_frequency_mhz() == 2419.25

    files.clear()
    assert read_cpu_frequency_mhz() is None


def test_read_cpu_frequency_ignores_empty_sysfs_before_cpuinfo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq": "\n",
        "/proc/cpuinfo": "cpu MHz\t\t: 1000.000\n",
    }
    monkeypatch.setattr(_evidence, "Path", _fake_system_path_factory(files))

    assert read_cpu_frequency_mhz() == 1000.0


def test_read_cpu_frequency_returns_none_when_cpuinfo_has_no_frequency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq": "\n",
        "/proc/cpuinfo": "processor: 0\nmodel name: test\n",
    }
    monkeypatch.setattr(_evidence, "Path", _fake_system_path_factory(files))

    assert read_cpu_frequency_mhz() is None


def test_runtime_version_metadata_ignores_incomplete_pairs() -> None:
    metadata = capture_accelerator_metadata(
        {
            "SCPN_BENCH_ACCELERATOR_DISABLE_DISCOVERY": "1",
            "SCPN_BENCH_ACCELERATOR_RUNTIME": "valid=1.0,missing_version=,=missing_key",
        }
    )

    assert metadata.runtime_versions == {"valid": "1.0"}
