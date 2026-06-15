# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable benchmark evidence metadata.
"""CI-only benchmark evidence metadata and artefact writers."""

from __future__ import annotations

import csv
import importlib.metadata as importlib_metadata
import json
import os
import platform
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AcceleratorEvidenceMetadata:
    """Explicit accelerator metadata for benchmark claim boundaries."""

    requested_backend: str
    detected_backend: str
    device_ids: tuple[str, ...]
    device_names: tuple[str, ...]
    runtime_versions: dict[str, str]
    cpu_fallback_detected: bool
    claim_boundary: str

    @classmethod
    def cpu_only(cls) -> AcceleratorEvidenceMetadata:
        """Return explicit CPU-only accelerator metadata."""

        return cls(
            requested_backend="cpu",
            detected_backend="cpu",
            device_ids=(),
            device_names=(),
            runtime_versions={},
            cpu_fallback_detected=False,
            claim_boundary=(
                "CPU-only differentiable benchmark evidence. This row carries no CUDA, "
                "ROCm, GPU, provider, or QPU performance claim."
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready accelerator metadata."""

        return {
            "requested_backend": self.requested_backend,
            "detected_backend": self.detected_backend,
            "device_ids": list(self.device_ids),
            "device_names": list(self.device_names),
            "runtime_versions": dict(self.runtime_versions),
            "cpu_fallback_detected": self.cpu_fallback_detected,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class BenchmarkIsolationMetadata:
    """Isolation metadata required before benchmark evidence can be promoted."""

    command: tuple[str, ...]
    runner_type: str
    runner_name: str
    runner_labels: tuple[str, ...]
    github_run_id: str | None
    commit_sha: str | None
    runner_os: str
    python_version: str
    platform: str
    cpu_count: int | None
    cpu_affinity: str | None
    isolation_method: str | None
    load_before: tuple[float, float, float] | None
    load_after: tuple[float, float, float] | None
    governor: str | None
    frequency_mhz: float | None
    heavy_jobs_running: bool
    accelerator: AcceleratorEvidenceMetadata
    classification: str
    failure_class: str | None
    production_eligible: bool
    gap_reason: str | None

    @classmethod
    def from_ci_environment(
        cls,
        env: Mapping[str, str],
        *,
        command: Sequence[str],
        cpu_affinity: str | None,
        isolation_method: str | None,
        load_before: tuple[float, float, float] | None,
        load_after: tuple[float, float, float] | None,
        governor: str | None,
        frequency_mhz: float | None,
        heavy_jobs_running: bool,
        accelerator_metadata: AcceleratorEvidenceMetadata | None = None,
    ) -> BenchmarkIsolationMetadata:
        """Classify a CI benchmark run from runner metadata."""

        accelerator = accelerator_metadata or capture_accelerator_metadata(env)
        labels = tuple(
            label.strip() for label in env.get("RUNNER_LABELS", "").split(",") if label.strip()
        )
        runner_environment = env.get("RUNNER_ENVIRONMENT", "")
        runner_type = "self-hosted" if runner_environment == "self-hosted" else "github-hosted"
        has_isolated_label = "isolated-benchmark" in labels
        cpu_count = os.cpu_count()
        has_required_context = all(
            (
                cpu_affinity,
                isolation_method,
                load_before is not None,
                load_after is not None,
                governor,
                frequency_mhz is not None,
            )
        )
        has_low_host_load = _host_load_is_low(load_before) and _host_load_is_low(load_after)
        production_eligible = (
            runner_type == "self-hosted"
            and has_isolated_label
            and has_required_context
            and has_low_host_load
            and not heavy_jobs_running
            and not accelerator.cpu_fallback_detected
        )
        if accelerator.cpu_fallback_detected:
            classification = "hard_gap"
            failure_class = "silent_accelerator_fallback"
            gap_reason = (
                f"Benchmark requested {accelerator.requested_backend} accelerator execution, "
                f"but detected backend is {accelerator.detected_backend} with no visible "
                "device evidence. CPU fallback cannot support accelerator benchmark claims."
            )
        elif production_eligible:
            classification = "isolated_affinity"
            failure_class = None
            gap_reason = None
        elif runner_type == "self-hosted" and has_isolated_label:
            classification = "hard_gap"
            if not has_required_context:
                failure_class = "insufficient_isolation_metadata"
                gap_reason = (
                    "Self-hosted isolated benchmark runner did not provide complete CPU "
                    "affinity, host load, governor/frequency, and heavy-job metadata."
                )
            elif not has_low_host_load:
                failure_class = "host_load_not_isolated"
                gap_reason = (
                    "Self-hosted isolated benchmark runner reported host load above the "
                    "isolated promotion threshold before or after the run."
                )
            else:
                failure_class = "heavy_concurrent_jobs"
                gap_reason = (
                    "Self-hosted isolated benchmark runner reported concurrent heavy jobs."
                )
        else:
            classification = "functional_non_isolated"
            failure_class = "non_isolated_runner"
            gap_reason = (
                "Production benchmark promotion requires a self-hosted isolated benchmark "
                "runner, CPU affinity, isolation method, host load, governor/frequency "
                "context, and no concurrent heavy jobs."
            )
        return cls(
            command=tuple(command),
            runner_type=runner_type,
            runner_name=env.get("RUNNER_NAME", ""),
            runner_labels=labels,
            github_run_id=env.get("GITHUB_RUN_ID"),
            commit_sha=env.get("GITHUB_SHA"),
            runner_os=env.get("RUNNER_OS", platform.system()),
            python_version=platform.python_version(),
            platform=platform.platform(),
            cpu_count=cpu_count,
            cpu_affinity=cpu_affinity,
            isolation_method=isolation_method,
            load_before=load_before,
            load_after=load_after,
            governor=governor,
            frequency_mhz=frequency_mhz,
            heavy_jobs_running=heavy_jobs_running,
            accelerator=accelerator,
            classification=classification,
            failure_class=failure_class,
            production_eligible=production_eligible,
            gap_reason=gap_reason,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready metadata."""

        return {
            "command": list(self.command),
            "runner_type": self.runner_type,
            "runner_name": self.runner_name,
            "runner_labels": list(self.runner_labels),
            "github_run_id": self.github_run_id,
            "commit_sha": self.commit_sha,
            "runner_os": self.runner_os,
            "python_version": self.python_version,
            "platform": self.platform,
            "cpu_count": self.cpu_count,
            "cpu_affinity": self.cpu_affinity,
            "isolation_method": self.isolation_method,
            "load_before": list(self.load_before) if self.load_before is not None else None,
            "load_after": list(self.load_after) if self.load_after is not None else None,
            "governor": self.governor,
            "frequency_mhz": self.frequency_mhz,
            "heavy_jobs_running": self.heavy_jobs_running,
            "accelerator": self.accelerator.to_dict(),
            "classification": self.classification,
            "failure_class": self.failure_class,
            "production_eligible": self.production_eligible,
            "gap_reason": self.gap_reason,
        }


@dataclass(frozen=True)
class DifferentiableBenchmarkEvidenceBundle:
    """Paths and metadata for one written differentiable benchmark bundle."""

    artifact_id: str
    classification: str
    raw_json_path: Path
    csv_path: Path
    markdown_path: Path
    generated_at_epoch: float


def write_differentiable_benchmark_evidence_bundle(
    output_dir: Path,
    *,
    metadata: BenchmarkIsolationMetadata,
    timing_rows: Sequence[Mapping[str, object]],
    artifact_id: str | None = None,
    external_artifact_ids: Sequence[str] | None = None,
) -> DifferentiableBenchmarkEvidenceBundle:
    """Write raw JSON, CSV timing rows, and Markdown summary for benchmark evidence."""

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = time.time()
    resolved_artifact_id = artifact_id or (
        f"diff-qnode-{metadata.classification}-{metadata.github_run_id or 'local'}"
    )
    raw_path = output_dir / f"{resolved_artifact_id}.json"
    csv_path = output_dir / f"{resolved_artifact_id}.csv"
    markdown_path = output_dir / f"{resolved_artifact_id}.md"
    rows = [dict(row) for row in timing_rows]
    companion_artifact_ids = tuple(
        external_artifact_ids or ("diff-qnode-external-comparison-schema-v1",)
    )
    payload = {
        "schema": "scpn_qc_differentiable_benchmark_evidence_v1",
        "artifact_id": resolved_artifact_id,
        "generated_at_epoch": generated_at,
        "metadata": metadata.to_dict(),
        "timing_rows": rows,
        "evidence_artifact_ids": [resolved_artifact_id, *companion_artifact_ids],
        "claim_boundary": (
            "CI benchmark evidence only. functional_non_isolated rows are parity/local "
            "regression artefacts and cannot support production performance claims."
        ),
    }
    raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    markdown_path.write_text(
        "\n".join(
            (
                "# Differentiable Phase-QNode Benchmark Evidence",
                "",
                f"- Artefact ID: `{resolved_artifact_id}`",
                f"- Classification: `{metadata.classification}`",
                f"- Production eligible: `{metadata.production_eligible}`",
                f"- Failure class: {metadata.failure_class or 'none'}",
                f"- Gap reason: {metadata.gap_reason or 'none'}",
                f"- Accelerator requested: `{metadata.accelerator.requested_backend}`",
                f"- Accelerator detected: `{metadata.accelerator.detected_backend}`",
                f"- Accelerator CPU fallback: `{metadata.accelerator.cpu_fallback_detected}`",
                "",
                "No provider or QPU execution is performed by this evidence writer.",
                "",
            )
        ),
        encoding="utf-8",
    )
    return DifferentiableBenchmarkEvidenceBundle(
        artifact_id=resolved_artifact_id,
        classification=metadata.classification,
        raw_json_path=raw_path,
        csv_path=csv_path,
        markdown_path=markdown_path,
        generated_at_epoch=generated_at,
    )


def capture_host_load() -> tuple[float, float, float] | None:
    """Return host load averages when the platform exposes them."""

    try:
        one, five, fifteen = os.getloadavg()
        return (float(one), float(five), float(fifteen))
    except (AttributeError, OSError):
        return None


def read_cpu_governor(cpu_index: int = 0) -> str | None:
    """Read Linux CPU frequency governor metadata when available."""

    path = Path(f"/sys/devices/system/cpu/cpu{cpu_index}/cpufreq/scaling_governor")
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8", errors="replace").strip()
    return value or None


def read_cpu_frequency_mhz(cpu_index: int = 0) -> float | None:
    """Read Linux CPU frequency metadata in MHz when available."""

    sysfs_path = Path(f"/sys/devices/system/cpu/cpu{cpu_index}/cpufreq/scaling_cur_freq")
    if sysfs_path.exists():
        value = sysfs_path.read_text(encoding="utf-8", errors="replace").strip()
        if value:
            return float(value) / 1000.0
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("cpu mhz"):
                return float(line.split(":", maxsplit=1)[1].strip())
    return None


def capture_accelerator_metadata(env: Mapping[str, str]) -> AcceleratorEvidenceMetadata:
    """Capture explicit accelerator metadata from deterministic benchmark environment."""

    requested_backend = _normalise_accelerator_backend(
        env.get("SCPN_BENCH_ACCELERATOR_BACKEND") or env.get("SCPN_ACCELERATOR_BACKEND") or "cpu"
    )
    if env.get("SCPN_BENCH_ACCELERATOR_DISABLE_DISCOVERY") == "1":
        probed_device_ids: tuple[str, ...] = ()
        probed_device_names: tuple[str, ...] = ()
        probed_runtime_versions: dict[str, str] = {}
    else:
        probed_device_ids, probed_device_names, probed_runtime_versions = (
            _probe_requested_accelerator(requested_backend)
        )
    device_ids = _accelerator_device_ids(env, requested_backend) or probed_device_ids
    detected_backend = _detected_accelerator_backend(requested_backend, device_ids, env)
    cpu_fallback_detected = requested_backend not in {"cpu", "none"} and (
        detected_backend != requested_backend or len(device_ids) == 0
    )
    return AcceleratorEvidenceMetadata(
        requested_backend=requested_backend,
        detected_backend=detected_backend,
        device_ids=device_ids,
        device_names=_split_metadata_list(env.get("SCPN_BENCH_ACCELERATOR_DEVICE_NAMES", ""))
        or probed_device_names,
        runtime_versions={
            **probed_runtime_versions,
            **_runtime_version_metadata(
                env.get("SCPN_BENCH_ACCELERATOR_RUNTIME")
                or env.get("SCPN_ACCELERATOR_RUNTIME")
                or ""
            ),
        },
        cpu_fallback_detected=cpu_fallback_detected,
        claim_boundary=_accelerator_claim_boundary(
            requested_backend=requested_backend,
            detected_backend=detected_backend,
            cpu_fallback_detected=cpu_fallback_detected,
        ),
    )


def infer_heavy_jobs_running(load: tuple[float, float, float] | None) -> bool:
    """Infer whether current host load is too high for production promotion."""

    cpu_count = os.cpu_count() or 1
    return bool(load and load[0] > max(1.0, cpu_count * 0.75))


def _host_load_is_low(load: tuple[float, float, float] | None) -> bool:
    return load is not None and max(load) <= 1.0


def _normalise_accelerator_backend(value: str) -> str:
    backend = value.strip().lower()
    aliases = {
        "gpu": "cuda",
        "nvidia": "cuda",
        "hip": "rocm",
        "amd": "rocm",
        "none": "cpu",
        "": "cpu",
    }
    return aliases.get(backend, backend)


def _accelerator_device_ids(env: Mapping[str, str], requested_backend: str) -> tuple[str, ...]:
    explicit_ids = _visible_device_ids(env.get("SCPN_BENCH_ACCELERATOR_DEVICE_IDS", ""))
    if explicit_ids:
        return explicit_ids
    if requested_backend == "cuda":
        return _visible_device_ids(env.get("CUDA_VISIBLE_DEVICES", ""))
    if requested_backend == "rocm":
        visible = env.get("ROCR_VISIBLE_DEVICES") or env.get("HIP_VISIBLE_DEVICES") or ""
        return _visible_device_ids(visible)
    return ()


def _probe_requested_accelerator(
    requested_backend: str,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, str]]:
    if requested_backend != "cuda":
        return (), (), {}
    try:
        import jax
    except Exception:
        return (), (), {}
    try:
        devices = tuple(jax.devices("gpu"))
    except Exception:
        return (), (), _jax_runtime_versions()
    return (
        tuple(str(index) for index, _device in enumerate(devices)),
        tuple(str(device) for device in devices),
        _jax_runtime_versions(),
    )


def _jax_runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt"):
        try:
            versions[package] = str(importlib_metadata.version(package))
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


def _detected_accelerator_backend(
    requested_backend: str,
    device_ids: tuple[str, ...],
    env: Mapping[str, str],
) -> str:
    if requested_backend in {"cuda", "rocm"} and device_ids:
        return requested_backend
    if _visible_device_ids(env.get("CUDA_VISIBLE_DEVICES", "")):
        return "cuda"
    if _visible_device_ids(
        env.get("ROCR_VISIBLE_DEVICES") or env.get("HIP_VISIBLE_DEVICES") or ""
    ):
        return "rocm"
    return "cpu"


def _visible_device_ids(value: str) -> tuple[str, ...]:
    devices = _split_metadata_list(value)
    blocked = {"", "-1", "none", "no", "void"}
    return tuple(device for device in devices if device.lower() not in blocked)


def _split_metadata_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _runtime_version_metadata(value: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for item in _split_metadata_list(value):
        if "=" not in item:
            metadata[item] = "unknown"
            continue
        key, version = item.split("=", maxsplit=1)
        key = key.strip().lower()
        version = version.strip()
        if key and version:
            metadata[key] = version
    return metadata


def _accelerator_claim_boundary(
    *,
    requested_backend: str,
    detected_backend: str,
    cpu_fallback_detected: bool,
) -> str:
    if cpu_fallback_detected:
        return (
            f"{requested_backend} accelerator execution was requested, but benchmark metadata "
            "does not prove a visible accelerator device. Treat the artefact as a hard gap, "
            "not accelerator evidence."
        )
    if requested_backend in {"cuda", "rocm"} and detected_backend == requested_backend:
        return (
            f"{requested_backend} accelerator metadata is present for the benchmark host. "
            "This records device visibility only; production performance claims still require "
            "isolated benchmark classification and matching timing rows."
        )
    return (
        "CPU-only differentiable benchmark evidence. This row carries no CUDA, ROCm, GPU, "
        "provider, or QPU performance claim."
    )


__all__ = [
    "AcceleratorEvidenceMetadata",
    "BenchmarkIsolationMetadata",
    "DifferentiableBenchmarkEvidenceBundle",
    "capture_accelerator_metadata",
    "capture_host_load",
    "infer_heavy_jobs_running",
    "read_cpu_frequency_mhz",
    "read_cpu_governor",
    "write_differentiable_benchmark_evidence_bundle",
]
