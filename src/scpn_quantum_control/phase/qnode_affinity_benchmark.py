# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase QNode Affinity Benchmark
"""Small Phase-QNode benchmark harness with isolation metadata."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from .qnode_circuit import (
    PauliTerm,
    PhaseQNodeCircuit,
    execute_phase_qnode_circuit,
    parameter_shift_phase_qnode_gradient,
)

FloatArray: TypeAlias = NDArray[np.float64]
EvidenceLabel = Literal["isolated_affinity", "functional_non_isolated"]


@dataclass(frozen=True)
class PhaseQNodeAffinityBenchmarkMetadata:
    """Benchmark command and host metadata."""

    command: str
    affinity_cpus: tuple[int, ...]
    observed_affinity_cpus: tuple[int, ...]
    isolation_method: str
    host_load_before: tuple[float, float, float]
    host_load_after: tuple[float, float, float]
    cpu_model: str
    governor: str
    frequency_mhz: tuple[float, ...]
    python_version: str
    dependency_versions: dict[str, str]
    runner_environment: str
    runner_labels: tuple[str, ...]
    repetitions: int
    warmups: int
    heavy_concurrent_jobs: bool

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready benchmark metadata."""
        return {
            "command": self.command,
            "affinity_cpus": list(self.affinity_cpus),
            "observed_affinity_cpus": list(self.observed_affinity_cpus),
            "isolation_method": self.isolation_method,
            "host_load_before": list(self.host_load_before),
            "host_load_after": list(self.host_load_after),
            "cpu_model": self.cpu_model,
            "governor": self.governor,
            "frequency_mhz": list(self.frequency_mhz),
            "python_version": self.python_version,
            "dependency_versions": dict(self.dependency_versions),
            "runner_environment": self.runner_environment,
            "runner_labels": list(self.runner_labels),
            "repetitions": self.repetitions,
            "warmups": self.warmups,
            "heavy_concurrent_jobs": self.heavy_concurrent_jobs,
        }


@dataclass(frozen=True)
class PhaseQNodeAffinityBenchmarkResult:
    """Raw timing rows and isolation classification."""

    evidence_label: EvidenceLabel
    production_benchmark: bool
    metadata: PhaseQNodeAffinityBenchmarkMetadata
    raw_timing_rows: tuple[dict[str, float], ...]
    isolation_failures: tuple[str, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready benchmark evidence."""
        return {
            "evidence_label": self.evidence_label,
            "production_benchmark": self.production_benchmark,
            "metadata": self.metadata.to_dict(),
            "raw_timing_rows": list(self.raw_timing_rows),
            "isolation_failures": list(self.isolation_failures),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class PhaseQNodeAffinityArtifactValidation:
    """Fail-closed attachment verdict for a Phase-QNode benchmark JSON file."""

    artifact_path: str
    artifact_sha256: str
    benchmark_artifact_id: str
    evidence_label: str
    production_benchmark: bool
    promotion_ready: bool
    raw_timing_row_count: int
    missing_requirements: tuple[str, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready validation evidence for claim-ledger attachment."""
        return {
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "benchmark_artifact_id": self.benchmark_artifact_id,
            "evidence_label": self.evidence_label,
            "production_benchmark": self.production_benchmark,
            "promotion_ready": self.promotion_ready,
            "raw_timing_row_count": self.raw_timing_row_count,
            "missing_requirements": list(self.missing_requirements),
            "claim_boundary": self.claim_boundary,
        }


def classify_affinity_evidence(
    *,
    reserved_cpus: tuple[int, ...],
    observed_affinity_cpus: tuple[int, ...],
    host_load_before: tuple[float, float, float],
    host_load_after: tuple[float, float, float],
    command: str,
    governor: str,
    frequency_mhz: tuple[float, ...],
    heavy_concurrent_jobs: bool,
    runner_environment: str = "",
    runner_labels: tuple[str, ...] = (),
) -> EvidenceLabel:
    """Classify benchmark evidence under the core-isolation policy."""
    failures = _isolation_failures(
        reserved_cpus=reserved_cpus,
        observed_affinity_cpus=observed_affinity_cpus,
        host_load_before=host_load_before,
        host_load_after=host_load_after,
        command=command,
        governor=governor,
        frequency_mhz=frequency_mhz,
        heavy_concurrent_jobs=heavy_concurrent_jobs,
        runner_environment=runner_environment,
        runner_labels=runner_labels,
    )
    return "isolated_affinity" if not failures else "functional_non_isolated"


def run_phase_qnode_affinity_benchmark(
    *,
    repetitions: int = 10,
    warmups: int = 3,
    reserved_cpus: tuple[int, ...] | None = None,
    host_load_before: tuple[float, float, float] | None = None,
    host_load_after: tuple[float, float, float] | None = None,
    command: str | None = None,
    heavy_concurrent_jobs: bool = False,
) -> PhaseQNodeAffinityBenchmarkResult:
    """Run a small local Phase-QNode timing loop with reproducibility metadata."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if warmups < 0:
        raise ValueError("warmups must be non-negative")
    requested_affinity = tuple(
        sorted(reserved_cpus if reserved_cpus is not None else _current_affinity())
    )
    if any(cpu < 0 for cpu in requested_affinity):
        raise ValueError("reserved CPU indexes must be non-negative")
    observed_affinity = _current_affinity()
    command_text = command or "python -m scpn_quantum_control.phase.qnode_affinity_benchmark"
    before = host_load_before or _load_average()
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(("ry", (0,), 0), ("cnot", (0, 1)), ("rzz", (0, 1), 1)),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )
    params = np.array([0.2, -0.3], dtype=np.float64)
    rows: list[dict[str, float]] = []
    for index in range(warmups + repetitions):
        start = time.perf_counter()
        value = execute_phase_qnode_circuit(circuit, params).value
        gradient_norm = float(
            np.linalg.norm(parameter_shift_phase_qnode_gradient(circuit, params).gradient)
        )
        elapsed = time.perf_counter() - start
        if index >= warmups:
            rows.append(
                {
                    "iteration": float(index - warmups),
                    "seconds": elapsed,
                    "value": value,
                    "gradient_norm": gradient_norm,
                }
            )
    after = host_load_after or _load_average()
    governor = _cpu_governor()
    frequency_mhz = _cpu_frequency_mhz()
    runner_environment = os.environ.get("RUNNER_ENVIRONMENT", "")
    runner_labels = _runner_labels()
    failures = _isolation_failures(
        reserved_cpus=requested_affinity if reserved_cpus is not None else (),
        observed_affinity_cpus=observed_affinity,
        host_load_before=before,
        host_load_after=after,
        command=command_text,
        governor=governor,
        frequency_mhz=frequency_mhz,
        heavy_concurrent_jobs=heavy_concurrent_jobs,
        runner_environment=runner_environment,
        runner_labels=runner_labels,
    )
    label: EvidenceLabel = "isolated_affinity" if not failures else "functional_non_isolated"
    metadata_row = PhaseQNodeAffinityBenchmarkMetadata(
        command=command_text,
        affinity_cpus=requested_affinity,
        observed_affinity_cpus=observed_affinity,
        isolation_method=_isolation_method(command_text),
        host_load_before=before,
        host_load_after=after,
        cpu_model=_cpu_model(),
        governor=governor,
        frequency_mhz=frequency_mhz,
        python_version=platform.python_version(),
        dependency_versions=_dependency_versions(),
        runner_environment=runner_environment,
        runner_labels=runner_labels,
        repetitions=repetitions,
        warmups=warmups,
        heavy_concurrent_jobs=heavy_concurrent_jobs,
    )
    return PhaseQNodeAffinityBenchmarkResult(
        evidence_label=label,
        production_benchmark=label == "isolated_affinity",
        metadata=metadata_row,
        raw_timing_rows=tuple(rows),
        isolation_failures=tuple(failures),
        claim_boundary=(
            "isolated_affinity only when reserved CPU affinity, low host load, "
            "observed process affinity match, governor or frequency context, fixed "
            "command metadata, and no heavy concurrent jobs are recorded"
        ),
    )


def validate_phase_qnode_affinity_artifact(
    artifact_path: str | Path,
    *,
    require_isolated: bool = True,
) -> PhaseQNodeAffinityArtifactValidation:
    """Validate raw Phase-QNode benchmark JSON before promotion attachment.

    The validator does not upgrade local evidence. It only reports
    ``promotion_ready=True`` when the committed raw JSON already carries the
    isolated label, production flag, timing rows, host metadata, and no
    isolation failures required by the benchmark runner contract.
    """
    path = Path(artifact_path)
    artifact_bytes = path.read_bytes()
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    benchmark_artifact_id = f"phase-qnode-affinity:{artifact_sha256[:16]}"
    payload = _json_object_from_bytes(artifact_bytes, path)
    metadata_payload = _mapping_value(payload, "metadata")
    timing_rows = _list_value(payload, "raw_timing_rows")
    isolation_failures = _list_value(payload, "isolation_failures")

    evidence_label = _string_value(payload, "evidence_label")
    production_benchmark = _bool_value(payload, "production_benchmark")
    raw_timing_row_count = len(timing_rows)
    missing_requirements: list[str] = []

    if require_isolated and evidence_label != "isolated_affinity":
        missing_requirements.append("isolated_affinity evidence label")
    if require_isolated and not production_benchmark:
        missing_requirements.append("production benchmark flag")
    if raw_timing_row_count < 1:
        missing_requirements.append("raw timing rows")
    if require_isolated and isolation_failures:
        missing_requirements.append("empty isolation failures")

    missing_requirements.extend(
        _missing_metadata_requirements(metadata_payload, require_isolated=require_isolated)
    )
    promotion_ready = not missing_requirements
    claim_boundary = _string_value(payload, "claim_boundary") or (
        "Phase-QNode affinity benchmark artefacts are promotional only when "
        "validated as isolated_affinity with raw timing rows and complete host "
        "isolation metadata."
    )
    return PhaseQNodeAffinityArtifactValidation(
        artifact_path=str(path),
        artifact_sha256=artifact_sha256,
        benchmark_artifact_id=benchmark_artifact_id,
        evidence_label=evidence_label,
        production_benchmark=production_benchmark,
        promotion_ready=promotion_ready,
        raw_timing_row_count=raw_timing_row_count,
        missing_requirements=tuple(missing_requirements),
        claim_boundary=claim_boundary,
    )


def _isolation_failures(
    *,
    reserved_cpus: tuple[int, ...],
    observed_affinity_cpus: tuple[int, ...],
    host_load_before: tuple[float, float, float],
    host_load_after: tuple[float, float, float],
    command: str,
    governor: str,
    frequency_mhz: tuple[float, ...],
    heavy_concurrent_jobs: bool,
    runner_environment: str = "",
    runner_labels: tuple[str, ...] = (),
) -> list[str]:
    failures: list[str] = []
    if os.environ.get("GITHUB_ACTIONS") == "true" and (
        runner_environment != "self-hosted" or "isolated-benchmark" not in runner_labels
    ):
        failures.append("remote self-hosted isolated-benchmark runner")
    if not reserved_cpus:
        failures.append("reserved CPU affinity")
    if not observed_affinity_cpus:
        failures.append("observed CPU affinity")
    if reserved_cpus and tuple(sorted(reserved_cpus)) != tuple(sorted(observed_affinity_cpus)):
        failures.append("observed CPU affinity must match reserved CPU affinity")
    if max(host_load_before + host_load_after) > 1.0:
        failures.append("host load must remain low before and after benchmark")
    if not command.strip():
        failures.append("fixed command metadata is required")
    if reserved_cpus and "taskset" not in command and "chrt" not in command:
        failures.append("taskset or chrt isolation marker is required")
    if governor == "unknown" and not frequency_mhz:
        failures.append("governor or frequency metadata is required")
    if heavy_concurrent_jobs:
        failures.append("heavy concurrent jobs were reported")
    return failures


def _json_object_from_bytes(raw_payload: bytes, path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(raw_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(Mapping[str, object], payload)


def _mapping_value(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if isinstance(value, dict):
        return cast(Mapping[str, object], value)
    return {}


def _list_value(payload: Mapping[str, object], key: str) -> tuple[object, ...]:
    value = payload.get(key)
    if isinstance(value, list):
        return tuple(value)
    return ()


def _string_value(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    return value if isinstance(value, str) else ""


def _bool_value(payload: Mapping[str, object], key: str) -> bool:
    value = payload.get(key)
    return value if isinstance(value, bool) else False


def _missing_metadata_requirements(
    metadata_payload: Mapping[str, object],
    *,
    require_isolated: bool,
) -> tuple[str, ...]:
    missing: list[str] = []
    if not _string_value(metadata_payload, "command"):
        missing.append("fixed command metadata")
    if not _list_value(metadata_payload, "affinity_cpus"):
        missing.append("reserved CPU affinity metadata")
    if not _list_value(metadata_payload, "observed_affinity_cpus"):
        missing.append("observed CPU affinity metadata")
    if len(_list_value(metadata_payload, "host_load_before")) != 3:
        missing.append("host load before metadata")
    if len(_list_value(metadata_payload, "host_load_after")) != 3:
        missing.append("host load after metadata")
    if (
        require_isolated
        and not _string_value(metadata_payload, "governor")
        and not _list_value(metadata_payload, "frequency_mhz")
    ):
        missing.append("governor or frequency metadata")
    if _bool_value(metadata_payload, "heavy_concurrent_jobs"):
        missing.append("no heavy concurrent jobs")
    return tuple(missing)


def _load_average() -> tuple[float, float, float]:
    try:
        load1, load5, load15 = os.getloadavg()
        return (float(load1), float(load5), float(load15))
    except OSError:
        return (0.0, 0.0, 0.0)


def _current_affinity() -> tuple[int, ...]:
    try:
        return tuple(sorted(os.sched_getaffinity(0)))
    except AttributeError:
        return ()


def _runner_labels() -> tuple[str, ...]:
    return tuple(
        label.strip() for label in os.environ.get("RUNNER_LABELS", "").split(",") if label
    )


def _isolation_method(command: str) -> str:
    if "taskset" in command:
        return "taskset"
    if "chrt" in command:
        return "chrt"
    return "not_declared"


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return platform.processor() or "unknown"
    return platform.processor() or "unknown"


def _cpu_governor() -> str:
    path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip() or "unknown"
    except OSError:
        return "unknown"


def _cpu_frequency_mhz() -> tuple[float, ...]:
    values: list[float] = []
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("cpu mhz"):
                    values.append(float(line.split(":", 1)[1].strip()))
    except (OSError, ValueError):
        return ()
    return tuple(values)


def _dependency_versions() -> dict[str, str]:
    names = ("numpy", "jax", "torch", "tensorflow", "pennylane")
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions


__all__ = [
    "PhaseQNodeAffinityArtifactValidation",
    "PhaseQNodeAffinityBenchmarkMetadata",
    "PhaseQNodeAffinityBenchmarkResult",
    "classify_affinity_evidence",
    "run_phase_qnode_affinity_benchmark",
    "validate_phase_qnode_affinity_artifact",
]
