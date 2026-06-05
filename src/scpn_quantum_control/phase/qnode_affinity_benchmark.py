# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase QNode Affinity Benchmark
"""Small Phase-QNode benchmark harness with isolation metadata."""

from __future__ import annotations

import os
import platform
import sys
import time
from dataclasses import dataclass
from importlib import metadata
from typing import Literal, TypeAlias

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
    isolation_method: str
    host_load_before: tuple[float, float, float]
    host_load_after: tuple[float, float, float]
    cpu_model: str
    governor: str
    python_version: str
    dependency_versions: dict[str, str]
    repetitions: int
    warmups: int
    heavy_concurrent_jobs: bool

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready benchmark metadata."""
        return {
            "command": self.command,
            "affinity_cpus": list(self.affinity_cpus),
            "isolation_method": self.isolation_method,
            "host_load_before": list(self.host_load_before),
            "host_load_after": list(self.host_load_after),
            "cpu_model": self.cpu_model,
            "governor": self.governor,
            "python_version": self.python_version,
            "dependency_versions": dict(self.dependency_versions),
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


def classify_affinity_evidence(
    *,
    reserved_cpus: tuple[int, ...],
    host_load_before: tuple[float, float, float],
    host_load_after: tuple[float, float, float],
    command: str,
    heavy_concurrent_jobs: bool,
) -> EvidenceLabel:
    """Classify benchmark evidence under the core-isolation policy."""
    failures = _isolation_failures(
        reserved_cpus=reserved_cpus,
        host_load_before=host_load_before,
        host_load_after=host_load_after,
        command=command,
        heavy_concurrent_jobs=heavy_concurrent_jobs,
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
    affinity = tuple(sorted(reserved_cpus if reserved_cpus is not None else _current_affinity()))
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
    failures = _isolation_failures(
        reserved_cpus=affinity if reserved_cpus is not None else (),
        host_load_before=before,
        host_load_after=after,
        command=command_text,
        heavy_concurrent_jobs=heavy_concurrent_jobs,
    )
    label: EvidenceLabel = "isolated_affinity" if not failures else "functional_non_isolated"
    metadata_row = PhaseQNodeAffinityBenchmarkMetadata(
        command=command_text,
        affinity_cpus=affinity,
        isolation_method=_isolation_method(command_text),
        host_load_before=before,
        host_load_after=after,
        cpu_model=_cpu_model(),
        governor=_cpu_governor(),
        python_version=platform.python_version(),
        dependency_versions=_dependency_versions(),
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
            "fixed command metadata, and no heavy concurrent jobs are recorded"
        ),
    )


def _isolation_failures(
    *,
    reserved_cpus: tuple[int, ...],
    host_load_before: tuple[float, float, float],
    host_load_after: tuple[float, float, float],
    command: str,
    heavy_concurrent_jobs: bool,
) -> list[str]:
    failures: list[str] = []
    if not reserved_cpus:
        failures.append("reserved CPU affinity")
    if max(host_load_before + host_load_after) > 1.0:
        failures.append("host load must remain low before and after benchmark")
    if not command.strip():
        failures.append("fixed command metadata is required")
    if reserved_cpus and "taskset" not in command and "chrt" not in command:
        failures.append("taskset or chrt isolation marker is required")
    if heavy_concurrent_jobs:
        failures.append("heavy concurrent jobs were reported")
    return failures


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
    "PhaseQNodeAffinityBenchmarkMetadata",
    "PhaseQNodeAffinityBenchmarkResult",
    "classify_affinity_evidence",
    "run_phase_qnode_affinity_benchmark",
]
