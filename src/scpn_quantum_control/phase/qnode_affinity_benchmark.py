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
import math
import os
import platform
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Literal, NoReturn, TypeAlias, cast

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
    """Benchmark command, host, dependency, and isolation metadata.

    Parameters
    ----------
    command:
        Shell-escaped command capable of reproducing the measured run.
    affinity_cpus, observed_affinity_cpus:
        Requested and operating-system-observed CPU-index tuples.
    isolation_method:
        Declared process-isolation mechanism.
    host_load_before, host_load_after:
        One-, five-, and fifteen-minute host load samples.
    cpu_model, governor, frequency_mhz:
        Captured processor identity, frequency policy, and frequency samples.
    python_version, dependency_versions:
        Python and benchmark dependency versions.
    runner_environment, runner_labels:
        CI runner classification and labels, empty for a local process.
    repetitions, warmups:
        Recorded sample and discarded warm-up counts.
    heavy_concurrent_jobs:
        Whether the operator reported competing heavy workloads.

    """

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
        """Return a JSON-ready copy of the captured metadata."""
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
    """Raw timing rows and their isolation classification.

    Parameters
    ----------
    evidence_label:
        Isolation-policy classification derived from measured metadata.
    production_benchmark:
        Whether the run is eligible for production benchmark claims.
    metadata:
        Reproduction and host-isolation metadata.
    raw_timing_rows:
        Per-repetition timing, value, and gradient-norm records.
    isolation_failures:
        Ordered policy failures that blocked isolated classification.
    claim_boundary:
        Human-readable limit on permitted benchmark claims.

    """

    evidence_label: EvidenceLabel
    production_benchmark: bool
    metadata: PhaseQNodeAffinityBenchmarkMetadata
    raw_timing_rows: tuple[dict[str, float], ...]
    isolation_failures: tuple[str, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return the complete benchmark evidence as JSON-ready values."""
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
    """Fail-closed attachment verdict for one benchmark JSON artefact.

    Parameters
    ----------
    artifact_path, artifact_sha256, benchmark_artifact_id:
        Source path, full content digest, and deterministic attachment ID.
    evidence_label, production_benchmark:
        Untrusted raw classification fields retained for audit output.
    promotion_ready:
        Whether strict isolated-promotion requirements pass. Observation-only
        validation never changes this meaning.
    raw_timing_row_count:
        Number of timing rows found in the raw JSON list.
    missing_requirements:
        Requested-mode schema, consistency, and isolation-policy failures.
    claim_boundary:
        Raw claim boundary or a conservative fallback when it is invalid.

    """

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
    """Classify benchmark evidence under the core-isolation policy.

    Parameters
    ----------
    reserved_cpus, observed_affinity_cpus:
        Requested and operating-system-observed CPU indexes.
    host_load_before, host_load_after:
        Host load samples captured around the benchmark loop.
    command:
        Reproduction command carrying an admitted isolation marker.
    governor, frequency_mhz:
        CPU frequency-policy evidence.
    heavy_concurrent_jobs:
        Whether competing workloads were present.
    runner_environment, runner_labels:
        Optional GitHub Actions runner evidence.

    Returns
    -------
    EvidenceLabel
        ``isolated_affinity`` only when every policy check passes; otherwise
        ``functional_non_isolated``.

    """
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
    """Run a bounded Phase-QNode timing loop with reproduction metadata.

    Parameters
    ----------
    repetitions, warmups:
        Recorded repetitions and discarded warm-up iterations.
    reserved_cpus:
        CPU indexes reserved by the outer runner, or ``None`` to observe only.
    host_load_before, host_load_after:
        Optional externally captured load samples for deterministic replay.
    command:
        Reproduction command, or ``None`` for the module-level default.
    heavy_concurrent_jobs:
        Operator declaration of competing heavy workloads.

    Returns
    -------
    PhaseQNodeAffinityBenchmarkResult
        Raw timing rows, measured host metadata, and isolation verdict.

    Raises
    ------
    ValueError
        If counts or reserved CPU indexes violate the input contract.

    """
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

    The validator does not trust or upgrade recorded labels. It parses all
    numerical and host metadata, recomputes the isolation policy, and compares
    the result with recorded failures. ``promotion_ready`` always means strict
    isolated promotion; ``require_isolated=False`` only makes
    ``missing_requirements`` report observation-mode schema consistency.

    Parameters
    ----------
    artifact_path:
        UTF-8 JSON artefact to hash and validate.
    require_isolated:
        Whether ``missing_requirements`` includes isolated-promotion failures.

    Returns
    -------
    PhaseQNodeAffinityArtifactValidation
        Content-addressed, fail-closed validation evidence.

    Raises
    ------
    OSError
        If the artefact cannot be read.
    ValueError
        If the artefact is not UTF-8 JSON with an object at its root.

    """
    path = Path(artifact_path)
    artifact_bytes = path.read_bytes()
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    benchmark_artifact_id = f"phase-qnode-affinity:{artifact_sha256[:16]}"
    payload = _json_object_from_bytes(artifact_bytes, path)
    observational_requirements: list[str] = []

    evidence_label_value = payload.get("evidence_label")
    evidence_label = evidence_label_value if isinstance(evidence_label_value, str) else ""
    if evidence_label not in ("isolated_affinity", "functional_non_isolated"):
        observational_requirements.append("recognized evidence label")

    production_value = payload.get("production_benchmark")
    production_benchmark = production_value if isinstance(production_value, bool) else False
    if not isinstance(production_value, bool):
        observational_requirements.append("boolean production benchmark flag")

    isolation_failures = _string_sequence(payload.get("isolation_failures"))
    if isolation_failures is None:
        isolation_failures = ()
        observational_requirements.append("string isolation failures")

    raw_timing_row_count, timing_requirements = _timing_row_requirements(
        payload.get("raw_timing_rows")
    )
    observational_requirements.extend(timing_requirements)

    metadata_payload = payload.get("metadata")
    metadata, metadata_requirements = _parse_artifact_metadata(metadata_payload)
    observational_requirements.extend(metadata_requirements)
    if metadata is not None and raw_timing_row_count != metadata.repetitions:
        observational_requirements.append("raw timing row count matching repetitions")

    claim_boundary_value = payload.get("claim_boundary")
    claim_boundary = claim_boundary_value.strip() if isinstance(claim_boundary_value, str) else ""
    if not claim_boundary:
        observational_requirements.append("non-empty claim boundary")
        claim_boundary = (
            "Phase-QNode affinity benchmark artefacts are promotional only when "
            "validated as isolated_affinity with raw timing rows and complete host "
            "isolation metadata."
        )

    recomputed_failures: tuple[str, ...] = ()
    if metadata is not None:
        recomputed_failures = tuple(
            _isolation_failures(
                reserved_cpus=metadata.affinity_cpus,
                observed_affinity_cpus=metadata.observed_affinity_cpus,
                host_load_before=metadata.host_load_before,
                host_load_after=metadata.host_load_after,
                command=metadata.command,
                governor=metadata.governor,
                frequency_mhz=metadata.frequency_mhz,
                heavy_concurrent_jobs=metadata.heavy_concurrent_jobs,
                runner_environment=metadata.runner_environment,
                runner_labels=metadata.runner_labels,
                github_actions=bool(metadata.runner_environment or metadata.runner_labels),
            )
        )
        expected_label = (
            "isolated_affinity" if not recomputed_failures else "functional_non_isolated"
        )
        if evidence_label in ("isolated_affinity", "functional_non_isolated") and (
            evidence_label != expected_label
        ):
            observational_requirements.append(
                "evidence label consistent with recomputed isolation policy"
            )
        if isolation_failures != recomputed_failures:
            observational_requirements.append(
                "recorded isolation failures match recomputed policy"
            )

    if (
        isinstance(production_value, bool)
        and evidence_label in ("isolated_affinity", "functional_non_isolated")
        and production_benchmark != (evidence_label == "isolated_affinity")
    ):
        observational_requirements.append(
            "production benchmark flag consistent with evidence label"
        )

    observational_requirements = list(_deduplicated_requirements(observational_requirements))
    promotion_requirements = list(observational_requirements)
    if evidence_label != "isolated_affinity":
        promotion_requirements.append("isolated_affinity evidence label")
    if not production_benchmark:
        promotion_requirements.append("production benchmark flag")
    if isolation_failures:
        promotion_requirements.append("empty isolation failures")
    promotion_requirements.extend(
        f"isolation policy: {failure}" for failure in recomputed_failures
    )
    promotion_requirements = list(_deduplicated_requirements(promotion_requirements))

    missing_requirements = (
        promotion_requirements if require_isolated else observational_requirements
    )
    promotion_ready = not promotion_requirements
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
    github_actions: bool | None = None,
) -> list[str]:
    failures: list[str] = []
    running_on_github = (
        os.environ.get("GITHUB_ACTIONS") == "true" if github_actions is None else github_actions
    )
    if running_on_github and (
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
        payload = json.loads(
            raw_payload.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{path} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(Mapping[str, object], payload)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build one JSON object while rejecting duplicate field names."""
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON object key: {key}")
        payload[key] = value
    return payload


def _reject_json_constant(value: str) -> NoReturn:
    """Reject non-standard JSON constants such as NaN and Infinity."""
    raise ValueError(f"non-finite JSON constant: {value}")


@dataclass(frozen=True)
class _ParsedAffinityMetadata:
    """Typed metadata recovered from an untrusted benchmark JSON object."""

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


def _parse_artifact_metadata(
    value: object,
) -> tuple[_ParsedAffinityMetadata | None, tuple[str, ...]]:
    """Return typed benchmark metadata plus all violated schema requirements."""
    if not isinstance(value, dict):
        return None, ("metadata object",)
    payload = cast(Mapping[str, object], value)
    requirements: list[str] = []

    command = _non_empty_string(payload.get("command"))
    if command is None:
        requirements.append("fixed command metadata")

    affinity_cpus = _cpu_sequence(payload.get("affinity_cpus"))
    if affinity_cpus is None:
        requirements.append("reserved CPU affinity metadata")
    observed_affinity_cpus = _cpu_sequence(payload.get("observed_affinity_cpus"))
    if observed_affinity_cpus is None:
        requirements.append("observed CPU affinity metadata")
    if (
        affinity_cpus is not None
        and observed_affinity_cpus is not None
        and affinity_cpus != observed_affinity_cpus
    ):
        requirements.append("matching observed CPU affinity metadata")

    isolation_method = _non_empty_string(payload.get("isolation_method"))
    if isolation_method not in {"taskset", "chrt", "not_declared"}:
        requirements.append("recognized isolation method metadata")
    if (
        command is not None
        and isolation_method in {"taskset", "chrt", "not_declared"}
        and isolation_method != _isolation_method(command)
    ):
        requirements.append("isolation method matching command metadata")

    host_load_before = _fixed_float_sequence(payload.get("host_load_before"), length=3)
    if host_load_before is None or any(load < 0.0 for load in host_load_before):
        requirements.append("host load before metadata")
    host_load_after = _fixed_float_sequence(payload.get("host_load_after"), length=3)
    if host_load_after is None or any(load < 0.0 for load in host_load_after):
        requirements.append("host load after metadata")

    cpu_model = _non_empty_string(payload.get("cpu_model"))
    if cpu_model is None:
        requirements.append("CPU model metadata")
    governor = _non_empty_string(payload.get("governor"))
    if governor is None:
        requirements.append("governor metadata")
    frequency_mhz = _float_sequence(payload.get("frequency_mhz"), allow_empty=True)
    if frequency_mhz is None or any(frequency <= 0.0 for frequency in frequency_mhz):
        requirements.append("positive CPU frequency metadata")
    if (governor is None or governor == "unknown") and not frequency_mhz:
        requirements.append("governor or frequency metadata")

    python_version = _non_empty_string(payload.get("python_version"))
    if python_version is None:
        requirements.append("Python version metadata")
    dependency_versions = _dependency_version_mapping(payload.get("dependency_versions"))
    if dependency_versions is None:
        requirements.append("dependency version metadata")

    runner_environment_value = payload.get("runner_environment")
    runner_environment = (
        runner_environment_value.strip() if isinstance(runner_environment_value, str) else None
    )
    if runner_environment is None:
        requirements.append("runner environment metadata")
    runner_labels = _string_sequence(payload.get("runner_labels"))
    if runner_labels is None or len(set(runner_labels)) != len(runner_labels):
        requirements.append("runner label metadata")

    repetitions = _integer_at_least(payload.get("repetitions"), minimum=1)
    if repetitions is None:
        requirements.append("positive repetition count")
    warmups = _integer_at_least(payload.get("warmups"), minimum=0)
    if warmups is None:
        requirements.append("non-negative warmup count")
    heavy_jobs_value = payload.get("heavy_concurrent_jobs")
    heavy_concurrent_jobs = heavy_jobs_value if isinstance(heavy_jobs_value, bool) else None
    if heavy_concurrent_jobs is None:
        requirements.append("boolean heavy concurrent jobs metadata")

    if requirements:
        return None, _deduplicated_requirements(requirements)
    return (
        _ParsedAffinityMetadata(
            command=cast(str, command),
            affinity_cpus=cast(tuple[int, ...], affinity_cpus),
            observed_affinity_cpus=cast(tuple[int, ...], observed_affinity_cpus),
            isolation_method=cast(str, isolation_method),
            host_load_before=cast(tuple[float, float, float], host_load_before),
            host_load_after=cast(tuple[float, float, float], host_load_after),
            cpu_model=cast(str, cpu_model),
            governor=cast(str, governor),
            frequency_mhz=cast(tuple[float, ...], frequency_mhz),
            python_version=cast(str, python_version),
            dependency_versions=cast(dict[str, str], dependency_versions),
            runner_environment=cast(str, runner_environment),
            runner_labels=cast(tuple[str, ...], runner_labels),
            repetitions=cast(int, repetitions),
            warmups=cast(int, warmups),
            heavy_concurrent_jobs=cast(bool, heavy_concurrent_jobs),
        ),
        (),
    )


def _timing_row_requirements(value: object) -> tuple[int, tuple[str, ...]]:
    """Return raw row count plus schema and numerical-integrity failures."""
    if not isinstance(value, list):
        return 0, ("raw timing row list", "raw timing rows")
    requirements: list[str] = []
    if not value:
        requirements.append("raw timing rows")
    for index, row_value in enumerate(value):
        if not isinstance(row_value, dict):
            requirements.append("well-formed raw timing rows")
            continue
        row = cast(Mapping[str, object], row_value)
        iteration = _finite_float(row.get("iteration"))
        seconds = _finite_float(row.get("seconds"))
        measured_value = _finite_float(row.get("value"))
        gradient_norm = _finite_float(row.get("gradient_norm"))
        if (
            iteration != float(index)
            or seconds is None
            or seconds < 0.0
            or measured_value is None
            or gradient_norm is None
            or gradient_norm < 0.0
        ):
            requirements.append("well-formed raw timing rows")
    return len(value), _deduplicated_requirements(requirements)


def _non_empty_string(value: object) -> str | None:
    """Return a stripped non-empty string, otherwise ``None``."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _string_sequence(value: object) -> tuple[str, ...] | None:
    """Return a tuple of non-empty strings from a JSON list."""
    if not isinstance(value, list):
        return None
    strings: list[str] = []
    for item in value:
        parsed = _non_empty_string(item)
        if parsed is None:
            return None
        strings.append(parsed)
    return tuple(strings)


def _cpu_sequence(value: object) -> tuple[int, ...] | None:
    """Return a sorted, unique, non-negative CPU-index tuple."""
    if not isinstance(value, list) or not value:
        return None
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in value):
        return None
    cpus = tuple(cast(list[int], value))
    if cpus != tuple(sorted(set(cpus))):
        return None
    return cpus


def _finite_float(value: object) -> float | None:
    """Return a finite JSON number while rejecting booleans."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except OverflowError:
        return None
    return result if math.isfinite(result) else None


def _float_sequence(value: object, *, allow_empty: bool) -> tuple[float, ...] | None:
    """Return finite floats from a JSON list under the requested empty policy."""
    if not isinstance(value, list) or (not allow_empty and not value):
        return None
    parsed = tuple(_finite_float(item) for item in value)
    if any(item is None for item in parsed):
        return None
    return tuple(cast(tuple[float, ...], parsed))


def _fixed_float_sequence(
    value: object,
    *,
    length: int,
) -> tuple[float, float, float] | None:
    """Return one three-value finite sequence used by load averages."""
    parsed = _float_sequence(value, allow_empty=False)
    if parsed is None or len(parsed) != length or length != 3:
        return None
    return (parsed[0], parsed[1], parsed[2])


def _integer_at_least(value: object, *, minimum: int) -> int | None:
    """Return a JSON integer at or above ``minimum`` while rejecting booleans."""
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        return None
    return value


def _dependency_version_mapping(value: object) -> dict[str, str] | None:
    """Return required non-empty dependency-version strings."""
    if not isinstance(value, dict):
        return None
    versions: dict[str, str] = {}
    for key, item in cast(dict[str, object], value).items():
        parsed = _non_empty_string(item)
        if not key.strip() or parsed is None:
            return None
        versions[key.strip()] = parsed
    if not {"python", "numpy"}.issubset(versions):
        return None
    return versions


def _deduplicated_requirements(requirements: list[str]) -> tuple[str, ...]:
    """Return first-seen validation requirements without duplicates."""
    return tuple(dict.fromkeys(requirements))


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
