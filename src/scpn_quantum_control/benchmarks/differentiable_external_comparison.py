# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — external differentiable comparison rows.
"""External framework comparison rows for bounded Phase-QNode claims."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import tracemalloc
from dataclasses import dataclass
from importlib import import_module, metadata
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ..phase.jax_bridge import is_phase_jax_available
from ..phase.pennylane_bridge import is_phase_pennylane_available
from ..phase.qnode_circuit import (
    PauliTerm,
    PhaseQNodeCircuit,
    execute_phase_qnode_circuit,
    parameter_shift_phase_qnode_gradient,
)
from ..phase.tensorflow_bridge import is_phase_tensorflow_available
from ..phase.torch_bridge import is_phase_torch_available

ComparisonStatus = Literal["success", "hard_gap"]

REQUIRED_EXTERNAL_COMPARISON_ROW_FIELDS = frozenset(
    {
        "case_id",
        "backend",
        "status",
        "failure_class",
        "value_error",
        "gradient_error",
        "runtime_seconds",
        "memory_peak_bytes",
        "batching_support",
        "transform_support",
        "dtype",
        "device",
        "source_of_truth",
        "setup_instructions",
        "claim_boundary",
        "dependency_versions",
        "toolchain",
    }
)


@dataclass(frozen=True)
class ExternalComparisonRow:
    """One external framework/compiler comparison row."""

    case_id: str
    backend: str
    status: ComparisonStatus
    failure_class: str | None
    value_error: float | None
    gradient_error: float | None
    runtime_seconds: float | None
    memory_peak_bytes: int | None
    batching_support: str
    transform_support: str
    dtype: str
    device: str
    source_of_truth: str
    setup_instructions: str | None
    claim_boundary: str
    dependency_versions: dict[str, str] | None = None
    toolchain: dict[str, str] | None = None

    def __post_init__(self) -> None:
        """Validate external comparison row evidence invariants."""
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if not self.backend:
            raise ValueError("backend must be non-empty")
        if self.status not in {"success", "hard_gap"}:
            raise ValueError("status must be success or hard_gap")
        if self.source_of_truth != "scpn_reference":
            raise ValueError("source_of_truth must be scpn_reference")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")
        if self.status == "success":
            if (
                self.failure_class is not None
                or self.value_error is None
                or self.gradient_error is None
                or self.runtime_seconds is None
                or self.memory_peak_bytes is None
            ):
                raise ValueError("success rows require numeric evidence and no failure class")
            if self.value_error < 0 or self.gradient_error < 0:
                raise ValueError("success row errors must be non-negative")
        if self.status == "hard_gap" and (not self.failure_class or not self.setup_instructions):
            raise ValueError("hard_gap rows require failure_class and setup_instructions")
        if self.toolchain is not None and (
            any(not isinstance(key, str) or not key for key in self.toolchain)
            or any(not isinstance(value, str) or not value for value in self.toolchain.values())
        ):
            raise ValueError("toolchain metadata must map non-empty strings to non-empty strings")
        if self.dependency_versions is not None and (
            any(not isinstance(key, str) or not key for key in self.dependency_versions)
            or any(
                not isinstance(value, str) or not value
                for value in self.dependency_versions.values()
            )
        ):
            raise ValueError(
                "dependency version metadata must map non-empty strings to non-empty strings"
            )

    @property
    def artifact_fields_ready(self) -> bool:
        """Return whether this row is serializable as an evidence artefact."""
        payload = self.to_dict()
        return bool(
            self.case_id
            and self.backend
            and self.status
            and self.claim_boundary
            and REQUIRED_EXTERNAL_COMPARISON_ROW_FIELDS.issubset(payload)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready row."""
        return {
            "case_id": self.case_id,
            "backend": self.backend,
            "status": self.status,
            "failure_class": self.failure_class,
            "value_error": self.value_error,
            "gradient_error": self.gradient_error,
            "runtime_seconds": self.runtime_seconds,
            "memory_peak_bytes": self.memory_peak_bytes,
            "batching_support": self.batching_support,
            "transform_support": self.transform_support,
            "dtype": self.dtype,
            "device": self.device,
            "source_of_truth": self.source_of_truth,
            "setup_instructions": self.setup_instructions,
            "claim_boundary": self.claim_boundary,
            "dependency_versions": (
                dict(self.dependency_versions) if self.dependency_versions is not None else None
            ),
            "toolchain": dict(self.toolchain) if self.toolchain is not None else None,
        }


@dataclass(frozen=True)
class ExternalComparisonArtifact:
    """Written external comparison artefact paths and summary metadata."""

    artifact_id: str
    path: Path
    row_count: int
    success_count: int
    hard_gap_count: int
    classification: str
    claim_boundary: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready artefact summary."""
        return {
            "artifact_id": self.artifact_id,
            "path": str(self.path),
            "row_count": self.row_count,
            "success_count": self.success_count,
            "hard_gap_count": self.hard_gap_count,
            "classification": self.classification,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class IdenticalCircuitGradientComparisonRow:
    """One same-circuit gradient comparison row against an external framework."""

    case_id: str
    backend: str
    status: ComparisonStatus
    failure_class: str | None
    circuit_fingerprint: str
    operations: tuple[tuple[object, ...], ...]
    observable: str
    parameter_values: tuple[float, ...]
    execution_mode: str
    shots: int | None
    scpn_value: float | None
    backend_value: float | None
    value_error: float | None
    scpn_gradient: tuple[float, ...] | None
    backend_gradient: tuple[float, ...] | None
    gradient_error: float | None
    evaluations: int | None
    dependency_versions: dict[str, str] | None
    claim_boundary: str
    performance_claim_eligible: bool = False

    def __post_init__(self) -> None:
        """Validate same-circuit comparison row evidence invariants."""
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if self.backend not in {"qiskit", "pennylane"}:
            raise ValueError("backend must be qiskit or pennylane")
        if self.status not in {"success", "hard_gap"}:
            raise ValueError("status must be success or hard_gap")
        if not self.circuit_fingerprint:
            raise ValueError("circuit_fingerprint must be non-empty")
        if not self.operations:
            raise ValueError("operations must be non-empty")
        if not self.observable:
            raise ValueError("observable must be non-empty")
        if self.execution_mode != "exact_state":
            raise ValueError("execution_mode must be exact_state")
        if self.shots is not None:
            raise ValueError(
                "identical-circuit comparison uses exact-state mode; shots must be None"
            )
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")
        if self.status == "success":
            if (
                self.failure_class is not None
                or self.scpn_value is None
                or self.backend_value is None
                or self.value_error is None
                or self.scpn_gradient is None
                or self.backend_gradient is None
                or self.gradient_error is None
                or self.evaluations is None
            ):
                raise ValueError("success rows require numeric value and gradient evidence")
            if self.value_error < 0.0 or self.gradient_error < 0.0:
                raise ValueError("success row errors must be non-negative")
        if self.status == "hard_gap" and self.failure_class is None:
            raise ValueError("hard_gap rows require a failure_class")

    @property
    def artifact_fields_ready(self) -> bool:
        """Return whether the row carries the required same-circuit fields."""
        return bool(
            self.case_id
            and self.backend
            and self.circuit_fingerprint
            and self.operations
            and self.observable
            and self.execution_mode
            and self.claim_boundary
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready row."""
        return {
            "case_id": self.case_id,
            "backend": self.backend,
            "status": self.status,
            "failure_class": self.failure_class,
            "circuit_fingerprint": self.circuit_fingerprint,
            "operations": _jsonify_operations(self.operations),
            "observable": self.observable,
            "parameter_values": list(self.parameter_values),
            "execution_mode": self.execution_mode,
            "shots": self.shots,
            "scpn_value": self.scpn_value,
            "backend_value": self.backend_value,
            "value_error": self.value_error,
            "scpn_gradient": list(self.scpn_gradient) if self.scpn_gradient is not None else None,
            "backend_gradient": (
                list(self.backend_gradient) if self.backend_gradient is not None else None
            ),
            "gradient_error": self.gradient_error,
            "evaluations": self.evaluations,
            "dependency_versions": (
                dict(self.dependency_versions) if self.dependency_versions is not None else None
            ),
            "claim_boundary": self.claim_boundary,
            "performance_claim_eligible": self.performance_claim_eligible,
        }


@dataclass(frozen=True)
class IdenticalCircuitGradientComparisonArtifact:
    """Written same-circuit comparison artefact summary."""

    artifact_id: str
    path: Path
    row_count: int
    success_count: int
    hard_gap_count: int
    identical_circuit_ready: bool
    promotion_ready: bool
    classification: str
    claim_boundary: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready artefact summary."""
        return {
            "artifact_id": self.artifact_id,
            "path": str(self.path),
            "row_count": self.row_count,
            "success_count": self.success_count,
            "hard_gap_count": self.hard_gap_count,
            "identical_circuit_ready": self.identical_circuit_ready,
            "promotion_ready": self.promotion_ready,
            "classification": self.classification,
            "claim_boundary": self.claim_boundary,
        }


def run_differentiable_external_comparison_suite() -> tuple[ExternalComparisonRow, ...]:
    """Run or classify optional external comparison rows.

    The SCPN analytic parameter-shift reference remains the source of truth.
    Missing optional tooling is recorded as hard-gap evidence instead of being
    silently omitted.
    """
    rows: list[ExternalComparisonRow] = []
    for backend, available, runner, batching, transform, setup in (
        (
            "jax",
            is_phase_jax_available(),
            _run_jax_reference,
            "vmap",
            "value_and_grad",
            "Install the CPU overlay wheel jax[cpu].",
        ),
        (
            "pytorch",
            is_phase_torch_available(),
            _run_pytorch_reference,
            "torch.func.vmap",
            "torch.func.grad/jacrev",
            "Install the CPU overlay wheel torch.",
        ),
        (
            "tensorflow",
            is_phase_tensorflow_available(),
            _run_tensorflow_reference,
            "vectorized_map",
            "GradientTape",
            "Install the CPU overlay wheel tensorflow-cpu.",
        ),
        (
            "pennylane",
            is_phase_pennylane_available(),
            _run_pennylane_reference,
            "not_native",
            "QNode",
            "Install the CPU overlay wheel pennylane.",
        ),
    ):
        rows.append(
            _framework_row(backend, runner, batching, transform, setup)
            if available
            else _dependency_gap_row(backend, batching, transform, setup)
        )
    rows.append(
        _enzyme_row()
        if _enzyme_runner_configured()
        else _dependency_gap_row(
            "enzyme",
            "not_evaluated",
            "LLVM Enzyme",
            "Install LLVM/Enzyme tooling and configure the Enzyme runner.",
        )
    )
    rows.append(
        _catalyst_row()
        if _catalyst_runner_configured()
        else _dependency_gap_row(
            "catalyst",
            "not_evaluated",
            "Catalyst qjit/MLIR/QIR",
            "Install PennyLane Catalyst and configure SCPN_CATALYST_RUNNER.",
        )
    )
    rows.extend(external_comparison_failure_mode_rows())
    return tuple(rows)


def write_differentiable_external_comparison(
    output_path: str | os.PathLike[str],
    rows: tuple[ExternalComparisonRow, ...] | None = None,
    *,
    artifact_id: str = "differentiable-external-comparison-local",
) -> ExternalComparisonArtifact:
    """Write external comparison rows as a bounded JSON evidence artefact."""
    destination = Path(output_path)
    if destination.suffix.lower() != ".json":
        raise ValueError("output_path must end with .json")
    if not artifact_id.strip():
        raise ValueError("artifact_id must be non-empty")
    evidence_rows = rows if rows is not None else run_differentiable_external_comparison_suite()
    if not evidence_rows:
        raise ValueError("at least one external comparison row is required")
    row_payloads = [row.to_dict() for row in evidence_rows]
    for row, row_payload in zip(evidence_rows, row_payloads, strict=True):
        missing_fields = REQUIRED_EXTERNAL_COMPARISON_ROW_FIELDS.difference(row_payload)
        if not getattr(row, "artifact_fields_ready", False) or missing_fields:
            missing = ", ".join(sorted(missing_fields)) or "row readiness predicate"
            raise ValueError(
                f"external comparison row missing required artefact fields: {missing}"
            )
    success_count = sum(1 for row in evidence_rows if row.status == "success")
    hard_gap_count = sum(1 for row in evidence_rows if row.status == "hard_gap")
    failure_classes = sorted(
        {
            row.failure_class
            for row in evidence_rows
            if row.status == "hard_gap" and row.failure_class is not None
        }
    )
    payload = {
        "schema": "scpn_qc_differentiable_external_comparison_v1",
        "artifact_id": artifact_id.strip(),
        "generated_at_unix": int(time.time()),
        "classification": "functional_non_isolated",
        "production_eligible": False,
        "promotion_ready": False,
        "source_of_truth": "scpn_reference",
        "row_schema": {
            "required_fields": sorted(REQUIRED_EXTERNAL_COMPARISON_ROW_FIELDS),
            "success_numeric_fields": [
                "value_error",
                "gradient_error",
                "runtime_seconds",
                "memory_peak_bytes",
            ],
            "hard_gap_required_fields": ["failure_class", "setup_instructions"],
        },
        "claim_boundary": (
            "External comparison artefact for bounded CPU framework correctness rows; "
            "not isolated benchmark evidence, not provider execution, and not a "
            "promotion artefact until the isolated benchmark gate supplies artefact IDs."
        ),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
        },
        "summary": {
            "row_count": len(evidence_rows),
            "success_count": success_count,
            "hard_gap_count": hard_gap_count,
            "failure_classes": failure_classes,
        },
        "rows": row_payloads,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ExternalComparisonArtifact(
        artifact_id=artifact_id.strip(),
        path=destination,
        row_count=len(evidence_rows),
        success_count=success_count,
        hard_gap_count=hard_gap_count,
        classification="functional_non_isolated",
        claim_boundary=str(payload["claim_boundary"]),
    )


def run_identical_circuit_gradient_comparison_suite() -> tuple[
    IdenticalCircuitGradientComparisonRow, ...
]:
    """Run exact-state same-circuit gradient comparisons for Qiskit and PennyLane."""
    circuit, values, operations, observable_label, fingerprint = _identical_circuit_problem()
    scpn_value = execute_phase_qnode_circuit(circuit, values).value
    scpn_gradient_result = parameter_shift_phase_qnode_gradient(circuit, values)
    scpn_gradient = tuple(float(item) for item in scpn_gradient_result.gradient)
    return (
        _qiskit_identical_circuit_row(
            values=values,
            operations=operations,
            observable_label=observable_label,
            fingerprint=fingerprint,
            scpn_value=float(scpn_value),
            scpn_gradient=scpn_gradient,
        ),
        _pennylane_identical_circuit_row(
            circuit=circuit,
            values=values,
            operations=operations,
            observable_label=observable_label,
            fingerprint=fingerprint,
            scpn_value=float(scpn_value),
            scpn_gradient=scpn_gradient,
        ),
    )


def write_identical_circuit_gradient_comparison(
    output_path: str | os.PathLike[str],
    rows: tuple[IdenticalCircuitGradientComparisonRow, ...] | None = None,
    *,
    artifact_id: str = "identical-circuit-gradient-comparison-local",
) -> IdenticalCircuitGradientComparisonArtifact:
    """Write exact-state same-circuit comparison rows as JSON evidence."""
    destination = Path(output_path)
    if destination.suffix.lower() != ".json":
        raise ValueError("output_path must end with .json")
    if not artifact_id.strip():
        raise ValueError("artifact_id must be non-empty")
    evidence_rows = rows if rows is not None else run_identical_circuit_gradient_comparison_suite()
    if not evidence_rows:
        raise ValueError("at least one identical-circuit comparison row is required")
    fingerprints = {row.circuit_fingerprint for row in evidence_rows}
    backends = {row.backend for row in evidence_rows}
    success_count = sum(row.status == "success" for row in evidence_rows)
    hard_gap_count = sum(row.status == "hard_gap" for row in evidence_rows)
    identical_circuit_ready = (
        fingerprints == {next(iter(fingerprints))}
        and {"qiskit", "pennylane"}.issubset(backends)
        and all(row.status == "success" for row in evidence_rows)
        and all(row.execution_mode == "exact_state" and row.shots is None for row in evidence_rows)
    )
    first_row = evidence_rows[0]
    payload = {
        "schema": "scpn_qc_identical_circuit_gradient_comparison_v1",
        "artifact_id": artifact_id.strip(),
        "generated_at_unix": int(time.time()),
        "classification": "functional_non_isolated",
        "production_eligible": False,
        "identical_circuit_ready": identical_circuit_ready,
        "promotion_ready": False,
        "source_of_truth": "scpn_phase_qnode_reference",
        "claim_boundary": (
            "Exact-state identical-circuit gradient comparison for local Qiskit and "
            "PennyLane correctness. It is not hardware execution, finite-shot evidence, "
            "or isolated benchmark promotion."
        ),
        "same_circuit_contract": {
            "case_id": first_row.case_id,
            "circuit_fingerprint": first_row.circuit_fingerprint,
            "operations": _jsonify_operations(first_row.operations),
            "observable": first_row.observable,
            "parameter_values": list(first_row.parameter_values),
            "execution_mode": "exact_state",
            "shots": None,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
        },
        "summary": {
            "row_count": len(evidence_rows),
            "success_count": success_count,
            "hard_gap_count": hard_gap_count,
            "failure_classes": sorted(
                {
                    row.failure_class
                    for row in evidence_rows
                    if row.status == "hard_gap" and row.failure_class is not None
                }
            ),
        },
        "rows": [row.to_dict() for row in evidence_rows],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return IdenticalCircuitGradientComparisonArtifact(
        artifact_id=artifact_id.strip(),
        path=destination,
        row_count=len(evidence_rows),
        success_count=success_count,
        hard_gap_count=hard_gap_count,
        identical_circuit_ready=identical_circuit_ready,
        promotion_ready=False,
        classification="functional_non_isolated",
        claim_boundary=str(payload["claim_boundary"]),
    )


def external_comparison_failure_mode_rows() -> tuple[ExternalComparisonRow, ...]:
    """Return explicit unsupported-route rows for promotion-evidence artefacts."""
    return (
        _unsupported_gap_row(
            backend="jax",
            failure_class="unsupported_batching",
            batching_support="single-host vmap only; multi-host pmap is not promotion evidence",
            transform_support="value_and_grad",
            dtype="float64",
            device="cpu",
            setup=(
                "Bounded JAX external comparison evidence covers CPU value_and_grad/vmap only; "
                "multi-host pmap or sharded hardware batches require a separate isolated artefact."
            ),
        ),
        _unsupported_gap_row(
            backend="pytorch",
            failure_class="unsupported_transform",
            batching_support="torch.func.vmap",
            transform_support="nested torch.compile grad-of-grad not evaluated",
            dtype="float64",
            device="cpu",
            setup=(
                "Bounded PyTorch external comparison evidence covers torch.func.grad, vmap, "
                "jacrev, compile, and module wrappers separately; nested compile grad-of-grad "
                "promotion requires a dedicated correctness and isolation artefact."
            ),
        ),
        _unsupported_gap_row(
            backend="tensorflow",
            failure_class="unsupported_dtype",
            batching_support="vectorized_map",
            transform_support="GradientTape",
            dtype="complex128",
            device="cpu",
            setup=(
                "External comparison promotion is real-valued float64 only. Complex or "
                "Wirtinger TensorFlow routes must use the explicit real/imaginary contract."
            ),
        ),
        _unsupported_gap_row(
            backend="pennylane",
            failure_class="unsupported_device",
            batching_support="not_native",
            transform_support="QNode",
            dtype="float64",
            device="hardware_qpu",
            setup=(
                "External comparison rows do not submit live provider or QPU jobs. Hardware "
                "PennyLane evidence requires a provider policy ticket, artefact ID, shot budget, "
                "and isolated benchmark classification."
            ),
        ),
    )


def _framework_row(
    backend: str,
    runner: Any,
    batching_support: str,
    transform_support: str,
    setup: str,
) -> ExternalComparisonRow:
    values = np.array([0.2, -0.4], dtype=np.float64)
    reference_value = _bounded_phase_objective(values)
    reference_gradient = _bounded_phase_gradient(values)
    tracemalloc.start()
    start = time.perf_counter()
    try:
        external_value, external_gradient = runner(values)
    except ImportError:
        tracemalloc.stop()
        return _dependency_gap_row(backend, batching_support, transform_support, setup)
    except Exception as exc:  # pragma: no cover - defensive optional boundary
        tracemalloc.stop()
        return _runtime_gap_row(backend, batching_support, transform_support, str(exc))
    runtime = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return ExternalComparisonRow(
        case_id="bounded_phase_objective",
        backend=backend,
        status="success",
        failure_class=None,
        value_error=abs(reference_value - external_value),
        gradient_error=float(np.max(np.abs(reference_gradient - external_gradient))),
        runtime_seconds=runtime,
        memory_peak_bytes=max(int(peak), int(values.nbytes + reference_gradient.nbytes)),
        batching_support=batching_support,
        transform_support=transform_support,
        dtype="float64",
        device="cpu",
        source_of_truth="scpn_reference",
        setup_instructions=None,
        claim_boundary=(
            "Bounded CPU external comparison against the SCPN reference; no provider, "
            "QPU, GPU, or production performance claim."
        ),
        dependency_versions=_backend_dependency_versions(backend),
    )


def _enzyme_row() -> ExternalComparisonRow:
    if not _enzyme_runner_configured():
        return _dependency_gap_row(
            "enzyme",
            "not_evaluated",
            "LLVM Enzyme",
            "Configure SCPN_ENZYME_RUNNER to an executable runner that emits value and gradient JSON.",
        )
    values = np.array([0.2, -0.4], dtype=np.float64)
    reference_value = _bounded_phase_objective(values)
    reference_gradient = _bounded_phase_gradient(values)
    tracemalloc.start()
    start = time.perf_counter()
    try:
        external_value, external_gradient, toolchain = _run_enzyme_reference(values)
    except TimeoutError as exc:
        tracemalloc.stop()
        return _runtime_gap_row("enzyme", "not_supported", "LLVM Enzyme runner", str(exc))
    except (RuntimeError, ValueError) as exc:
        tracemalloc.stop()
        return _runtime_gap_row("enzyme", "not_supported", "LLVM Enzyme runner", str(exc))
    runtime = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    value_error = abs(reference_value - external_value)
    gradient_error = float(np.max(np.abs(reference_gradient - external_gradient)))
    if value_error > 1.0e-8 or gradient_error > 1.0e-8:
        return ExternalComparisonRow(
            case_id="bounded_phase_objective",
            backend="enzyme",
            status="hard_gap",
            failure_class="correctness_mismatch",
            value_error=None,
            gradient_error=None,
            runtime_seconds=None,
            memory_peak_bytes=None,
            batching_support="not_supported",
            transform_support="LLVM Enzyme runner",
            dtype="float64",
            device="cpu",
            source_of_truth="scpn_reference",
            setup_instructions=(
                "Configured Enzyme runner output did not match the SCPN reference "
                f"(value_error={value_error:.3e}, gradient_error={gradient_error:.3e})."
            ),
            claim_boundary="Correctness hard gap only; no hidden success or promoted claim.",
            dependency_versions=_backend_dependency_versions("enzyme"),
            toolchain=toolchain,
        )
    return ExternalComparisonRow(
        case_id="bounded_phase_objective",
        backend="enzyme",
        status="success",
        failure_class=None,
        value_error=value_error,
        gradient_error=gradient_error,
        runtime_seconds=runtime,
        memory_peak_bytes=max(int(peak), int(values.nbytes + reference_gradient.nbytes)),
        batching_support="not_supported",
        transform_support="LLVM Enzyme runner",
        dtype="float64",
        device="cpu",
        source_of_truth="scpn_reference",
        setup_instructions=None,
        claim_boundary=(
            "Bounded CPU LLVM/Enzyme runner comparison against the SCPN reference; "
            "no provider, QPU, GPU, arbitrary-program AD, or production performance claim."
        ),
        dependency_versions=_backend_dependency_versions("enzyme"),
        toolchain=toolchain,
    )


def _catalyst_row() -> ExternalComparisonRow:
    if not _catalyst_runner_configured():
        return _dependency_gap_row(
            "catalyst",
            "not_evaluated",
            "Catalyst qjit/MLIR/QIR runner",
            "Configure SCPN_CATALYST_RUNNER to an executable Catalyst comparison runner.",
        )
    values = np.array([0.2, -0.4], dtype=np.float64)
    reference_value = _bounded_phase_objective(values)
    reference_gradient = _bounded_phase_gradient(values)
    tracemalloc.start()
    start = time.perf_counter()
    try:
        external_value, external_gradient, toolchain = _run_catalyst_reference(values)
    except TimeoutError as exc:
        tracemalloc.stop()
        return _runtime_gap_row(
            "catalyst",
            "not_supported",
            "Catalyst qjit/MLIR/QIR runner",
            str(exc),
        )
    except (RuntimeError, ValueError) as exc:
        tracemalloc.stop()
        return _runtime_gap_row(
            "catalyst",
            "not_supported",
            "Catalyst qjit/MLIR/QIR runner",
            str(exc),
        )
    runtime = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    value_error = abs(reference_value - external_value)
    gradient_error = float(np.max(np.abs(reference_gradient - external_gradient)))
    if value_error > 1.0e-8 or gradient_error > 1.0e-8:
        return ExternalComparisonRow(
            case_id="bounded_phase_objective",
            backend="catalyst",
            status="hard_gap",
            failure_class="correctness_mismatch",
            value_error=None,
            gradient_error=None,
            runtime_seconds=None,
            memory_peak_bytes=None,
            batching_support="not_supported",
            transform_support="Catalyst qjit/MLIR/QIR runner",
            dtype="float64",
            device="cpu",
            source_of_truth="scpn_reference",
            setup_instructions=(
                "Configured Catalyst runner output did not match the SCPN reference "
                f"(value_error={value_error:.3e}, gradient_error={gradient_error:.3e})."
            ),
            claim_boundary="Correctness hard gap only; no hidden success or promoted claim.",
            dependency_versions=_backend_dependency_versions("catalyst"),
            toolchain=toolchain,
        )
    return ExternalComparisonRow(
        case_id="bounded_phase_objective",
        backend="catalyst",
        status="success",
        failure_class=None,
        value_error=value_error,
        gradient_error=gradient_error,
        runtime_seconds=runtime,
        memory_peak_bytes=max(int(peak), int(values.nbytes + reference_gradient.nbytes)),
        batching_support="not_supported",
        transform_support="Catalyst qjit/MLIR/QIR runner",
        dtype="float64",
        device="cpu",
        source_of_truth="scpn_reference",
        setup_instructions=None,
        claim_boundary=(
            "Bounded CPU Catalyst qjit/MLIR/QIR comparison against the SCPN reference; "
            "no provider, QPU, GPU, arbitrary-program AD, or production performance claim."
        ),
        dependency_versions=_backend_dependency_versions("catalyst"),
        toolchain=toolchain,
    )


def _dependency_gap_row(
    backend: str,
    batching_support: str,
    transform_support: str,
    setup: str,
) -> ExternalComparisonRow:
    return ExternalComparisonRow(
        case_id="bounded_phase_objective",
        backend=backend,
        status="hard_gap",
        failure_class="dependency_missing",
        value_error=None,
        gradient_error=None,
        runtime_seconds=None,
        memory_peak_bytes=None,
        batching_support=batching_support,
        transform_support=transform_support,
        dtype="float64",
        device="cpu",
        source_of_truth="scpn_reference",
        setup_instructions=setup,
        claim_boundary="Dependency hard gap only; no hidden success or promoted claim.",
        dependency_versions=_backend_dependency_versions(backend),
    )


def _runtime_gap_row(
    backend: str,
    batching_support: str,
    transform_support: str,
    reason: str,
) -> ExternalComparisonRow:
    return ExternalComparisonRow(
        case_id="bounded_phase_objective",
        backend=backend,
        status="hard_gap",
        failure_class="runtime_error",
        value_error=None,
        gradient_error=None,
        runtime_seconds=None,
        memory_peak_bytes=None,
        batching_support=batching_support,
        transform_support=transform_support,
        dtype="float64",
        device="cpu",
        source_of_truth="scpn_reference",
        setup_instructions=reason,
        claim_boundary="Runtime comparison gap only; no hidden success or promoted claim.",
        dependency_versions=_backend_dependency_versions(backend),
    )


def _unsupported_gap_row(
    *,
    backend: str,
    failure_class: str,
    batching_support: str,
    transform_support: str,
    dtype: str,
    device: str,
    setup: str,
) -> ExternalComparisonRow:
    return ExternalComparisonRow(
        case_id=f"bounded_phase_objective_{failure_class}",
        backend=backend,
        status="hard_gap",
        failure_class=failure_class,
        value_error=None,
        gradient_error=None,
        runtime_seconds=None,
        memory_peak_bytes=None,
        batching_support=batching_support,
        transform_support=transform_support,
        dtype=dtype,
        device=device,
        source_of_truth="scpn_reference",
        setup_instructions=setup,
        claim_boundary="Unsupported-route hard gap only; no hidden success or promoted claim.",
        dependency_versions=_backend_dependency_versions(backend),
    )


def _identical_circuit_problem() -> tuple[
    PhaseQNodeCircuit,
    NDArray[np.float64],
    tuple[tuple[object, ...], ...],
    str,
    str,
]:
    operations: tuple[tuple[object, ...], ...] = (("ry", (0,), 0),)
    observable_label = "Z0"
    values = np.array([0.4], dtype=np.float64)
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=operations,
        observable=PauliTerm(1.0, ((0, "z"),)),
    )
    fingerprint = _same_circuit_fingerprint(operations, observable_label, values)
    return circuit, values, operations, observable_label, fingerprint


def _same_circuit_fingerprint(
    operations: tuple[tuple[object, ...], ...],
    observable: str,
    values: NDArray[np.float64],
) -> str:
    payload = {
        "case_id": "single_ry_z_expectation_exact_state",
        "operations": _jsonify_operations(operations),
        "observable": observable,
        "parameter_values": values.tolist(),
        "execution_mode": "exact_state",
        "shots": None,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _qiskit_identical_circuit_row(
    *,
    values: NDArray[np.float64],
    operations: tuple[tuple[object, ...], ...],
    observable_label: str,
    fingerprint: str,
    scpn_value: float,
    scpn_gradient: tuple[float, ...],
) -> IdenticalCircuitGradientComparisonRow:
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter
        from qiskit.quantum_info import SparsePauliOp

        from ..phase.qiskit_bridge import execute_qiskit_statevector_parameter_shift
    except ImportError:
        return _identical_circuit_gap_row(
            backend="qiskit",
            failure_class="dependency_missing",
            setup="Install qiskit to run exact-state identical-circuit comparison.",
            operations=operations,
            observable_label=observable_label,
            values=values,
            fingerprint=fingerprint,
        )
    try:
        theta = Parameter("theta")
        qiskit_circuit = QuantumCircuit(1)
        qiskit_circuit.ry(theta, 0)
        result = execute_qiskit_statevector_parameter_shift(
            qiskit_circuit,
            SparsePauliOp.from_list([("Z", 1.0)]),
            (theta,),
            values,
        )
    except Exception as exc:  # pragma: no cover - defensive optional boundary
        return _identical_circuit_gap_row(
            backend="qiskit",
            failure_class="runtime_error",
            setup=str(exc),
            operations=operations,
            observable_label=observable_label,
            values=values,
            fingerprint=fingerprint,
        )
    backend_gradient = tuple(float(item) for item in result.gradient)
    return _identical_circuit_success_row(
        backend="qiskit",
        operations=operations,
        observable_label=observable_label,
        values=values,
        fingerprint=fingerprint,
        scpn_value=scpn_value,
        backend_value=float(result.value),
        scpn_gradient=scpn_gradient,
        backend_gradient=backend_gradient,
        evaluations=result.evaluations,
    )


def _pennylane_identical_circuit_row(
    *,
    circuit: PhaseQNodeCircuit,
    values: NDArray[np.float64],
    operations: tuple[tuple[object, ...], ...],
    observable_label: str,
    fingerprint: str,
    scpn_value: float,
    scpn_gradient: tuple[float, ...],
) -> IdenticalCircuitGradientComparisonRow:
    try:
        from ..phase.pennylane_bridge import check_pennylane_phase_qnode_round_trip
    except ImportError:
        return _identical_circuit_gap_row(
            backend="pennylane",
            failure_class="dependency_missing",
            setup="Install pennylane to run exact-state identical-circuit comparison.",
            operations=operations,
            observable_label=observable_label,
            values=values,
            fingerprint=fingerprint,
        )
    try:
        result = check_pennylane_phase_qnode_round_trip(
            circuit,
            values,
            shots=None,
            value_tolerance=1.0e-12,
            gradient_tolerance=1.0e-12,
        )
    except ImportError:
        return _identical_circuit_gap_row(
            backend="pennylane",
            failure_class="dependency_missing",
            setup="Install pennylane to run exact-state identical-circuit comparison.",
            operations=operations,
            observable_label=observable_label,
            values=values,
            fingerprint=fingerprint,
        )
    except Exception as exc:  # pragma: no cover - defensive optional boundary
        return _identical_circuit_gap_row(
            backend="pennylane",
            failure_class="runtime_error",
            setup=str(exc),
            operations=operations,
            observable_label=observable_label,
            values=values,
            fingerprint=fingerprint,
        )
    backend_gradient = tuple(float(item) for item in result.pennylane_gradient)
    return _identical_circuit_success_row(
        backend="pennylane",
        operations=operations,
        observable_label=observable_label,
        values=values,
        fingerprint=fingerprint,
        scpn_value=scpn_value,
        backend_value=float(result.pennylane_value),
        scpn_gradient=scpn_gradient,
        backend_gradient=backend_gradient,
        evaluations=result.evaluations,
    )


def _identical_circuit_success_row(
    *,
    backend: str,
    operations: tuple[tuple[object, ...], ...],
    observable_label: str,
    values: NDArray[np.float64],
    fingerprint: str,
    scpn_value: float,
    backend_value: float,
    scpn_gradient: tuple[float, ...],
    backend_gradient: tuple[float, ...],
    evaluations: int,
) -> IdenticalCircuitGradientComparisonRow:
    gradient_delta = np.asarray(scpn_gradient, dtype=np.float64) - np.asarray(
        backend_gradient,
        dtype=np.float64,
    )
    return IdenticalCircuitGradientComparisonRow(
        case_id="single_ry_z_expectation_exact_state",
        backend=backend,
        status="success",
        failure_class=None,
        circuit_fingerprint=fingerprint,
        operations=operations,
        observable=observable_label,
        parameter_values=tuple(float(item) for item in values),
        execution_mode="exact_state",
        shots=None,
        scpn_value=scpn_value,
        backend_value=backend_value,
        value_error=abs(scpn_value - backend_value),
        scpn_gradient=scpn_gradient,
        backend_gradient=backend_gradient,
        gradient_error=float(np.max(np.abs(gradient_delta))) if gradient_delta.size else 0.0,
        evaluations=int(evaluations),
        dependency_versions=_backend_dependency_versions(backend),
        claim_boundary=(
            "Exact-state same-circuit local gradient comparison against SCPN Phase-QNode; "
            "no provider submission, hardware execution, finite-shot sampling, or "
            "production performance claim."
        ),
    )


def _identical_circuit_gap_row(
    *,
    backend: str,
    failure_class: str,
    setup: str,
    operations: tuple[tuple[object, ...], ...],
    observable_label: str,
    values: NDArray[np.float64],
    fingerprint: str,
) -> IdenticalCircuitGradientComparisonRow:
    return IdenticalCircuitGradientComparisonRow(
        case_id="single_ry_z_expectation_exact_state",
        backend=backend,
        status="hard_gap",
        failure_class=failure_class,
        circuit_fingerprint=fingerprint,
        operations=operations,
        observable=observable_label,
        parameter_values=tuple(float(item) for item in values),
        execution_mode="exact_state",
        shots=None,
        scpn_value=None,
        backend_value=None,
        value_error=None,
        scpn_gradient=None,
        backend_gradient=None,
        gradient_error=None,
        evaluations=None,
        dependency_versions=_backend_dependency_versions(backend),
        claim_boundary=f"{setup} No hidden success or promoted comparison claim.",
    )


def _jsonify_operations(operations: tuple[tuple[object, ...], ...]) -> list[list[object]]:
    return [[_jsonify_operation_part(part) for part in operation] for operation in operations]


def _jsonify_operation_part(part: object) -> object:
    if isinstance(part, tuple):
        return [_jsonify_operation_part(item) for item in part]
    if isinstance(part, np.integer):
        return int(part)
    if isinstance(part, np.floating):
        return float(part)
    return part


def _bounded_phase_objective(values: NDArray[np.float64]) -> float:
    return float(math.cos(values[0]) + 0.25 * math.sin(values[1]))


def _bounded_phase_gradient(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([-math.sin(values[0]), 0.25 * math.cos(values[1])], dtype=np.float64)


def _run_jax_reference(values: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    def objective(x: Any) -> Any:
        return jnp.cos(x[0]) + 0.25 * jnp.sin(x[1])

    value, gradient = jax.value_and_grad(objective)(jnp.asarray(values, dtype=jnp.float64))
    return float(value), np.asarray(gradient, dtype=np.float64)


def _run_pytorch_reference(values: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
    import torch

    def objective(x: Any) -> Any:
        return torch.cos(x[0]) + 0.25 * torch.sin(x[1])

    tensor = torch.tensor(values, dtype=torch.float64)
    gradient = torch.func.grad(objective)(tensor)
    return float(objective(tensor).detach().cpu().item()), gradient.detach().cpu().numpy()


def _run_tensorflow_reference(values: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
    import tensorflow as tf

    tensor = tf.Variable(values, dtype=tf.float64)
    with tf.GradientTape() as tape:
        value = tf.cos(tensor[0]) + tf.constant(0.25, dtype=tf.float64) * tf.sin(tensor[1])
    gradient = tape.gradient(value, tensor)
    if gradient is None:
        raise RuntimeError("TensorFlow GradientTape returned no gradient")
    return float(value.numpy()), np.asarray(gradient.numpy(), dtype=np.float64)


def _run_pennylane_reference(values: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
    import pennylane as qml

    try:
        from pennylane import numpy as pnp
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise ImportError("PennyLane NumPy interface is unavailable") from exc

    dev = qml.device("default.qubit", wires=2)

    def circuit_body(x: Any) -> Any:
        qml.RY(x[0], wires=0)
        qml.RY(x[1], wires=1)
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

    circuit: Any = qml.qnode(dev, interface="autograd")(circuit_body)

    def objective(x: Any) -> Any:
        z0, x1 = circuit(x)
        return z0 + 0.25 * x1

    params = pnp.array(values, requires_grad=True)
    value = objective(params)
    gradient = qml.grad(objective)(params)
    return float(value), np.asarray(gradient, dtype=np.float64)


def _run_enzyme_reference(
    values: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64], dict[str, str]]:
    runner = os.environ.get("SCPN_ENZYME_RUNNER")
    if not runner:
        raise RuntimeError("SCPN_ENZYME_RUNNER is not configured")
    payload = json.dumps(
        {
            "schema": "scpn_qc_enzyme_runner_request_v1",
            "case_id": "bounded_phase_objective",
            "values": values.tolist(),
            "objective": "cos(x0)+0.25*sin(x1)",
            "gradient_contract": ["-sin(x0)", "0.25*cos(x1)"],
            "dtype": "float64",
        },
        sort_keys=True,
    )
    try:
        completed = subprocess.run(
            [runner],
            input=payload,
            text=True,
            capture_output=True,
            check=False,
            timeout=_enzyme_runner_timeout_seconds(),
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("Configured Enzyme runner timed out") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "no stderr"
        raise RuntimeError(f"Configured Enzyme runner failed: {stderr}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Configured Enzyme runner did not emit valid JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("Configured Enzyme runner JSON must be an object")
    value = _as_finite_scalar("Enzyme runner value", result.get("value"))
    gradient = _as_gradient_vector("Enzyme runner gradient", result.get("gradient"), values.size)
    toolchain = _as_toolchain_metadata(result.get("toolchain"), label="Enzyme")
    return value, gradient, toolchain


def _run_catalyst_reference(
    values: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64], dict[str, str]]:
    runner = os.environ.get("SCPN_CATALYST_RUNNER")
    if not runner:
        raise RuntimeError("SCPN_CATALYST_RUNNER is not configured")
    payload = json.dumps(
        {
            "schema": "scpn_qc_catalyst_runner_request_v1",
            "case_id": "bounded_phase_objective",
            "values": values.tolist(),
            "objective": "cos(x0)+0.25*sin(x1)",
            "gradient_contract": ["-sin(x0)", "0.25*cos(x1)"],
            "dtype": "float64",
            "compiler_workflow": "catalyst_qjit_mlir_qir",
        },
        sort_keys=True,
    )
    try:
        completed = subprocess.run(
            [runner],
            input=payload,
            text=True,
            capture_output=True,
            check=False,
            timeout=_catalyst_runner_timeout_seconds(),
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("Configured Catalyst runner timed out") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "no stderr"
        raise RuntimeError(f"Configured Catalyst runner failed: {stderr}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Configured Catalyst runner did not emit valid JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("Configured Catalyst runner JSON must be an object")
    value = _as_finite_scalar("Catalyst runner value", result.get("value"))
    gradient = _as_gradient_vector("Catalyst runner gradient", result.get("gradient"), values.size)
    toolchain = _as_toolchain_metadata(result.get("toolchain"), label="Catalyst")
    return value, gradient, toolchain


def _as_finite_scalar(name: str, value: object) -> float:
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind in {"b", "c", "O", "S", "U"}:
        raise ValueError(f"{name} must be a finite real scalar")
    scalar = float(raw.item())
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _as_gradient_vector(name: str, value: object, width: int) -> NDArray[np.float64]:
    raw = np.asarray(value)
    if raw.dtype.kind in {"b", "c", "O", "S", "U"}:
        raise ValueError(f"{name} must contain finite real numeric values")
    gradient = np.asarray(value, dtype=np.float64)
    if gradient.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {gradient.shape}")
    if not np.all(np.isfinite(gradient)):
        raise ValueError(f"{name} must contain finite real numeric values")
    return gradient.astype(np.float64, copy=True)


def _as_toolchain_metadata(value: object, *, label: str) -> dict[str, str]:
    if value is None:
        return {label.lower(): "configured-runner"}
    if not isinstance(value, dict):
        raise ValueError(f"{label} runner toolchain metadata must be an object")
    metadata = {str(key): str(item) for key, item in value.items()}
    if any(not key or not item for key, item in metadata.items()):
        raise ValueError(f"{label} runner toolchain metadata cannot contain empty keys or values")
    return metadata


def _backend_dependency_versions(backend: str) -> dict[str, str]:
    packages_by_backend: dict[str, tuple[str, ...]] = {
        "jax": ("jax", "jaxlib"),
        "pytorch": ("torch",),
        "tensorflow": ("tensorflow", "tensorflow-cpu"),
        "pennylane": ("pennylane",),
        "qiskit": ("qiskit",),
        "enzyme": ("llvm", "enzyme", "enzyme_ad"),
        "catalyst": ("pennylane-catalyst", "catalyst", "mlir", "llvm"),
    }
    versions = {
        package: _installed_version(package) for package in packages_by_backend.get(backend, ())
    }
    if backend == "enzyme":
        plugin = os.environ.get("ENZYME_LLVM_PLUGIN")
        if plugin:
            versions["enzyme_llvm_plugin"] = (
                f"file:{plugin}" if os.path.exists(plugin) else f"missing:{plugin}"
            )
        runner = os.environ.get("SCPN_ENZYME_RUNNER")
        if runner:
            versions["enzyme_runner"] = (
                f"executable:{runner}" if os.path.exists(runner) else f"missing:{runner}"
            )
    if backend == "catalyst":
        runner = os.environ.get("SCPN_CATALYST_RUNNER")
        if runner:
            versions["catalyst_runner"] = (
                f"executable:{runner}" if os.path.exists(runner) else f"missing:{runner}"
            )
    return versions


def _installed_version(package: str) -> str:
    binary = shutil.which(package)
    if package in {"llvm", "enzyme", "mlir"} and binary:
        return f"executable:{binary}"
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        module_name = "tensorflow" if package == "tensorflow" else package
        try:
            module = import_module(module_name)
        except Exception:
            return "not_installed"
        version = getattr(module, "__version__", None)
        return str(version) if version else "importable_unknown_version"


def _enzyme_runner_timeout_seconds() -> float:
    raw = os.environ.get("SCPN_ENZYME_RUNNER_TIMEOUT_SECONDS", "10")
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ValueError("SCPN_ENZYME_RUNNER_TIMEOUT_SECONDS must be numeric") from exc
    if not np.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("SCPN_ENZYME_RUNNER_TIMEOUT_SECONDS must be positive")
    return timeout


def _catalyst_runner_timeout_seconds() -> float:
    raw = os.environ.get("SCPN_CATALYST_RUNNER_TIMEOUT_SECONDS", "10")
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ValueError("SCPN_CATALYST_RUNNER_TIMEOUT_SECONDS must be numeric") from exc
    if not np.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("SCPN_CATALYST_RUNNER_TIMEOUT_SECONDS must be positive")
    return timeout


def _enzyme_tooling_available() -> bool:
    configured_plugin = os.environ.get("ENZYME_LLVM_PLUGIN")
    return shutil.which("enzyme") is not None or bool(
        configured_plugin and os.path.exists(configured_plugin)
    )


def _catalyst_tooling_available() -> bool:
    try:
        import_module("catalyst")
    except Exception:
        return shutil.which("catalyst") is not None or shutil.which("mlir-opt") is not None
    return True


def _enzyme_runner_configured() -> bool:
    runner = os.environ.get("SCPN_ENZYME_RUNNER")
    return bool(runner and os.path.exists(runner) and _enzyme_tooling_available())


def _catalyst_runner_configured() -> bool:
    runner = os.environ.get("SCPN_CATALYST_RUNNER")
    return bool(runner and os.path.exists(runner) and _catalyst_tooling_available())


__all__ = [
    "ExternalComparisonArtifact",
    "ExternalComparisonRow",
    "IdenticalCircuitGradientComparisonArtifact",
    "IdenticalCircuitGradientComparisonRow",
    "REQUIRED_EXTERNAL_COMPARISON_ROW_FIELDS",
    "external_comparison_failure_mode_rows",
    "run_differentiable_external_comparison_suite",
    "run_identical_circuit_gradient_comparison_suite",
    "write_differentiable_external_comparison",
    "write_identical_circuit_gradient_comparison",
]
