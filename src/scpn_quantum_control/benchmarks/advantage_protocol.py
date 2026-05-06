# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — S2 scaling benchmark protocol
"""Claim-bounded protocol for S2 scaling and advantage benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

BaselineKind = Literal[
    "classical_ode",
    "dense_exact_diagonalisation",
    "dense_trotter_expm",
    "sparse_eigsh",
    "mps_tensor_network",
    "gpu_dense_reference",
    "aer_statevector",
    "qpu_hardware",
]


@dataclass(frozen=True)
class ScalingBaseline:
    """One baseline column required by the S2 scaling benchmark."""

    kind: BaselineKind
    label: str
    required: bool
    max_qubits: int | None
    metrics: tuple[str, ...]
    claim_boundary: str

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("label must be non-empty")
        if self.max_qubits is not None and self.max_qubits < 1:
            raise ValueError("max_qubits must be positive when provided")
        if not self.metrics:
            raise ValueError("metrics must be non-empty")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        """Serialise this baseline."""
        return {
            "kind": self.kind,
            "label": self.label,
            "required": self.required,
            "max_qubits": self.max_qubits,
            "metrics": list(self.metrics),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class ScalingProtocol:
    """Complete S2 benchmark protocol manifest."""

    protocol_id: str
    sizes: tuple[int, ...]
    baselines: tuple[ScalingBaseline, ...]
    acceptance: tuple[str, ...]
    falsification: tuple[str, ...]
    claim_boundary: str
    output_schema: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.protocol_id:
            raise ValueError("protocol_id must be non-empty")
        if not self.sizes or any(size < 1 for size in self.sizes):
            raise ValueError("sizes must contain positive integers")
        if sorted(self.sizes) != list(self.sizes):
            raise ValueError("sizes must be sorted")
        if not self.baselines:
            raise ValueError("baselines must be non-empty")
        if not self.acceptance:
            raise ValueError("acceptance must be non-empty")
        if not self.falsification:
            raise ValueError("falsification must be non-empty")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")

    @property
    def required_baselines(self) -> tuple[str, ...]:
        """Return labels for required baseline columns."""
        return tuple(baseline.label for baseline in self.baselines if baseline.required)

    def to_dict(self) -> dict[str, Any]:
        """Serialise this protocol manifest."""
        return {
            "protocol_id": self.protocol_id,
            "sizes": list(self.sizes),
            "baselines": [baseline.to_dict() for baseline in self.baselines],
            "required_baselines": list(self.required_baselines),
            "acceptance": list(self.acceptance),
            "falsification": list(self.falsification),
            "claim_boundary": self.claim_boundary,
            "output_schema": dict(self.output_schema),
        }


@dataclass(frozen=True)
class ScalingRowValidation:
    """Validation result for a set of S2 scaling rows."""

    valid: bool
    missing_required: tuple[str, ...]
    invalid_rows: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the validation result."""
        return {
            "valid": self.valid,
            "missing_required": list(self.missing_required),
            "invalid_rows": list(self.invalid_rows),
        }


def validate_scaling_rows(
    protocol: ScalingProtocol,
    rows: list[dict[str, Any]],
) -> ScalingRowValidation:
    """Validate measured S2 rows against the preregistered protocol."""
    present_required = {
        str(row.get("baseline"))
        for row in rows
        if row.get("baseline") in protocol.required_baselines
        and row.get("status") in {"ok", "skipped"}
    }
    missing_required = tuple(
        baseline for baseline in protocol.required_baselines if baseline not in present_required
    )
    required_keys = tuple(protocol.output_schema.get("row_keys", ()))
    invalid: list[str] = []
    allowed_baselines = {baseline.label for baseline in protocol.baselines}
    for index, row in enumerate(rows):
        missing_keys = [key for key in required_keys if key not in row]
        baseline = row.get("baseline")
        status = row.get("status")
        if missing_keys:
            invalid.append(f"row {index}: missing keys {missing_keys}")
        if baseline not in allowed_baselines:
            invalid.append(f"row {index}: unknown baseline {baseline!r}")
        if status not in {"ok", "skipped", "failed"}:
            invalid.append(f"row {index}: invalid status {status!r}")
        if status == "ok" and row.get("wall_time_ms") is None:
            invalid.append(f"row {index}: ok row requires wall_time_ms")
    return ScalingRowValidation(
        valid=not missing_required and not invalid,
        missing_required=missing_required,
        invalid_rows=tuple(invalid),
    )


def default_s2_scaling_protocol() -> ScalingProtocol:
    """Return the default no-claim S2 scaling benchmark protocol."""
    common_metrics = ("wall_time_ms", "memory_bytes", "status", "notes")
    return ScalingProtocol(
        protocol_id="s2_quantum_advantage_scaling_2026-05-06",
        sizes=(4, 6, 8, 10, 12, 14, 16, 18, 20),
        baselines=(
            ScalingBaseline(
                kind="classical_ode",
                label="classical_ode",
                required=True,
                max_qubits=20,
                metrics=common_metrics + ("final_R",),
                claim_boundary="Classical oscillator ODE baseline; not a Hilbert-space solver.",
            ),
            ScalingBaseline(
                kind="dense_exact_diagonalisation",
                label="dense_eigh",
                required=True,
                max_qubits=14,
                metrics=common_metrics + ("ground_energy",),
                claim_boundary="Dense exact diagonalisation is skipped by memory gate above its cap.",
            ),
            ScalingBaseline(
                kind="sparse_eigsh",
                label="sparse_eigsh",
                required=True,
                max_qubits=20,
                metrics=common_metrics + ("ground_energy", "residual_norm"),
                claim_boundary="Sparse eigensolver timing is not equivalent to full dynamics.",
            ),
            ScalingBaseline(
                kind="mps_tensor_network",
                label="mps_tensor_network",
                required=True,
                max_qubits=20,
                metrics=common_metrics + ("max_bond", "discarded_weight"),
                claim_boundary="MPS results bound classical spoofability; low entanglement weakens advantage claims.",
            ),
            ScalingBaseline(
                kind="gpu_dense_reference",
                label="gpu_dense_reference",
                required=False,
                max_qubits=13,
                metrics=common_metrics + ("device",),
                claim_boundary="GPU dense timings are classical-validation support, not a quantum baseline.",
            ),
            ScalingBaseline(
                kind="aer_statevector",
                label="aer_statevector",
                required=True,
                max_qubits=20,
                metrics=common_metrics + ("trotter_steps", "circuit_depth"),
                claim_boundary="Aer statevector is a simulator baseline, not hardware evidence.",
            ),
            ScalingBaseline(
                kind="qpu_hardware",
                label="qpu_hardware",
                required=False,
                max_qubits=8,
                metrics=("shots", "job_ids", "backend", "raw_counts_path", "status", "notes"),
                claim_boundary=(
                    "Hardware column is evidence only for submitted preregistered sizes; "
                    "absence of IBM credits must degrade gracefully without broad advantage claims."
                ),
            ),
        ),
        acceptance=(
            "All required classical and simulator columns either contain measured rows or explicit size-gated skips.",
            "Every timing row records command, machine, dependency versions, and git commit.",
            "MPS/TN diagnostics are present before any advantage-language figure is promoted.",
            "Hardware rows are optional until credits are approved and must never be extrapolated as broad advantage.",
        ),
        falsification=(
            "If MPS discarded weight remains small at target sizes, any strong quantum-advantage framing is rejected.",
            "If Aer/statevector or sparse classical baselines dominate the proposed quantum path, crossover language is rejected.",
            "If hardware rows are unavailable or noisy, the result remains a classical scaling study only.",
        ),
        claim_boundary=(
            "S2 may publish scaling, memory, and spoofability boundaries. It must not claim broad quantum advantage "
            "without preregistered hardware data and classical tensor-network baselines at the same problem family."
        ),
        output_schema={
            "row_keys": [
                "protocol_id",
                "n_qubits",
                "baseline",
                "status",
                "wall_time_ms",
                "memory_bytes",
                "metric_payload",
                "command",
                "machine",
                "dependencies",
                "git_commit",
                "notes",
            ]
        },
    )
