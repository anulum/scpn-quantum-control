# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- TN/MPS baseline design
"""CPU-first TN/MPS baseline design for S2 scaling follow-up."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

AdapterStatus = Literal["ready", "optional_dependency", "blocked", "owner_gated"]
SizeClass = Literal["pilot", "extension"]

TN_MPS_BASELINE_DESIGN_SCHEMA = "scpn_qc_tn_mps_baseline_design_v1"
TN_MPS_BASELINE_DESIGN_CLAIM_BOUNDARY = (
    "Design and preregistration evidence only; no N>=30 tensor-network row, "
    "hardware advantage, or broad quantum-advantage claim is established."
)
DEFAULT_TARGET_SIZES = (30, 32, 36, 40)


@dataclass(frozen=True)
class TNBaselineAdapter:
    """One candidate implementation path for TN/MPS baseline rows."""

    name: str
    language: str
    dependency: str
    status: AdapterStatus
    role: str
    max_target_qubits: int
    claim_boundary: str
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.name, "name")
        _require_text(self.language, "language")
        _require_text(self.dependency, "dependency")
        _require_text(self.role, "role")
        _require_positive_int(self.max_target_qubits, "max_target_qubits")
        _require_text(self.claim_boundary, "claim_boundary")
        _require_nonempty_texts(self.notes, "notes")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible adapter data."""
        return asdict(self)


@dataclass(frozen=True)
class TNBaselineSizePlan:
    """One target-size row in the N=30-40 TN/MPS baseline design."""

    n_qubits: int
    size_class: SizeClass
    cpu_first_adapter: str
    required_rows: tuple[str, ...]
    acceptance_gates: tuple[str, ...]
    blocked_claims: tuple[str, ...]
    gpu_followup: str
    qwc5_1_unblocker: bool

    def __post_init__(self) -> None:
        _require_positive_int(self.n_qubits, "n_qubits")
        _require_text(self.cpu_first_adapter, "cpu_first_adapter")
        _require_nonempty_texts(self.required_rows, "required_rows")
        _require_nonempty_texts(self.acceptance_gates, "acceptance_gates")
        _require_nonempty_texts(self.blocked_claims, "blocked_claims")
        _require_text(self.gpu_followup, "gpu_followup")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible size-plan data."""
        return asdict(self)


@dataclass(frozen=True)
class TNBaselineDesign:
    """Complete deterministic TN/MPS baseline design manifest."""

    schema: str
    target_sizes: tuple[int, ...]
    decision: str
    adapters: tuple[TNBaselineAdapter, ...]
    size_plan: tuple[TNBaselineSizePlan, ...]
    acceptance_gates: tuple[str, ...]
    blocked_claims: tuple[str, ...]
    qwc5_1_unblocked_by: str
    claim_boundary: str = TN_MPS_BASELINE_DESIGN_CLAIM_BOUNDARY
    benchmark_execution_performed: bool = False
    hardware_submission_allowed: bool = False
    advantage_claim_allowed: bool = False

    def __post_init__(self) -> None:
        _validate_target_sizes(self.target_sizes)
        _require_text(self.schema, "schema")
        _require_text(self.decision, "decision")
        if not self.adapters:
            raise ValueError("adapters must be non-empty")
        if len(self.size_plan) != len(self.target_sizes):
            raise ValueError("size_plan must have one row per target size")
        _require_nonempty_texts(self.acceptance_gates, "acceptance_gates")
        _require_nonempty_texts(self.blocked_claims, "blocked_claims")
        _require_text(self.qwc5_1_unblocked_by, "qwc5_1_unblocked_by")
        _require_text(self.claim_boundary, "claim_boundary")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible design data."""
        return {
            "schema": self.schema,
            "target_sizes": list(self.target_sizes),
            "decision": self.decision,
            "adapters": [adapter.to_dict() for adapter in self.adapters],
            "size_plan": [row.to_dict() for row in self.size_plan],
            "acceptance_gates": list(self.acceptance_gates),
            "blocked_claims": list(self.blocked_claims),
            "qwc5_1_unblocked_by": self.qwc5_1_unblocked_by,
            "claim_boundary": self.claim_boundary,
            "benchmark_execution_performed": self.benchmark_execution_performed,
            "hardware_submission_allowed": self.hardware_submission_allowed,
            "advantage_claim_allowed": self.advantage_claim_allowed,
        }


def build_tn_mps_baseline_design(
    target_sizes: Sequence[int] = DEFAULT_TARGET_SIZES,
) -> TNBaselineDesign:
    """Build the CPU-first N=30-40 TN/MPS baseline design manifest."""
    sizes = _validate_target_sizes(tuple(int(size) for size in target_sizes))
    adapters = _default_adapters(max(sizes))
    return TNBaselineDesign(
        schema=TN_MPS_BASELINE_DESIGN_SCHEMA,
        target_sizes=sizes,
        decision=(
            "Use the Python/quimb CPU MPS adapter as the first execution path, "
            "with the bounded native Schmidt/resource model as a deterministic "
            "fallback and validation scaffold. Keep ITensor/Julia and GPU TN "
            "as explicitly blocked follow-ups until owner-gated toolchain work."
        ),
        adapters=adapters,
        size_plan=tuple(_size_plan_row(size) for size in sizes),
        acceptance_gates=(
            "Every N=30-40 row must record status, wall_time_ms, memory_bytes, max_bond, "
            "discarded_weight, entropy proxy, command, machine, dependencies, and git_commit.",
            "Skipped rows must carry explicit size, dependency, or resource-gate reasons.",
            "TN/MPS rows must be compared against the S2 protocol matrix before any "
            "QWC-5.1 promotion.",
        ),
        blocked_claims=(
            "No broad quantum advantage claim from this design artifact.",
            "No tensor-network hardness claim until measured N=30-40 TN rows exist.",
            "No GPU TN comparison until the owner-gated GPU lane promotes it.",
            "No Julia/ITensor parity claim until a maintained Julia adapter is measured.",
        ),
        qwc5_1_unblocked_by=(
            "QWC-5.1 can execute the CPU-first TN/MPS rows once the quimb dependency, "
            "resource caps, and S2 row schema are pinned by this manifest."
        ),
    )


def render_tn_mps_baseline_design_markdown(design: TNBaselineDesign) -> str:
    """Render a human-reviewable TN/MPS baseline design report."""
    lines = [
        "# TN/MPS Baseline Design",
        "",
        "This is the QWC-4.2 design artifact for the N=30-40 tensor-network",
        "baseline path. It is planning and preregistration evidence only.",
        "",
        "## Boundary",
        "",
        design.claim_boundary,
        "",
        "## Decision",
        "",
        design.decision,
        "",
        "## Adapters",
        "",
        "| adapter | language | dependency | status | role | max N |",
        "| --- | --- | --- | --- | --- | ---: |",
    ]
    for adapter in design.adapters:
        lines.append(
            "| {name} | {language} | {dependency} | {status} | {role} | "
            "{max_target_qubits} |".format(**adapter.to_dict())
        )
    lines.extend(
        [
            "",
            "## Size Plan",
            "",
            "| N | class | CPU-first adapter | QWC-5.1 unblocker | GPU follow-up |",
            "| ---: | --- | --- | --- | --- |",
        ]
    )
    for row in design.size_plan:
        lines.append(
            "| {n_qubits} | {size_class} | {cpu_first_adapter} | "
            "{qwc5_1_unblocker} | {gpu_followup} |".format(**row.to_dict())
        )
    lines.extend(["", "## Acceptance Gates", ""])
    lines.extend(f"- {gate}" for gate in design.acceptance_gates)
    lines.extend(["", "## Blocked Claims", ""])
    lines.extend(f"- {claim}" for claim in design.blocked_claims)
    lines.extend(
        [
            "",
            "## QWC-5.1 Unblocker",
            "",
            design.qwc5_1_unblocked_by,
            "",
            "## Regeneration",
            "",
            "```bash",
            "scpn-bench s2-tn-mps-baseline-design",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def _default_adapters(max_target_qubits: int) -> tuple[TNBaselineAdapter, ...]:
    return (
        TNBaselineAdapter(
            name="quimb_mps_cpu",
            language="Python",
            dependency="quimb[tensor] optional extra",
            status="optional_dependency",
            role="CPU DMRG/TEBD execution adapter for nearest-neighbour or explicitly truncated K_nm",
            max_target_qubits=max_target_qubits,
            claim_boundary="Measured rows only; optional dependency absence must emit skipped rows.",
            notes=(
                "First execution path for QWC-5.1.",
                "Long-range K_nm truncation must report omitted coupling mass.",
            ),
        ),
        TNBaselineAdapter(
            name="bounded_native_schmidt",
            language="Python/NumPy",
            dependency="none",
            status="ready",
            role="Deterministic resource and discarded-weight scaffold for row validation",
            max_target_qubits=max_target_qubits,
            claim_boundary="Resource/design scaffold only; not an executed MPS solver row.",
            notes=(
                "Uses existing MPS memory and Schmidt-bound estimators.",
                "Provides fail-closed fallback metadata when quimb is absent.",
            ),
        ),
        TNBaselineAdapter(
            name="itensor_julia",
            language="Julia",
            dependency="ITensor.jl",
            status="blocked",
            role="Future independent MPS parity adapter after Julia ownership is assigned",
            max_target_qubits=max_target_qubits,
            claim_boundary="No Julia parity or speed claim until measured and maintained.",
            notes=(
                "Not part of the CPU-first QWC-5.1 gate.",
                "Requires separate dependency and CI ownership.",
            ),
        ),
        TNBaselineAdapter(
            name="gpu_tn",
            language="CUDA/Python",
            dependency="cuTensorNet or equivalent",
            status="owner_gated",
            role="Future GPU tensor-network comparison lane",
            max_target_qubits=max_target_qubits,
            claim_boundary="GPU comparison is blocked until the owner-gated GPU lane promotes it.",
            notes=(
                "Kept out of CPU-first QWC-4.2/QWC-5.1.",
                "Must not be used to imply current GPU TN evidence.",
            ),
        ),
    )


def _size_plan_row(n_qubits: int) -> TNBaselineSizePlan:
    return TNBaselineSizePlan(
        n_qubits=n_qubits,
        size_class="pilot" if n_qubits <= 32 else "extension",
        cpu_first_adapter="quimb_mps_cpu",
        required_rows=(
            "classical_ode",
            "mps_tensor_network",
            "aer_statevector_or_skip",
            "qpu_hardware_optional",
        ),
        acceptance_gates=(
            "mps_tensor_network row is ok or skipped with explicit dependency/resource reason",
            "max_bond and discarded_weight are recorded for ok rows",
            "advantage_claim_allowed remains false",
        ),
        blocked_claims=(
            "quantum advantage",
            "tensor-network hardness",
            "hardware scaling win",
        ),
        gpu_followup="defer to owner-gated GPU lane #32",
        qwc5_1_unblocker=True,
    )


def _validate_target_sizes(target_sizes: tuple[int, ...]) -> tuple[int, ...]:
    if not target_sizes:
        raise ValueError("target_sizes must be non-empty")
    if any(size < 2 for size in target_sizes):
        raise ValueError("target_sizes must contain qubit counts >= 2")
    if tuple(sorted(target_sizes)) != target_sizes:
        raise ValueError("target_sizes must be sorted")
    if len(set(target_sizes)) != len(target_sizes):
        raise ValueError("target_sizes must be unique")
    return target_sizes


def _require_text(value: str, name: str) -> None:
    if not value:
        raise ValueError(f"{name} must be non-empty")


def _require_nonempty_texts(values: tuple[str, ...], name: str) -> None:
    if not values or any(not value for value in values):
        raise ValueError(f"{name} must contain non-empty entries")


def _require_positive_int(value: int, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive")


__all__ = [
    "DEFAULT_TARGET_SIZES",
    "TN_MPS_BASELINE_DESIGN_CLAIM_BOUNDARY",
    "TN_MPS_BASELINE_DESIGN_SCHEMA",
    "TNBaselineAdapter",
    "TNBaselineDesign",
    "TNBaselineSizePlan",
    "build_tn_mps_baseline_design",
    "render_tn_mps_baseline_design_markdown",
]
