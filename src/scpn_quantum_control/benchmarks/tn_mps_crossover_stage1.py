# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — TN/MPS crossover stage-1 gate
"""QWC-5.1 stage-1 gate for larger-than-16 TN/MPS crossover rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

from .tn_mps_baseline_design import TNBaselineDesign, build_tn_mps_baseline_design

TN_MPS_CROSSOVER_STAGE1_SCHEMA = "scpn_qc_tn_mps_crossover_stage1_v1"
TN_MPS_CROSSOVER_PROTOCOL_ID = "qwc_5_1_tn_mps_crossover_stage1"
TN_MPS_CROSSOVER_CLAIM_BOUNDARY = (
    "Stage-1 schema and admission evidence for N=30-40 tensor-network crossover "
    "rows only; no N=30-40 solver row, hardware advantage, tensor-network "
    "hardness, or broad quantum-advantage claim is established."
)
TN_MPS_CROSSOVER_REQUIRED_FIELDS = (
    "protocol_id",
    "n_qubits",
    "baseline",
    "status",
    "wall_time_ms",
    "memory_bytes",
    "max_bond",
    "discarded_weight",
    "entropy_proxy",
    "truncation_policy",
    "omitted_coupling_mass",
    "command",
    "machine",
    "dependencies",
    "git_commit",
    "host_load",
    "claim_boundary",
    "notes",
)


@dataclass(frozen=True)
class TNMPSCrossoverRowSchema:
    """Schema required before QWC-5.1 N=30-40 TN/MPS rows can be admitted."""

    protocol_id: str
    target_sizes: tuple[int, ...]
    required_fields: tuple[str, ...]
    required_baselines: tuple[str, ...]
    allowed_statuses: tuple[str, ...]
    claim_boundary: str

    def __post_init__(self) -> None:
        _require_text(self.protocol_id, "protocol_id")
        _require_target_sizes(self.target_sizes)
        _require_nonempty_texts(self.required_fields, "required_fields")
        _require_nonempty_texts(self.required_baselines, "required_baselines")
        _require_nonempty_texts(self.allowed_statuses, "allowed_statuses")
        _require_text(self.claim_boundary, "claim_boundary")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible schema data."""
        return {
            "protocol_id": self.protocol_id,
            "target_sizes": list(self.target_sizes),
            "required_fields": list(self.required_fields),
            "required_baselines": list(self.required_baselines),
            "allowed_statuses": list(self.allowed_statuses),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class TNMPSCrossoverGate:
    """One pass/fail admission gate for the QWC-5.1 stage-1 report."""

    gate_id: str
    passed: bool
    evidence: str
    blocker: str

    def __post_init__(self) -> None:
        _require_text(self.gate_id, "gate_id")
        _require_text(self.evidence, "evidence")
        if not self.passed:
            _require_text(self.blocker, "blocker")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible gate data."""
        return {
            "gate_id": self.gate_id,
            "passed": self.passed,
            "evidence": self.evidence,
            "blocker": self.blocker,
        }


@dataclass(frozen=True)
class TNMPSCrossoverStage1Report:
    """Complete QWC-5.1 stage-1 crossover admission report."""

    schema: str
    row_schema: TNMPSCrossoverRowSchema
    design_schema: str
    gates: tuple[TNMPSCrossoverGate, ...]
    blocked_claims: tuple[str, ...]
    owner_gated_followups: tuple[str, ...]
    claim_boundary: str
    stage2_compute_owner_gated: bool
    benchmark_execution_performed: bool
    advantage_claim_allowed: bool

    def __post_init__(self) -> None:
        _require_text(self.schema, "schema")
        _require_text(self.design_schema, "design_schema")
        if not self.gates:
            raise ValueError("gates must be non-empty")
        _require_nonempty_texts(self.blocked_claims, "blocked_claims")
        _require_nonempty_texts(self.owner_gated_followups, "owner_gated_followups")
        _require_text(self.claim_boundary, "claim_boundary")

    @property
    def passed(self) -> bool:
        """Whether every stage-1 admission gate passed."""
        return all(gate.passed for gate in self.gates)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible report data."""
        return {
            "schema": self.schema,
            "row_schema": self.row_schema.to_dict(),
            "design_schema": self.design_schema,
            "gates": [gate.to_dict() for gate in self.gates],
            "blocked_claims": list(self.blocked_claims),
            "owner_gated_followups": list(self.owner_gated_followups),
            "claim_boundary": self.claim_boundary,
            "stage2_compute_owner_gated": self.stage2_compute_owner_gated,
            "benchmark_execution_performed": self.benchmark_execution_performed,
            "advantage_claim_allowed": self.advantage_claim_allowed,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class TNMPSCrossoverRowValidation:
    """Validation result for future QWC-5.1 measured or skipped rows."""

    valid: bool
    invalid_rows: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible validation data."""
        return {"valid": self.valid, "invalid_rows": list(self.invalid_rows)}


def build_tn_mps_crossover_stage1(
    design: TNBaselineDesign | None = None,
) -> TNMPSCrossoverStage1Report:
    """Build the QWC-5.1 N=30-40 TN/MPS crossover stage-1 report.

    Parameters
    ----------
    design:
        Existing TN/MPS baseline design. When omitted, the default
        N=30-40 design is used.

    Returns
    -------
    TNMPSCrossoverStage1Report
        Deterministic admission report for schema, claim, and owner-gate
        readiness. It does not execute the stage-2 compute rows.
    """
    source_design = design if design is not None else build_tn_mps_baseline_design()
    row_schema = TNMPSCrossoverRowSchema(
        protocol_id=TN_MPS_CROSSOVER_PROTOCOL_ID,
        target_sizes=source_design.target_sizes,
        required_fields=TN_MPS_CROSSOVER_REQUIRED_FIELDS,
        required_baselines=("classical_ode", "mps_tensor_network", "aer_statevector_or_skip"),
        allowed_statuses=("ok", "skipped", "failed"),
        claim_boundary=TN_MPS_CROSSOVER_CLAIM_BOUNDARY,
    )
    gates = (
        TNMPSCrossoverGate(
            gate_id="target_sizes_exceed_sixteen",
            passed=all(size > 16 for size in source_design.target_sizes),
            evidence=f"target_sizes={source_design.target_sizes!r}",
            blocker="target sizes must all exceed the 16-node exact-simulation boundary",
        ),
        TNMPSCrossoverGate(
            gate_id="row_schema_pinned",
            passed=set(TN_MPS_CROSSOVER_REQUIRED_FIELDS).issubset(row_schema.required_fields),
            evidence=f"{len(row_schema.required_fields)} required row fields pinned",
            blocker="row schema is missing required provenance or TN diagnostic fields",
        ),
        TNMPSCrossoverGate(
            gate_id="stage2_compute_owner_gated",
            passed=True,
            evidence="stage-2 N=30-40 execution remains owner-gated",
            blocker="",
        ),
        TNMPSCrossoverGate(
            gate_id="claim_boundary_closed",
            passed=not source_design.advantage_claim_allowed
            and not source_design.benchmark_execution_performed,
            evidence="design keeps advantage_claim_allowed=false and benchmark_execution_performed=false",
            blocker="design must not promote broad advantage or imply measured N=30-40 rows",
        ),
    )
    return TNMPSCrossoverStage1Report(
        schema=TN_MPS_CROSSOVER_STAGE1_SCHEMA,
        row_schema=row_schema,
        design_schema=source_design.schema,
        gates=gates,
        blocked_claims=(
            "broad quantum advantage",
            "tensor-network hardness",
            "hardware scaling win",
            "GPU tensor-network comparison",
            "Julia/ITensor parity",
        ),
        owner_gated_followups=(
            "execute N=30-40 CPU-first quimb rows with isolated host metadata",
            "promote or skip GPU tensor-network rows only through the owner-gated GPU lane",
            "add maintained Julia/ITensor parity only after Julia toolchain ownership exists",
        ),
        claim_boundary=TN_MPS_CROSSOVER_CLAIM_BOUNDARY,
        stage2_compute_owner_gated=True,
        benchmark_execution_performed=False,
        advantage_claim_allowed=False,
    )


def validate_tn_mps_crossover_rows(
    rows: Sequence[Mapping[str, object]],
    report: TNMPSCrossoverStage1Report | None = None,
) -> TNMPSCrossoverRowValidation:
    """Validate future QWC-5.1 TN/MPS rows against the stage-1 schema."""
    active_report = report if report is not None else build_tn_mps_crossover_stage1()
    schema = active_report.row_schema
    invalid: list[str] = []
    seen: set[tuple[int, str]] = set()
    for index, row in enumerate(rows):
        missing = [field for field in schema.required_fields if field not in row]
        if missing:
            invalid.append(f"row {index}: missing fields {missing}")
        _validate_row_identity(index, row, schema, invalid, seen)
        _validate_row_metrics(index, row, invalid)
        _validate_row_provenance(index, row, invalid)
    return TNMPSCrossoverRowValidation(valid=not invalid, invalid_rows=tuple(invalid))


def render_tn_mps_crossover_stage1_markdown(report: TNMPSCrossoverStage1Report) -> str:
    """Render the QWC-5.1 stage-1 report for public documentation."""
    lines = [
        "# TN/MPS Crossover Stage-1 Gate",
        "",
        "This QWC-5.1 artifact admits the larger-than-16-node N=30-40",
        "tensor-network crossover row format before any owner-gated compute run.",
        "",
        "## Boundary",
        "",
        report.claim_boundary,
        "",
        "## Row Schema",
        "",
        f"- protocol: `{report.row_schema.protocol_id}`",
        f"- target sizes: `{', '.join(str(size) for size in report.row_schema.target_sizes)}`",
        f"- required baselines: `{', '.join(report.row_schema.required_baselines)}`",
        "",
        "| required field |",
        "| --- |",
    ]
    lines.extend(f"| `{field}` |" for field in report.row_schema.required_fields)
    lines.extend(
        ["", "## Stage-1 Gates", "", "| gate | passed | evidence |", "| --- | --- | --- |"]
    )
    lines.extend(
        f"| `{gate.gate_id}` | `{gate.passed}` | {gate.evidence} |" for gate in report.gates
    )
    lines.extend(["", "## Blocked Claims", ""])
    lines.extend(f"- {claim}" for claim in report.blocked_claims)
    lines.extend(["", "## Owner-Gated Follow-ups", ""])
    lines.extend(f"- {item}" for item in report.owner_gated_followups)
    lines.extend(
        [
            "",
            "## Regeneration",
            "",
            "```bash",
            "scpn-bench s2-tn-crossover-stage1",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def _validate_row_identity(
    index: int,
    row: Mapping[str, object],
    schema: TNMPSCrossoverRowSchema,
    invalid: list[str],
    seen: set[tuple[int, str]],
) -> None:
    if row.get("protocol_id") != schema.protocol_id:
        invalid.append(f"row {index}: protocol_id must be {schema.protocol_id!r}")
    n_qubits = row.get("n_qubits")
    if not isinstance(n_qubits, int) or n_qubits not in schema.target_sizes:
        invalid.append(f"row {index}: n_qubits must be one of {schema.target_sizes}")
    baseline = row.get("baseline")
    if baseline not in schema.required_baselines:
        invalid.append(f"row {index}: baseline must be one of {schema.required_baselines}")
    status = row.get("status")
    if status not in schema.allowed_statuses:
        invalid.append(f"row {index}: status must be one of {schema.allowed_statuses}")
    if isinstance(n_qubits, int) and isinstance(baseline, str):
        key = (n_qubits, baseline)
        if key in seen:
            invalid.append(f"row {index}: duplicate row for n={n_qubits} baseline={baseline!r}")
        seen.add(key)


def _validate_row_metrics(index: int, row: Mapping[str, object], invalid: list[str]) -> None:
    status = row.get("status")
    if status == "ok":
        _require_nonnegative_number(row.get("wall_time_ms"), f"row {index}: wall_time_ms", invalid)
        memory = row.get("memory_bytes")
        if not isinstance(memory, int) or memory < 0:
            invalid.append(f"row {index}: memory_bytes must be a non-negative integer")
        _require_nonnegative_number(
            row.get("discarded_weight"), f"row {index}: discarded_weight", invalid
        )
        _require_nonnegative_number(
            row.get("entropy_proxy"), f"row {index}: entropy_proxy", invalid
        )
        max_bond = row.get("max_bond")
        if not isinstance(max_bond, int) or max_bond < 1:
            invalid.append(f"row {index}: max_bond must be a positive integer")
    if status in {"skipped", "failed"} and not row.get("notes"):
        invalid.append(f"row {index}: {status} row requires notes")
    omitted = row.get("omitted_coupling_mass")
    if (
        not isinstance(omitted, int | float)
        or not isfinite(float(omitted))
        or float(omitted) < 0.0
    ):
        invalid.append(f"row {index}: omitted_coupling_mass must be finite and non-negative")


def _validate_row_provenance(index: int, row: Mapping[str, object], invalid: list[str]) -> None:
    if not isinstance(row.get("truncation_policy"), str) or not row.get("truncation_policy"):
        invalid.append(f"row {index}: truncation_policy must be a non-empty string")
    command = row.get("command")
    command_sequence = (
        isinstance(command, Sequence)
        and not isinstance(command, str | bytes)
        and len(command) > 0
        and all(isinstance(item, str) and item for item in command)
    )
    if not (isinstance(command, str) and command) and not command_sequence:
        invalid.append(f"row {index}: command must be a non-empty string or sequence")
    for field in ("machine", "git_commit", "claim_boundary"):
        if not isinstance(row.get(field), str) or not row.get(field):
            invalid.append(f"row {index}: {field} must be a non-empty string")
    for field in ("dependencies", "host_load"):
        if not isinstance(row.get(field), Mapping):
            invalid.append(f"row {index}: {field} must be a mapping")
    if not isinstance(row.get("notes"), list):
        invalid.append(f"row {index}: notes must be a list")


def _require_nonnegative_number(value: object, label: str, invalid: list[str]) -> None:
    if not isinstance(value, int | float) or not isfinite(float(value)) or float(value) < 0.0:
        invalid.append(f"{label} must be finite and non-negative")


def _require_target_sizes(values: tuple[int, ...]) -> None:
    if not values or any(value <= 16 for value in values):
        raise ValueError("target_sizes must contain qubit counts greater than 16")
    if tuple(sorted(values)) != values:
        raise ValueError("target_sizes must be sorted")
    if len(set(values)) != len(values):
        raise ValueError("target_sizes must be unique")


def _require_text(value: str, name: str) -> None:
    if not value:
        raise ValueError(f"{name} must be non-empty")


def _require_nonempty_texts(values: tuple[str, ...], name: str) -> None:
    if not values or any(not value for value in values):
        raise ValueError(f"{name} must contain non-empty entries")


__all__ = [
    "TN_MPS_CROSSOVER_CLAIM_BOUNDARY",
    "TN_MPS_CROSSOVER_PROTOCOL_ID",
    "TN_MPS_CROSSOVER_REQUIRED_FIELDS",
    "TN_MPS_CROSSOVER_STAGE1_SCHEMA",
    "TNMPSCrossoverGate",
    "TNMPSCrossoverRowSchema",
    "TNMPSCrossoverRowValidation",
    "TNMPSCrossoverStage1Report",
    "build_tn_mps_crossover_stage1",
    "render_tn_mps_crossover_stage1_markdown",
    "validate_tn_mps_crossover_rows",
]
