# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- compiler promotion batch
"""Non-promotional compiler evidence batch assembly."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPILER_PROMOTION_BATCH_SCHEMA = "scpn_qc_compiler_promotion_batch_v1"
COMPILER_PROMOTION_BATCH_ID = "compiler-promotion-batch-20260706"
COMPILER_PROMOTION_BATCH_CLASSIFICATION = "functional_non_isolated"
COMPILER_PROMOTION_BATCH_STATUS = "blocked_missing_isolated_compiler_benchmark_ids"
COMPILER_PROMOTION_BATCH_MISSING_REQUIREMENTS = ("isolated compiler benchmark artifact IDs",)
COMPILER_PROMOTION_BATCH_BLOCKERS = ("isolated compiler benchmark artifact IDs missing",)
COMPILER_PROMOTION_BATCH_CLAIM_BOUNDARY = (
    "Compiler promotion batch assembly only: committed compiler-boundary, "
    "alias-activity, native LLVM/JIT, native whole-program AD, and Enzyme/MLIR "
    "maturity plus raw breadth evidence are checksummed for reviewer triage; this does not "
    "promote general compiler AD, isolated benchmarks, provider, hardware, GPU, "
    "or performance claim."
)
_DEFAULT_EVIDENCE_INPUTS: tuple[tuple[str, str], ...] = (
    (
        "data/differentiable_phase_qnode/compiler_evidence_boundary_20260705.json",
        "Compiler evidence boundary and remaining promotion gates",
    ),
    (
        "data/differentiable_phase_qnode/compiler_evidence_boundary_20260705.md",
        "Compiler evidence boundary reviewer summary",
    ),
    (
        "data/differentiable_phase_qnode/compiler_alias_activity_evidence_20260706.json",
        "Program AD compiler alias-activity evidence",
    ),
    (
        "data/differentiable_phase_qnode/compiler_alias_activity_evidence_20260706.md",
        "Program AD compiler alias-activity reviewer summary",
    ),
    (
        "data/differentiable_phase_qnode/llvm_jit_claim_gate_20260704.json",
        "Native LLVM/JIT promotion claim gate",
    ),
    (
        "data/differentiable_phase_qnode/llvm_jit_claim_gate_20260704.md",
        "Native LLVM/JIT promotion claim-gate reviewer summary",
    ),
    (
        "data/differentiable_phase_qnode/native_whole_program_ad_execution_evidence_20260622.json",
        "Native LLVM/JIT whole-program AD execution evidence",
    ),
    (
        "data/differentiable_phase_qnode/native_whole_program_ad_execution_evidence_20260622.md",
        "Native LLVM/JIT whole-program AD execution reviewer summary",
    ),
    (
        "data/differentiable_phase_qnode/enzyme_mlir_maturity_audit_20260616.json",
        "Enzyme/MLIR maturity and hard-gap evidence",
    ),
    (
        "data/differentiable_phase_qnode/enzyme_mlir_maturity_audit_20260616.md",
        "Enzyme/MLIR maturity reviewer summary",
    ),
    (
        "data/differentiable_phase_qnode/enzyme_mlir_compiler_ad_breadth_artifact_20260706.json",
        "Enzyme/MLIR raw 11-case compiler-AD breadth artifact",
    ),
    (
        "data/differentiable_phase_qnode/enzyme_mlir_compiler_ad_breadth_artifact_20260706.md",
        "Enzyme/MLIR raw 11-case compiler-AD breadth reviewer summary",
    ),
)


@dataclass(frozen=True)
class CompilerPromotionBatchEvidenceFile:
    """Checksum summary for one compiler promotion-batch evidence file."""

    path: str
    role: str
    artifact_id: str
    sha256: str
    size_bytes: int
    promotion_ready: bool

    def __post_init__(self) -> None:
        """Validate evidence-file metadata."""

        if not self.path:
            raise ValueError("compiler promotion evidence path must be non-empty")
        if not self.role:
            raise ValueError("compiler promotion evidence role must be non-empty")
        if not self.artifact_id:
            raise ValueError("compiler promotion evidence artifact_id must be non-empty")
        if len(self.sha256) != 64:
            raise ValueError("compiler promotion evidence sha256 must be 64 hex characters")
        try:
            int(self.sha256, 16)
        except ValueError as exc:
            raise ValueError(
                "compiler promotion evidence sha256 must be 64 hex characters"
            ) from exc
        if self.size_bytes <= 0:
            raise ValueError("compiler promotion evidence size_bytes must be positive")
        if not isinstance(self.promotion_ready, bool):
            raise ValueError("compiler promotion evidence promotion_ready must be boolean")

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        repo_root: Path = REPO_ROOT,
        role: str,
        artifact_id: str,
        promotion_ready: bool,
    ) -> CompilerPromotionBatchEvidenceFile:
        """Build an evidence-file checksum summary from a repository path."""

        resolved = path if path.is_absolute() else repo_root / path
        if not resolved.exists():
            raise FileNotFoundError(f"compiler promotion evidence file is missing: {path}")
        data = resolved.read_bytes()
        return cls(
            path=resolved.relative_to(repo_root).as_posix(),
            role=role,
            artifact_id=artifact_id,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            promotion_ready=promotion_ready,
        )

    def as_dict(self) -> dict[str, object]:
        """Return a stable JSON-ready evidence-file summary."""

        return {
            "path": self.path,
            "role": self.role,
            "artifact_id": self.artifact_id,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "promotion_ready": self.promotion_ready,
        }


@dataclass(frozen=True)
class CompilerPromotionBatch:
    """Non-promotional compiler evidence batch for reviewer triage."""

    source_commit: str
    evidence_files: tuple[CompilerPromotionBatchEvidenceFile, ...]
    boundary_artifact_id: str
    alias_activity_artifact_id: str
    llvm_jit_claim_gate_artifact_id: str
    native_whole_program_artifact_id: str
    enzyme_mlir_maturity_artifact_id: str
    enzyme_mlir_breadth_artifact_id: str
    artifact_id: str = COMPILER_PROMOTION_BATCH_ID
    schema: str = COMPILER_PROMOTION_BATCH_SCHEMA
    classification: str = COMPILER_PROMOTION_BATCH_CLASSIFICATION
    status: str = COMPILER_PROMOTION_BATCH_STATUS
    missing_requirements: tuple[str, ...] = COMPILER_PROMOTION_BATCH_MISSING_REQUIREMENTS
    promotion_blockers: tuple[str, ...] = COMPILER_PROMOTION_BATCH_BLOCKERS
    promotion_ready: bool = False
    claim_boundary: str = COMPILER_PROMOTION_BATCH_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate compiler promotion-batch metadata."""

        if self.artifact_id != COMPILER_PROMOTION_BATCH_ID:
            raise ValueError("compiler promotion batch artifact_id drifted")
        if self.schema != COMPILER_PROMOTION_BATCH_SCHEMA:
            raise ValueError("compiler promotion batch schema drifted")
        if self.classification != COMPILER_PROMOTION_BATCH_CLASSIFICATION:
            raise ValueError("compiler promotion batch classification drifted")
        if self.status != COMPILER_PROMOTION_BATCH_STATUS:
            raise ValueError("compiler promotion batch status drifted")
        if self.missing_requirements != COMPILER_PROMOTION_BATCH_MISSING_REQUIREMENTS:
            raise ValueError("compiler promotion batch missing requirements drifted")
        if self.promotion_blockers != COMPILER_PROMOTION_BATCH_BLOCKERS:
            raise ValueError("compiler promotion batch blockers drifted")
        if self.promotion_ready:
            raise ValueError("compiler promotion batch cannot be promotion-ready")
        if not self.source_commit:
            raise ValueError("compiler promotion batch source_commit must be non-empty")
        if not self.evidence_files:
            raise ValueError("compiler promotion batch evidence_files must be non-empty")
        if tuple(sorted(self.evidence_files, key=lambda entry: entry.path)) != self.evidence_files:
            raise ValueError("compiler promotion batch evidence files must be sorted by path")
        paths = tuple(entry.path for entry in self.evidence_files)
        if tuple(sorted(set(paths))) != paths:
            raise ValueError("compiler promotion batch evidence file paths must be unique")
        for entry in self.evidence_files:
            if entry.promotion_ready:
                raise ValueError("compiler promotion batch cannot include promoted evidence")
        expected_ids = {
            self.boundary_artifact_id,
            self.alias_activity_artifact_id,
            self.llvm_jit_claim_gate_artifact_id,
            self.native_whole_program_artifact_id,
            self.enzyme_mlir_maturity_artifact_id,
            self.enzyme_mlir_breadth_artifact_id,
        }
        observed_ids = {entry.artifact_id for entry in self.evidence_files}
        missing_ids = expected_ids.difference(observed_ids)
        if missing_ids:
            missing = ", ".join(sorted(missing_ids))
            raise ValueError(f"compiler promotion batch missing evidence ids: {missing}")
        for phrase in (
            "general compiler AD",
            "isolated benchmarks",
            "provider, hardware, GPU, or performance claim",
        ):
            if phrase not in self.claim_boundary:
                raise ValueError("compiler promotion batch claim boundary is incomplete")

    @property
    def assembled_evidence_count(self) -> int:
        """Return the number of evidence files assembled into the batch."""

        return len(self.evidence_files)

    def as_dict(self) -> dict[str, object]:
        """Return a stable JSON-ready compiler promotion-batch payload."""

        return {
            "artifact_id": self.artifact_id,
            "schema": self.schema,
            "source_commit": self.source_commit,
            "classification": self.classification,
            "status": self.status,
            "promotion_ready": self.promotion_ready,
            "assembled_evidence_count": self.assembled_evidence_count,
            "missing_requirements": list(self.missing_requirements),
            "promotion_blockers": list(self.promotion_blockers),
            "boundary_artifact_id": self.boundary_artifact_id,
            "alias_activity_artifact_id": self.alias_activity_artifact_id,
            "llvm_jit_claim_gate_artifact_id": self.llvm_jit_claim_gate_artifact_id,
            "native_whole_program_artifact_id": self.native_whole_program_artifact_id,
            "enzyme_mlir_maturity_artifact_id": self.enzyme_mlir_maturity_artifact_id,
            "enzyme_mlir_breadth_artifact_id": self.enzyme_mlir_breadth_artifact_id,
            "evidence_files": [entry.as_dict() for entry in self.evidence_files],
            "claim_boundary": self.claim_boundary,
        }


def build_compiler_promotion_batch(
    *,
    repo_root: Path = REPO_ROOT,
    source_commit: str,
) -> CompilerPromotionBatch:
    """Assemble the current non-promotional compiler evidence batch."""

    boundary_payload = _load_json_mapping(
        repo_root / "data/differentiable_phase_qnode/compiler_evidence_boundary_20260705.json"
    )
    alias_payload = _load_json_mapping(
        repo_root
        / "data/differentiable_phase_qnode/compiler_alias_activity_evidence_20260706.json"
    )
    llvm_payload = _load_json_mapping(
        repo_root / "data/differentiable_phase_qnode/llvm_jit_claim_gate_20260704.json"
    )
    native_payload = _load_json_mapping(
        repo_root
        / "data/differentiable_phase_qnode/native_whole_program_ad_execution_evidence_20260622.json"
    )
    enzyme_payload = _load_json_mapping(
        repo_root / "data/differentiable_phase_qnode/enzyme_mlir_maturity_audit_20260616.json"
    )
    enzyme_breadth_payload = _load_json_mapping(
        repo_root / "data/differentiable_phase_qnode/"
        "enzyme_mlir_compiler_ad_breadth_artifact_20260706.json"
    )
    boundary_id = _required_artifact_id(boundary_payload, "compiler evidence boundary")
    alias_id = _required_artifact_id(alias_payload, "compiler alias-activity evidence")
    llvm_id = _required_artifact_id(llvm_payload, "LLVM/JIT claim gate")
    native_id = _required_artifact_id(native_payload, "native whole-program AD evidence")
    enzyme_id = _required_artifact_id(enzyme_payload, "Enzyme/MLIR maturity evidence")
    enzyme_breadth_id = _required_artifact_id(
        enzyme_breadth_payload,
        "Enzyme/MLIR raw breadth artifact",
    )
    _assert_not_promotion_ready(boundary_payload, "compiler evidence boundary")
    _assert_not_promotion_ready(alias_payload, "compiler alias-activity evidence")
    _assert_not_promotion_ready(llvm_payload, "LLVM/JIT claim gate")
    _assert_not_promotion_ready(enzyme_payload, "Enzyme/MLIR maturity evidence")

    artifact_ids = {
        "data/differentiable_phase_qnode/compiler_evidence_boundary_20260705.json": boundary_id,
        "data/differentiable_phase_qnode/compiler_evidence_boundary_20260705.md": boundary_id,
        "data/differentiable_phase_qnode/compiler_alias_activity_evidence_20260706.json": alias_id,
        "data/differentiable_phase_qnode/compiler_alias_activity_evidence_20260706.md": alias_id,
        "data/differentiable_phase_qnode/llvm_jit_claim_gate_20260704.json": llvm_id,
        "data/differentiable_phase_qnode/llvm_jit_claim_gate_20260704.md": llvm_id,
        (
            "data/differentiable_phase_qnode/"
            "native_whole_program_ad_execution_evidence_20260622.json"
        ): native_id,
        (
            "data/differentiable_phase_qnode/"
            "native_whole_program_ad_execution_evidence_20260622.md"
        ): native_id,
        "data/differentiable_phase_qnode/enzyme_mlir_maturity_audit_20260616.json": enzyme_id,
        "data/differentiable_phase_qnode/enzyme_mlir_maturity_audit_20260616.md": enzyme_id,
        (
            "data/differentiable_phase_qnode/"
            "enzyme_mlir_compiler_ad_breadth_artifact_20260706.json"
        ): enzyme_breadth_id,
        (
            "data/differentiable_phase_qnode/enzyme_mlir_compiler_ad_breadth_artifact_20260706.md"
        ): enzyme_breadth_id,
    }
    evidence_files = tuple(
        sorted(
            (
                CompilerPromotionBatchEvidenceFile.from_path(
                    repo_root / evidence_path,
                    repo_root=repo_root,
                    role=role,
                    artifact_id=artifact_ids[evidence_path],
                    promotion_ready=False,
                )
                for evidence_path, role in _DEFAULT_EVIDENCE_INPUTS
            ),
            key=lambda entry: entry.path,
        )
    )
    return CompilerPromotionBatch(
        source_commit=source_commit,
        evidence_files=evidence_files,
        boundary_artifact_id=boundary_id,
        alias_activity_artifact_id=alias_id,
        llvm_jit_claim_gate_artifact_id=llvm_id,
        native_whole_program_artifact_id=native_id,
        enzyme_mlir_maturity_artifact_id=enzyme_id,
        enzyme_mlir_breadth_artifact_id=enzyme_breadth_id,
    )


def render_compiler_promotion_batch_markdown(batch: CompilerPromotionBatch) -> str:
    """Render a reviewer-facing Markdown summary of the compiler batch."""

    lines = [
        "# Compiler Promotion Batch",
        "",
        f"- artifact_id: `{batch.artifact_id}`",
        f"- source_commit: `{batch.source_commit}`",
        f"- classification: `{batch.classification}`",
        f"- status: `{batch.status}`",
        f"- promotion_ready: `{batch.promotion_ready}`",
        f"- assembled evidence files: `{batch.assembled_evidence_count}`",
        "",
        "Missing requirements:",
        "",
        *[f"- `{requirement}`" for requirement in batch.missing_requirements],
        "",
        "Promotion blockers:",
        "",
        *[f"- {blocker}" for blocker in batch.promotion_blockers],
        "",
        "| Evidence file | Role | Artifact ID | SHA-256 |",
        "|---|---|---|---|",
    ]
    for entry in batch.evidence_files:
        lines.append(
            f"| `{entry.path}` | {entry.role} | `{entry.artifact_id}` | `{entry.sha256}` |"
        )
    lines.extend(
        [
            "",
            f"Claim boundary: {batch.claim_boundary}",
            "",
        ]
    )
    return "\n".join(lines)


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"compiler promotion evidence must be a JSON object: {path}")
    return payload


def _required_artifact_id(payload: Mapping[str, Any], label: str) -> str:
    artifact_id = payload.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise ValueError(f"{label} artifact_id must be non-empty")
    return artifact_id


def _assert_not_promotion_ready(payload: Mapping[str, Any], label: str) -> None:
    promotion_ready = payload.get("promotion_ready", False)
    if promotion_ready is True:
        raise ValueError(f"{label} cannot be promotion-ready")
