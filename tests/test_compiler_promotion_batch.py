# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- compiler promotion-batch tests
"""Tests for compiler promotion-batch evidence assembly."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.compiler import (
    CompilerPromotionBatch,
    CompilerPromotionBatchEvidenceFile,
    build_compiler_promotion_batch,
    render_compiler_promotion_batch_markdown,
)


def test_compiler_promotion_batch_assembles_existing_evidence_files() -> None:
    """Promotion batch assembly must checksum real committed evidence files."""
    batch = build_compiler_promotion_batch(source_commit="test-commit")

    payload = batch.as_dict()
    paths = {entry.path for entry in batch.evidence_files}

    assert payload["schema"] == "scpn_qc_compiler_promotion_batch_v1"
    assert payload["artifact_id"] == "compiler-promotion-batch-20260706"
    assert payload["source_commit"] == "test-commit"
    assert payload["classification"] == "functional_non_isolated"
    assert payload["promotion_ready"] is False
    assert payload["status"] == "blocked_missing_isolated_compiler_benchmark_ids"
    assert payload["missing_requirements"] == ["isolated compiler benchmark artifact IDs"]
    assert payload["promotion_blockers"] == ["isolated compiler benchmark artifact IDs missing"]
    assert payload["assembled_evidence_count"] == len(batch.evidence_files)
    assert "data/differentiable_phase_qnode/compiler_evidence_boundary_20260705.json" in paths
    assert (
        "data/differentiable_phase_qnode/compiler_alias_activity_evidence_20260706.json" in paths
    )
    assert "data/differentiable_phase_qnode/llvm_jit_claim_gate_20260704.json" in paths
    assert (
        "data/differentiable_phase_qnode/native_whole_program_ad_execution_evidence_20260622.json"
        in paths
    )
    assert "data/differentiable_phase_qnode/enzyme_mlir_maturity_audit_20260616.json" in paths
    assert all(len(entry.sha256) == 64 for entry in batch.evidence_files)
    assert all(entry.size_bytes > 0 for entry in batch.evidence_files)
    assert batch.boundary_artifact_id == "compiler-evidence-boundary-20260705"
    assert batch.alias_activity_artifact_id == "compiler-alias-activity-evidence-20260706"
    assert "isolated benchmarks" in batch.claim_boundary
    assert "provider, hardware, GPU, or performance claim" in batch.claim_boundary


def test_compiler_promotion_batch_markdown_preserves_blocked_status() -> None:
    """Promotion batch Markdown must expose the remaining non-promotional gate."""
    batch = build_compiler_promotion_batch(source_commit="test-commit")

    markdown = render_compiler_promotion_batch_markdown(batch)

    assert "# Compiler Promotion Batch" in markdown
    assert "promotion_ready: `False`" in markdown
    assert "`blocked_missing_isolated_compiler_benchmark_ids`" in markdown
    assert "isolated compiler benchmark artifact IDs missing" in markdown
    assert "Claim boundary:" in markdown


def test_committed_compiler_promotion_batch_matches_builder() -> None:
    """Committed compiler promotion-batch JSON must match current evidence files."""
    batch = build_compiler_promotion_batch(source_commit="8df68be7")
    committed = json.loads(
        Path("data/differentiable_phase_qnode/compiler_promotion_batch_20260706.json").read_text(
            encoding="utf-8"
        )
    )

    assert committed == batch.as_dict()


def test_compiler_promotion_batch_rejects_missing_evidence_file(tmp_path: Path) -> None:
    """Promotion batch assembly must fail closed when required evidence is missing."""
    evidence_path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError, match="compiler promotion evidence file is missing"):
        CompilerPromotionBatchEvidenceFile.from_path(
            evidence_path,
            repo_root=tmp_path,
            role="missing evidence",
            artifact_id="missing",
            promotion_ready=False,
        )


def test_compiler_promotion_batch_rejects_promotion_ready_payload() -> None:
    """Promotion batch validation must reject promotion-ready drift."""
    batch = build_compiler_promotion_batch(source_commit="test-commit")

    with pytest.raises(ValueError, match="cannot be promotion-ready"):
        CompilerPromotionBatch(
            source_commit=batch.source_commit,
            evidence_files=batch.evidence_files,
            boundary_artifact_id=batch.boundary_artifact_id,
            alias_activity_artifact_id=batch.alias_activity_artifact_id,
            llvm_jit_claim_gate_artifact_id=batch.llvm_jit_claim_gate_artifact_id,
            native_whole_program_artifact_id=batch.native_whole_program_artifact_id,
            enzyme_mlir_maturity_artifact_id=batch.enzyme_mlir_maturity_artifact_id,
            promotion_ready=True,
        )
