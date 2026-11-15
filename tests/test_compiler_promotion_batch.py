# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — compiler promotion batch tests
# scpn-quantum-control -- compiler promotion-batch tests
"""Tests for compiler promotion-batch evidence assembly."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

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
    assert (
        "data/differentiable_phase_qnode/enzyme_mlir_compiler_ad_breadth_artifact_20260706.json"
    ) in paths
    assert all(len(entry.sha256) == 64 for entry in batch.evidence_files)
    assert all(entry.size_bytes > 0 for entry in batch.evidence_files)
    assert batch.boundary_artifact_id == "compiler-evidence-boundary-20260705"
    assert batch.alias_activity_artifact_id == "compiler-alias-activity-evidence-20260706"
    assert (
        batch.enzyme_mlir_breadth_artifact_id
        == "enzyme-mlir-compiler-ad-breadth-artifact-20260706"
    )
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
    batch = build_compiler_promotion_batch(source_commit="37a9318c")
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
            enzyme_mlir_breadth_artifact_id=batch.enzyme_mlir_breadth_artifact_id,
            promotion_ready=True,
        )


_DATA = Path(__file__).resolve().parents[1] / "data" / "differentiable_phase_qnode"
_LOADED = (
    "compiler_evidence_boundary_20260705.json",
    "compiler_alias_activity_evidence_20260706.json",
    "llvm_jit_claim_gate_20260704.json",
    "native_whole_program_ad_execution_evidence_20260622.json",
    "enzyme_mlir_maturity_audit_20260616.json",
    "enzyme_mlir_compiler_ad_breadth_artifact_20260706.json",
)


def _evidence_repo(tmp_path: Path, name: str, content: object) -> Path:
    """Copy the six loaded evidence payloads into a repository root and replace one of them."""
    target = tmp_path / "data" / "differentiable_phase_qnode"
    target.mkdir(parents=True)
    for loaded in _LOADED:
        (target / loaded).write_bytes((_DATA / loaded).read_bytes())
    (target / name).write_text(json.dumps(content), encoding="utf-8")
    return tmp_path


def test_promotion_batch_builder_refuses_malformed_loaded_evidence(tmp_path: Path) -> None:
    """The builder refuses a non-object payload, a payload without an ID, and promoted evidence."""
    alias = json.loads((_DATA / _LOADED[1]).read_text(encoding="utf-8"))
    cases: tuple[tuple[str, object, str], ...] = (
        (_LOADED[0], [], "compiler promotion evidence must be a JSON object"),
        (_LOADED[0], {"schema": "boundary"}, "compiler evidence boundary artifact_id must be"),
        (_LOADED[1], {**alias, "promotion_ready": True}, "alias-activity evidence cannot be"),
    )
    for index, (name, content, message) in enumerate(cases):
        root = _evidence_repo(tmp_path / f"case-{index}", name, content)
        with pytest.raises(ValueError, match=message):
            build_compiler_promotion_batch(repo_root=root, source_commit="test-commit")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"path": ""}, "path must be non-empty"),
        ({"role": ""}, "role must be non-empty"),
        ({"artifact_id": ""}, "artifact_id must be non-empty"),
        ({"sha256": "a" * 63}, "sha256 must be 64 hex characters"),
        ({"sha256": "z" * 64}, "sha256 must be 64 hex characters"),
        ({"size_bytes": 0}, "size_bytes must be positive"),
        ({"promotion_ready": "no"}, "promotion_ready must be boolean"),
    ],
)
def test_promotion_evidence_file_refuses_incomplete_checksums(
    changes: dict[str, Any], message: str
) -> None:
    """An evidence-file summary needs its path, role, ID, a hex SHA-256, a size and a boolean."""
    entry = build_compiler_promotion_batch(source_commit="test-commit").evidence_files[0]
    with pytest.raises(ValueError, match=message):
        replace(entry, **changes)


def test_promotion_batch_refuses_drift_order_promoted_entries_and_gaps() -> None:
    """A batch refuses drifted identity, unordered, repeated or promoted entries, and gaps."""
    batch = build_compiler_promotion_batch(source_commit="test-commit")
    files = batch.evidence_files
    promoted = (replace(files[0], promotion_ready=True), *files[1:])
    without_boundary = tuple(
        entry for entry in files if entry.artifact_id != batch.boundary_artifact_id
    )
    cases: list[tuple[dict[str, Any], str]] = [
        ({"artifact_id": "other"}, "artifact_id drifted"),
        ({"schema": "other"}, "schema drifted"),
        ({"classification": "isolated"}, "classification drifted"),
        ({"status": "promoted"}, "status drifted"),
        ({"missing_requirements": ()}, "missing requirements drifted"),
        ({"promotion_blockers": ()}, "blockers drifted"),
        ({"source_commit": ""}, "source_commit must be non-empty"),
        ({"evidence_files": ()}, "evidence_files must be non-empty"),
        ({"evidence_files": tuple(reversed(files))}, "must be sorted by path"),
        ({"evidence_files": (files[0], files[0], *files[1:])}, "paths must be unique"),
        ({"evidence_files": promoted}, "cannot include promoted evidence"),
        ({"evidence_files": without_boundary}, "missing evidence ids"),
        ({"claim_boundary": "bounded"}, "claim boundary is incomplete"),
    ]
    for changes, message in cases:
        with pytest.raises(ValueError, match=message):
            replace(batch, **changes)
