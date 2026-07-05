# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable external-validation tests
"""Tests for differentiable external-validation package manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.differentiable_external_validation import (
    EXTERNAL_VALIDATION_ENVIRONMENT_LOCK_SCHEMA,
    EnvironmentLockfileSummary,
    ExternalValidationArtifactBundle,
    ExternalValidationArtifactEntry,
    ExternalValidationEnvironmentLock,
    build_external_validation_artifact_bundle,
    build_external_validation_environment_lock,
    load_external_validation_artifact_bundle,
    load_external_validation_environment_lock,
    render_external_validation_artifact_bundle_markdown,
    render_external_validation_environment_lock_markdown,
    summarize_artifact_entry,
    summarize_environment_lockfile,
    validate_external_validation_artifact_bundle,
    validate_external_validation_environment_lock,
)


def test_build_external_validation_environment_lock_records_exact_lockfiles() -> None:
    """Environment lock manifests must record every committed lockfile."""
    manifest = build_external_validation_environment_lock()

    payload = manifest.to_dict()
    paths = {lockfile.path for lockfile in manifest.lockfiles}
    assert payload["artifact_id"] == manifest.artifact_id
    assert payload["lockfiles"][0] == manifest.lockfiles[0].to_dict()
    assert manifest.schema == EXTERNAL_VALIDATION_ENVIRONMENT_LOCK_SCHEMA
    assert manifest.classification == "functional_non_isolated"
    assert "isolated_affinity benchmark claims" in manifest.claim_boundary
    assert "pyproject.toml" in paths
    assert "requirements-ci-py311-linux.txt" in paths
    assert "requirements-ci-py313-linux.txt" in paths
    assert (
        "data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/"
        "framework_overlay_freeze.txt"
    ) in paths
    assert all(len(lockfile.sha256) == 64 for lockfile in manifest.lockfiles)
    assert all(lockfile.size_bytes > 0 for lockfile in manifest.lockfiles)


def test_committed_external_validation_environment_lock_matches_files() -> None:
    """Committed environment lock manifests must match current files."""
    manifest = load_external_validation_environment_lock()
    validation = validate_external_validation_environment_lock(manifest)

    assert validation.passed
    assert not validation.errors
    assert "pyproject.toml" in validation.checked_paths
    assert "requirements-ci-py312-linux.txt" in validation.checked_paths


def test_environment_lock_validation_rejects_hash_drift() -> None:
    """Environment lock validation must reject checksum drift."""
    manifest = build_external_validation_environment_lock()
    first = manifest.lockfiles[0]
    drifted = EnvironmentLockfileSummary(
        path=first.path,
        role=first.role,
        sha256="0" * 64,
        size_bytes=first.size_bytes,
        line_count=first.line_count,
        pinned_package_count=first.pinned_package_count,
    )
    candidate = ExternalValidationEnvironmentLock(
        artifact_id=manifest.artifact_id,
        schema=manifest.schema,
        python_version=manifest.python_version,
        platform=manifest.platform,
        lockfiles=(drifted, *manifest.lockfiles[1:]),
        classification=manifest.classification,
        claim_boundary=manifest.claim_boundary,
    )

    validation = validate_external_validation_environment_lock(candidate)

    assert not validation.passed
    assert validation.errors == ("sha256 mismatch: pyproject.toml",)


def test_environment_lock_validation_rejects_contract_drift(tmp_path: Path) -> None:
    """Environment lock validation must reject schema and metadata drift."""
    lockfile = tmp_path / "requirements.txt"
    lockfile.write_text("numpy==2.0.0\nscipy==1.12.0\n", encoding="utf-8")
    summary = summarize_environment_lockfile(lockfile, repo_root=tmp_path, role="runtime")
    drifted = EnvironmentLockfileSummary(
        path=summary.path,
        role=summary.role,
        sha256=summary.sha256,
        size_bytes=summary.size_bytes + 1,
        line_count=summary.line_count + 1,
        pinned_package_count=summary.pinned_package_count + 1,
    )
    missing = EnvironmentLockfileSummary(
        path="missing.txt",
        role="missing",
        sha256="0" * 64,
        size_bytes=0,
        line_count=0,
        pinned_package_count=0,
    )
    candidate = ExternalValidationEnvironmentLock(
        artifact_id="candidate",
        schema="wrong",
        python_version="3.12.0",
        platform="test",
        lockfiles=(drifted, missing),
        classification="promoted",
        claim_boundary="too vague",
    )

    validation = validate_external_validation_environment_lock(
        candidate,
        repo_root=tmp_path,
    )

    assert validation.to_dict()["passed"] is False
    assert validation.errors == (
        "unexpected schema: wrong",
        "environment lock manifest must remain functional_non_isolated",
        "environment lock manifest claim boundary is not explicit enough",
        "size mismatch: requirements.txt",
        "line-count mismatch: requirements.txt",
        "pinned-package-count mismatch: requirements.txt",
        "missing lockfile: missing.txt",
    )


def test_summarize_environment_lockfile_counts_pinned_packages(tmp_path: Path) -> None:
    """Lockfile summaries must count pinned requirements and file metadata."""
    lockfile = tmp_path / "requirements.txt"
    lockfile.write_text(
        "numpy==2.0.0\nscipy==1.12.0 ; python_version >= '3.10'\n# comment\njax[cpu]==0.4.30\n",
        encoding="utf-8",
    )

    summary = summarize_environment_lockfile(
        lockfile,
        repo_root=tmp_path,
        role="test lockfile",
    )

    assert summary.path == "requirements.txt"
    assert summary.pinned_package_count == 3
    assert summary.line_count == 4
    assert len(summary.sha256) == 64


def test_environment_lockfile_summary_rejects_missing_file(tmp_path: Path) -> None:
    """Environment lockfile summaries must fail closed on missing files."""
    missing = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError) as exc_info:
        summarize_environment_lockfile(missing, repo_root=tmp_path, role="missing")

    assert str(exc_info.value) == f"environment lockfile is missing: {missing}"


def test_environment_lock_markdown_lists_claim_boundary() -> None:
    """Environment lock Markdown must expose the claim boundary."""
    markdown = render_external_validation_environment_lock_markdown(
        build_external_validation_environment_lock()
    )

    assert "# Differentiable External-Validation Environment Lock" in markdown
    assert "functional_non_isolated" in markdown
    assert "pyproject.toml" in markdown
    assert "isolated_affinity benchmark claims" in markdown


def test_build_external_validation_artifact_bundle_records_committed_evidence() -> None:
    """Artefact bundle manifests must record committed evidence files."""
    bundle = build_external_validation_artifact_bundle()

    payload = bundle.to_dict()
    paths = {entry.path for entry in bundle.entries}
    assert payload["artifact_id"] == bundle.artifact_id
    assert payload["entries"][0] == bundle.entries[0].to_dict()
    assert bundle.classification == "functional_non_isolated"
    assert "isolated_affinity benchmark claims" in bundle.claim_boundary
    assert "data/differentiable_phase_qnode/claim_ledger.json" in paths
    assert "data/differentiable_phase_qnode/public_claim_table_20260616.md" in paths
    assert (
        "data/differentiable_phase_qnode/differentiable_competitive_baseline_refresh_20260627.json"
        in paths
    )
    assert (
        "data/differentiable_phase_qnode/differentiable_competitive_baseline_refresh_20260627.md"
        in paths
    )
    assert (
        "data/differentiable_phase_qnode/differentiable_support_surface_alignment_20260627.json"
        in paths
    )
    assert "data/differentiable_phase_qnode/differentiable_architecture_map_20260627.json" in paths
    assert (
        "data/differentiable_phase_qnode/differentiable_dependency_environment_map_20260627.json"
        in paths
    )
    assert (
        "data/differentiable_phase_qnode/differentiable_isolated_benchmark_plan_20260627.json"
        in paths
    )
    assert "data/differentiable_phase_qnode/provider_gradient_boundary_20260705.json" in paths
    assert "data/differentiable_phase_qnode/compiler_evidence_boundary_20260705.json" in paths
    assert (
        "data/differentiable_phase_qnode/native_whole_program_ad_execution_evidence_20260622.json"
        in paths
    )
    assert "data/differentiable_phase_qnode/llvm_jit_claim_gate_20260704.json" in paths
    assert (
        "data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/"
        "diff-qnode-external-comparison.json"
    ) in paths
    assert all(len(entry.sha256) == 64 for entry in bundle.entries)
    assert all(entry.size_bytes > 0 for entry in bundle.entries)


def test_committed_external_validation_artifact_bundle_matches_files() -> None:
    """Committed artefact bundle manifests must match current evidence files."""
    bundle = load_external_validation_artifact_bundle()
    validation = validate_external_validation_artifact_bundle(bundle)

    assert validation.passed
    assert not validation.errors
    assert "data/differentiable_phase_qnode/claim_ledger.json" in validation.checked_paths
    assert (
        "data/differentiable_phase_qnode/differentiable_architecture_map_20260627.md"
        in validation.checked_paths
    )
    assert (
        "data/differentiable_phase_qnode/differentiable_dependency_environment_map_20260627.md"
        in validation.checked_paths
    )
    assert (
        "data/differentiable_phase_qnode/differentiable_isolated_benchmark_plan_20260627.md"
        in validation.checked_paths
    )
    assert "data/differentiable_phase_qnode/provider_gradient_boundary_20260705.md" in (
        validation.checked_paths
    )
    assert "data/differentiable_phase_qnode/compiler_evidence_boundary_20260705.md" in (
        validation.checked_paths
    )
    assert "data/differentiable_phase_qnode/llvm_jit_claim_gate_20260704.md" in (
        validation.checked_paths
    )


def test_provider_gradient_boundary_artifact_preserves_no_submit_boundary() -> None:
    """Provider boundary evidence must stay no-submit and non-promotional."""
    payload = json.loads(
        Path("data/differentiable_phase_qnode/provider_gradient_boundary_20260705.json").read_text(
            encoding="utf-8"
        )
    )
    surfaces = {
        str(surface["name"]): surface
        for surface in payload["surfaces"]
        if isinstance(surface, dict)
    }

    assert payload["schema"] == "scpn_qc_differentiable_provider_gradient_boundary_v1"
    assert payload["classification"] == "functional_no_submit"
    assert payload["no_submit"] is True
    assert payload["promotion_ready"] is False
    assert payload["hardware_execution_count"] == 0
    assert payload["gradient_available_count"] == 0
    assert payload["ready_for_hardware_gradient_promotion"] is False
    assert "live execution ticket missing" in payload["promotion_blockers"]
    assert surfaces["provider_gradient_readiness"]["supported_count"] == 3
    assert surfaces["provider_gradient_readiness"]["blocked_count"] == 3
    assert surfaces["hardware_gradient_policy_readiness"]["supported_count"] == 1
    assert surfaces["hardware_gradient_policy_readiness"]["blocked_count"] == 5
    assert surfaces["provider_hardware_gradient_preparation"]["supported_count"] == 2
    assert surfaces["provider_hardware_gradient_preparation"]["blocked_count"] == 4


def test_compiler_evidence_boundary_artifact_preserves_promotion_gate() -> None:
    """Compiler boundary evidence must stay non-promotional until every gate passes."""
    payload = json.loads(
        Path("data/differentiable_phase_qnode/compiler_evidence_boundary_20260705.json").read_text(
            encoding="utf-8"
        )
    )
    required = {str(row["name"]): row for row in payload["required_evidence"]}

    assert payload["schema"] == "scpn_qc_differentiable_compiler_evidence_boundary_v1"
    assert payload["classification"] == "functional_non_isolated"
    assert payload["promotion_ready"] is False
    assert payload["native_llvm_jit"]["focused_test_exit_code"] == 0
    assert payload["native_llvm_jit"]["promotion_ready"] is False
    assert payload["native_llvm_jit"]["executable_lowering_verified"] is True
    assert "benchmark_artifact_ids" in payload["native_llvm_jit"]["missing_requirements"]
    assert payload["enzyme_mlir"]["artifact_id"] == "enzyme-toolchain-ad-execution-20260705"
    assert payload["enzyme_mlir"]["promotion_ready"] is False
    assert "scalar_forward_mode" in required
    assert "native_enzyme_execution" in required
    assert required["native_enzyme_execution"]["status"] in {"evidence_attached", "hard_gap"}
    assert required["alias_activity"]["status"] == "blocked"
    assert "isolated compiler benchmark artifact IDs missing" in payload["promotion_blockers"]
    assert "provider, hardware, GPU, or performance claim" in payload["claim_boundary"]


def test_artifact_bundle_validation_rejects_hash_drift() -> None:
    """Artefact bundle validation must reject checksum drift."""
    bundle = build_external_validation_artifact_bundle()
    first = bundle.entries[0]
    drifted = ExternalValidationArtifactEntry(
        path=first.path,
        role=first.role,
        sha256="0" * 64,
        size_bytes=first.size_bytes,
    )
    candidate = ExternalValidationArtifactBundle(
        artifact_id=bundle.artifact_id,
        schema=bundle.schema,
        entries=(drifted, *bundle.entries[1:]),
        classification=bundle.classification,
        claim_boundary=bundle.claim_boundary,
    )

    validation = validate_external_validation_artifact_bundle(candidate)

    assert not validation.passed
    assert validation.errors == (
        "sha256 mismatch: data/differentiable_phase_qnode/claim_ledger.json",
    )


def test_artifact_bundle_validation_rejects_contract_drift(tmp_path: Path) -> None:
    """Artefact bundle validation must reject schema and metadata drift."""
    artefact = tmp_path / "evidence.json"
    artefact.write_text('{"ok": true}\n', encoding="utf-8")
    entry = summarize_artifact_entry(artefact, repo_root=tmp_path, role="evidence")
    summary = ExternalValidationArtifactEntry(
        path=entry.path,
        role=entry.role,
        sha256=entry.sha256,
        size_bytes=entry.size_bytes + 1,
    )
    missing = ExternalValidationArtifactEntry(
        path="missing.json",
        role="missing",
        sha256="0" * 64,
        size_bytes=0,
    )
    bundle = ExternalValidationArtifactBundle(
        artifact_id="candidate",
        schema="wrong",
        entries=(summary, missing),
        classification="promoted",
        claim_boundary="too vague",
    )

    validation = validate_external_validation_artifact_bundle(
        bundle,
        repo_root=tmp_path,
    )

    assert validation.errors == (
        "unexpected schema: wrong",
        "artifact bundle must remain functional_non_isolated",
        "artifact bundle claim boundary is not explicit enough",
        "size mismatch: evidence.json",
        "missing artefact: missing.json",
    )


def test_artifact_entry_summary_rejects_missing_file(tmp_path: Path) -> None:
    """Artefact entry summaries must fail closed on missing files."""
    missing = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError) as exc_info:
        summarize_artifact_entry(missing, repo_root=tmp_path, role="missing")

    assert str(exc_info.value) == f"external-validation artefact is missing: {missing}"


def test_artifact_bundle_markdown_lists_claim_boundary() -> None:
    """Artefact bundle Markdown must expose the claim boundary."""
    markdown = render_external_validation_artifact_bundle_markdown(
        build_external_validation_artifact_bundle()
    )

    assert "# Differentiable External-Validation Artefact Bundle" in markdown
    assert "functional_non_isolated" in markdown
    assert "claim_ledger.json" in markdown
    assert "isolated_affinity benchmark claims" in markdown
