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
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from scpn_quantum_control.differentiable_external_validation import (
    EXTERNAL_VALIDATION_ARTIFACT_BUNDLE_ARTIFACT_ID,
    EXTERNAL_VALIDATION_ARTIFACT_BUNDLE_SCHEMA,
    EXTERNAL_VALIDATION_ENVIRONMENT_LOCK_ARTIFACT_ID,
    EXTERNAL_VALIDATION_ENVIRONMENT_LOCK_SCHEMA,
    EnvironmentLockfileSummary,
    ExternalValidationArtifactBundle,
    ExternalValidationArtifactEntry,
    ExternalValidationEnvironmentLock,
    ExternalValidationEnvironmentLockValidation,
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
from tools import check_differentiable_external_validation as _manifest_gate


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
        "unexpected artifact_id: candidate",
        "environment lock manifest must remain functional_non_isolated",
        "environment lock manifest claim boundary is not explicit enough",
        "environment lockfile inventory does not match DEFAULT_ENVIRONMENT_LOCK_INPUTS",
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
        "data/differentiable_phase_qnode/compiler_alias_activity_evidence_20260706.json" in paths
    )
    assert "data/differentiable_phase_qnode/compiler_promotion_batch_20260706.json" in paths
    assert "data/differentiable_phase_qnode/compiler_promotion_batch_20260706.md" in paths
    assert (
        "data/differentiable_phase_qnode/native_whole_program_ad_execution_evidence_20260622.json"
        in paths
    )
    assert "data/differentiable_phase_qnode/llvm_jit_claim_gate_20260704.json" in paths
    assert (
        "data/differentiable_phase_qnode/enzyme_mlir_compiler_ad_breadth_artifact_20260706.json"
    ) in paths
    assert (
        "data/differentiable_phase_qnode/enzyme_mlir_compiler_ad_breadth_artifact_20260706.md"
    ) in paths
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
    assert "data/differentiable_phase_qnode/compiler_alias_activity_evidence_20260706.md" in (
        validation.checked_paths
    )
    assert "data/differentiable_phase_qnode/compiler_promotion_batch_20260706.json" in (
        validation.checked_paths
    )
    assert "data/differentiable_phase_qnode/compiler_promotion_batch_20260706.md" in (
        validation.checked_paths
    )
    assert "data/differentiable_phase_qnode/llvm_jit_claim_gate_20260704.md" in (
        validation.checked_paths
    )
    assert (
        "data/differentiable_phase_qnode/enzyme_mlir_compiler_ad_breadth_artifact_20260706.md"
    ) in validation.checked_paths


def test_external_validation_manifest_gate_passes_live_pairs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The repository gate must accept both current committed manifest pairs."""
    assert _manifest_gate.audit_manifests() == ()
    assert _manifest_gate.main([]) == 0
    assert "manifest gate: PASS" in capsys.readouterr().out


def test_external_validation_manifest_gate_prefixes_both_failure_classes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The gate must distinguish environment drift from bundle drift."""
    environment_failure = ExternalValidationEnvironmentLockValidation(
        passed=False,
        errors=("environment drift",),
        checked_paths=(),
    )
    bundle_failure = ExternalValidationEnvironmentLockValidation(
        passed=False,
        errors=("bundle drift",),
        checked_paths=(),
    )
    monkeypatch.setattr(
        _manifest_gate, "load_external_validation_environment_lock", lambda path: object()
    )
    monkeypatch.setattr(
        _manifest_gate, "load_external_validation_artifact_bundle", lambda path: object()
    )
    monkeypatch.setattr(
        _manifest_gate,
        "validate_external_validation_environment_lock",
        lambda manifest, repo_root: environment_failure,
    )
    monkeypatch.setattr(
        _manifest_gate,
        "validate_external_validation_artifact_bundle",
        lambda bundle, repo_root: bundle_failure,
    )

    assert _manifest_gate.audit_manifests() == (
        "environment: environment drift",
        "bundle: bundle drift",
    )
    assert _manifest_gate.main([]) == 1
    output = capsys.readouterr().out
    assert "manifest gate: FAIL" in output
    assert "environment: environment drift" in output
    assert "bundle: bundle drift" in output


def test_external_validation_manifest_gate_refreshes_dependency_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refresh the environment pair before building its dependent bundle."""
    environment_path = tmp_path / "environment.json"
    environment_markdown_path = tmp_path / "environment.md"
    bundle_path = tmp_path / "bundle.json"
    bundle_markdown_path = tmp_path / "bundle.md"
    environment = ExternalValidationEnvironmentLock(
        artifact_id="environment",
        schema=EXTERNAL_VALIDATION_ENVIRONMENT_LOCK_SCHEMA,
        python_version="3.12.0",
        platform="test",
        lockfiles=(),
        classification="functional_non_isolated",
        claim_boundary="no isolated_affinity benchmark claims",
    )
    bundle = ExternalValidationArtifactBundle(
        artifact_id="bundle",
        schema="scpn_qc_differentiable_external_validation_artifact_bundle_v1",
        entries=(),
        classification="functional_non_isolated",
        claim_boundary="no isolated_affinity benchmark claims",
    )
    monkeypatch.setattr(_manifest_gate, "ROOT", tmp_path)
    monkeypatch.setattr(_manifest_gate, "ENVIRONMENT_PATH", environment_path)
    monkeypatch.setattr(
        _manifest_gate,
        "ENVIRONMENT_MARKDOWN_PATH",
        environment_markdown_path,
    )
    monkeypatch.setattr(_manifest_gate, "BUNDLE_PATH", bundle_path)
    monkeypatch.setattr(_manifest_gate, "BUNDLE_MARKDOWN_PATH", bundle_markdown_path)
    monkeypatch.setattr(
        _manifest_gate,
        "build_external_validation_environment_lock",
        lambda repo_root: environment,
    )

    def build_bundle(*, repo_root: Path) -> ExternalValidationArtifactBundle:
        assert repo_root == tmp_path
        assert environment_path.is_file()
        assert environment_markdown_path.is_file()
        return bundle

    monkeypatch.setattr(
        _manifest_gate,
        "build_external_validation_artifact_bundle",
        build_bundle,
    )
    monkeypatch.setattr(
        _manifest_gate,
        "render_external_validation_environment_lock_markdown",
        lambda manifest: "environment markdown",
    )
    monkeypatch.setattr(
        _manifest_gate,
        "render_external_validation_artifact_bundle_markdown",
        lambda manifest: "bundle markdown",
    )

    _manifest_gate.refresh_manifests()

    assert json.loads(environment_path.read_text(encoding="utf-8"))["artifact_id"] == (
        "environment"
    )
    assert json.loads(bundle_path.read_text(encoding="utf-8"))["artifact_id"] == "bundle"
    assert environment_markdown_path.read_text(encoding="utf-8") == "environment markdown\n"
    assert bundle_markdown_path.read_text(encoding="utf-8") == "bundle markdown\n"


def test_external_validation_manifest_gate_write_mode_refreshes_then_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The write CLI mode must refresh before its mandatory validation pass."""
    calls: list[str] = []

    def audit() -> tuple[()]:
        """Record the read-only audit call and return no findings."""
        calls.append("audit")
        return ()

    monkeypatch.setattr(_manifest_gate, "refresh_manifests", lambda: calls.append("refresh"))
    monkeypatch.setattr(_manifest_gate, "audit_manifests", audit)

    assert _manifest_gate.main(["--write"]) == 0
    assert calls == ["refresh", "audit"]


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
    alias_payload = json.loads(
        Path(
            "data/differentiable_phase_qnode/compiler_alias_activity_evidence_20260706.json"
        ).read_text(encoding="utf-8")
    )
    batch_payload = json.loads(
        Path("data/differentiable_phase_qnode/compiler_promotion_batch_20260706.json").read_text(
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
    assert payload["enzyme_mlir"]["breadth_artifact_id"] == (
        "enzyme-mlir-compiler-ad-breadth-artifact-20260706"
    )
    assert "scalar_forward_mode" in required
    assert "native_enzyme_execution" in required
    assert required["native_enzyme_execution"]["status"] in {"evidence_attached", "hard_gap"}
    assert alias_payload["schema"] == "scpn_qc_compiler_alias_activity_evidence_v1"
    assert alias_payload["artifact_id"] == "compiler-alias-activity-evidence-20260706"
    assert alias_payload["classification"] == "functional_non_isolated"
    assert alias_payload["promotion_ready"] is False
    assert alias_payload["alias_activity_verified"] is True
    assert set(alias_payload["observed_alias_edge_kinds"]) >= {
        "control_path_alias",
        "expression_rebinding_alias",
        "list_alias",
        "local_rebinding_alias",
        "loop_carried_state",
        "object_attribute_alias",
        "view_alias",
    }
    assert alias_payload["complete_lattice_case_count"] >= 3
    assert alias_payload["blocked_lattice_case_count"] >= 3
    assert batch_payload["schema"] == "scpn_qc_compiler_promotion_batch_v1"
    assert batch_payload["artifact_id"] == "compiler-promotion-batch-20260706"
    assert batch_payload["status"] == "blocked_missing_isolated_compiler_benchmark_ids"
    assert batch_payload["promotion_ready"] is False
    assert batch_payload["promotion_blockers"] == [
        "isolated compiler benchmark artifact IDs missing"
    ]
    assert required["alias_activity"]["status"] == "bounded_evidence_attached"
    assert required["alias_activity"]["artifact_ids"] == [
        "compiler-alias-activity-evidence-20260706"
    ]
    assert payload["promotion_batch"]["artifact_id"] == "compiler-promotion-batch-20260706"
    assert payload["promotion_batch"]["promotion_ready"] is False
    assert (
        payload["promotion_batch"]["status"] == "blocked_missing_isolated_compiler_benchmark_ids"
    )
    assert "isolated compiler benchmark artifact IDs missing" in payload["promotion_blockers"]
    assert "compiler promotion batch not assembled" not in payload["promotion_blockers"]
    assert "alias-activity compiler evidence missing" not in payload["promotion_blockers"]
    assert "provider, hardware, GPU, or performance claim" in payload["claim_boundary"]


def test_compiler_evidence_boundary_artifact_records_working_native_selector() -> None:
    """Compiler boundary evidence must not prescribe zero-selected selectors."""
    payload = json.loads(
        Path("data/differentiable_phase_qnode/compiler_evidence_boundary_20260705.json").read_text(
            encoding="utf-8"
        )
    )
    native = payload["native_llvm_jit"]

    assert native["prescribed_selector_policy"] == "path_targeted_positive_selector"
    assert native["prescribed_selector_exit_code"] == 0
    assert native["prescribed_selector_passed_count"] == native["focused_test_passed_count"]
    assert (
        native["prescribed_selector_deselected_count"] == native["focused_test_deselected_count"]
    )
    assert "not realtime" not in native["prescribed_selector_command"]
    assert "zero tests" not in str(native["prescribed_selector_observation"])
    assert (
        "prescribed native_llvm_jit selector selected zero tests"
        not in payload["promotion_blockers"]
    )


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
        "unexpected artifact_id: candidate",
        "artifact bundle must remain functional_non_isolated",
        "artifact bundle claim boundary is not explicit enough",
        "artifact bundle inventory does not match DEFAULT_ARTIFACT_BUNDLE_INPUTS",
        "size mismatch: evidence.json",
        "missing artefact: missing.json",
    )


def test_artifact_entry_summary_rejects_missing_file(tmp_path: Path) -> None:
    """Artefact entry summaries must fail closed on missing files."""
    missing = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError) as exc_info:
        summarize_artifact_entry(missing, repo_root=tmp_path, role="missing")

    assert str(exc_info.value) == f"external-validation artefact is missing: {missing}"


def test_external_validation_summaries_reject_repository_escape(tmp_path: Path) -> None:
    """Absolute, traversing, and escaping symlink evidence cannot be summarized."""
    outside = tmp_path.parent / "outside-external-evidence.txt"
    outside.write_text("untrusted\n", encoding="utf-8")
    symlink = tmp_path / "escaped-link.txt"
    symlink.symlink_to(outside)

    with pytest.raises(ValueError, match="environment lockfile escapes repository"):
        summarize_environment_lockfile(outside, repo_root=tmp_path, role="outside")
    with pytest.raises(ValueError, match="external-validation artefact escapes repository"):
        summarize_artifact_entry(
            Path("../outside-external-evidence.txt"),
            repo_root=tmp_path,
            role="outside",
        )
    with pytest.raises(ValueError, match="external-validation artefact escapes repository"):
        summarize_artifact_entry(symlink, repo_root=tmp_path, role="symlink")


def test_external_validation_manifests_report_unsafe_paths(tmp_path: Path) -> None:
    """Manifest validation converts repository escape into explicit findings."""
    lockfile = EnvironmentLockfileSummary(
        path="../outside-lock.txt",
        role="outside",
        sha256="0" * 64,
        size_bytes=1,
        line_count=1,
        pinned_package_count=0,
    )
    manifest = ExternalValidationEnvironmentLock(
        artifact_id=EXTERNAL_VALIDATION_ENVIRONMENT_LOCK_ARTIFACT_ID,
        schema=EXTERNAL_VALIDATION_ENVIRONMENT_LOCK_SCHEMA,
        python_version="3.12.0",
        platform="test",
        lockfiles=(lockfile,),
        classification="functional_non_isolated",
        claim_boundary="no isolated_affinity benchmark claims",
    )
    assert validate_external_validation_environment_lock(
        manifest,
        repo_root=tmp_path,
    ).errors == (
        "environment lockfile inventory does not match DEFAULT_ENVIRONMENT_LOCK_INPUTS",
        "unsafe lockfile path: ../outside-lock.txt",
    )

    entry = ExternalValidationArtifactEntry(
        path="../outside-evidence.json",
        role="outside",
        sha256="0" * 64,
        size_bytes=1,
    )
    bundle = ExternalValidationArtifactBundle(
        artifact_id=EXTERNAL_VALIDATION_ARTIFACT_BUNDLE_ARTIFACT_ID,
        schema=EXTERNAL_VALIDATION_ARTIFACT_BUNDLE_SCHEMA,
        entries=(entry,),
        classification="functional_non_isolated",
        claim_boundary="no isolated_affinity benchmark claims",
    )
    assert validate_external_validation_artifact_bundle(
        bundle,
        repo_root=tmp_path,
    ).errors == (
        "artifact bundle inventory does not match DEFAULT_ARTIFACT_BUNDLE_INPUTS",
        "unsafe artefact path: ../outside-evidence.json",
    )


def test_artifact_bundle_markdown_lists_claim_boundary() -> None:
    """Artefact bundle Markdown must expose the claim boundary."""
    markdown = render_external_validation_artifact_bundle_markdown(
        build_external_validation_artifact_bundle()
    )

    assert "# Differentiable External-Validation Artefact Bundle" in markdown
    assert "functional_non_isolated" in markdown
    assert "claim_ledger.json" in markdown
    assert "isolated_affinity benchmark claims" in markdown


def test_external_validation_loaders_reject_coercive_json_types(tmp_path: Path) -> None:
    """Manifest loaders cannot convert numeric or string-shaped evidence fields."""
    environment_path = tmp_path / "environment.json"
    environment_payload = {
        "artifact_id": 7,
        "schema": EXTERNAL_VALIDATION_ENVIRONMENT_LOCK_SCHEMA,
        "python_version": "3.12.0",
        "platform": "test",
        "lockfiles": [],
        "classification": "functional_non_isolated",
        "claim_boundary": "no isolated_affinity benchmark claims",
    }
    environment_path.write_text(json.dumps(environment_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact_id must be a non-empty string"):
        load_external_validation_environment_lock(environment_path)

    bundle_path = tmp_path / "bundle.json"
    bundle_payload = {
        "artifact_id": "bundle",
        "schema": "scpn_qc_differentiable_external_validation_artifact_bundle_v1",
        "entries": [
            {
                "path": "evidence.json",
                "role": "evidence",
                "sha256": "0" * 64,
                "size_bytes": "1",
            }
        ],
        "classification": "functional_non_isolated",
        "claim_boundary": "no isolated_affinity benchmark claims",
    }
    bundle_path.write_text(json.dumps(bundle_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="size_bytes must be a non-negative integer"):
        load_external_validation_artifact_bundle(bundle_path)


def test_external_validation_models_reject_duplicate_and_malformed_evidence() -> None:
    """Manifest identities remain unique and metadata retains exact runtime types."""
    summary = EnvironmentLockfileSummary(
        path="requirements.txt",
        role="runtime",
        sha256="0" * 64,
        size_bytes=1,
        line_count=1,
        pinned_package_count=1,
    )
    manifest = ExternalValidationEnvironmentLock(
        artifact_id="environment",
        schema=EXTERNAL_VALIDATION_ENVIRONMENT_LOCK_SCHEMA,
        python_version="3.12.0",
        platform="test",
        lockfiles=(summary,),
        classification="functional_non_isolated",
        claim_boundary="no isolated_affinity benchmark claims",
    )
    with pytest.raises(ValueError, match="lockfiles must have unique paths"):
        replace(manifest, lockfiles=(summary, summary))
    with pytest.raises(ValueError, match="size_bytes must be a non-negative integer"):
        replace(summary, size_bytes=cast(int, True))
    with pytest.raises(ValueError, match="size_bytes must be a non-negative integer"):
        replace(summary, size_bytes=-1)
    with pytest.raises(ValueError, match="sha256 must be a lowercase SHA-256 digest"):
        replace(summary, sha256="not-a-digest")
    with pytest.raises(ValueError, match="sha256 must be a lowercase SHA-256 digest"):
        replace(summary, sha256=cast(str, 1))
    with pytest.raises(ValueError, match="lockfiles must be a tuple"):
        replace(manifest, lockfiles=(cast(EnvironmentLockfileSummary, "bad"),))

    entry = ExternalValidationArtifactEntry(
        path="evidence.json",
        role="evidence",
        sha256="0" * 64,
        size_bytes=1,
    )
    bundle = ExternalValidationArtifactBundle(
        artifact_id="bundle",
        schema="scpn_qc_differentiable_external_validation_artifact_bundle_v1",
        entries=(entry,),
        classification="functional_non_isolated",
        claim_boundary="no isolated_affinity benchmark claims",
    )
    with pytest.raises(ValueError, match="entries must have unique paths"):
        replace(bundle, entries=(entry, entry))
    with pytest.raises(ValueError, match="entries must be a tuple"):
        replace(bundle, entries=cast(tuple[ExternalValidationArtifactEntry, ...], [entry]))
    with pytest.raises(ValueError, match="artifact_id must be a non-empty string"):
        replace(bundle, artifact_id=cast(str, 1))


def test_external_validation_result_requires_pass_error_coherence() -> None:
    """A passing validation cannot carry findings or omit its checked-path tuple."""
    with pytest.raises(ValueError, match="true exactly when errors are empty"):
        ExternalValidationEnvironmentLockValidation(
            passed=True,
            errors=("drift",),
            checked_paths=(),
        )
    with pytest.raises(ValueError, match="passed must be boolean"):
        ExternalValidationEnvironmentLockValidation(
            passed=cast(bool, 1),
            errors=(),
            checked_paths=(),
        )
    with pytest.raises(ValueError, match="checked_paths must contain"):
        ExternalValidationEnvironmentLockValidation(
            passed=True,
            errors=(),
            checked_paths=cast(tuple[str, ...], ["evidence.json"]),
        )
    with pytest.raises(ValueError, match="errors must contain"):
        ExternalValidationEnvironmentLockValidation(
            passed=False,
            errors=("",),
            checked_paths=(),
        )


def test_external_validation_loaders_reject_malformed_json_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manifest roots and evidence collections must retain exact JSON shapes."""
    path = tmp_path / "manifest.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_external_validation_artifact_bundle(path)

    base = {
        "artifact_id": "environment",
        "schema": EXTERNAL_VALIDATION_ENVIRONMENT_LOCK_SCHEMA,
        "python_version": "3.12.0",
        "platform": "test",
        "classification": "functional_non_isolated",
        "claim_boundary": "no isolated_affinity benchmark claims",
    }
    path.write_text(json.dumps({**base, "lockfiles": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="lockfiles must be a list"):
        load_external_validation_environment_lock(path)

    path.write_text(json.dumps({**base, "lockfiles": ["bad"]}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"lockfiles\[0\] must be an object"):
        load_external_validation_environment_lock(path)

    monkeypatch.setattr(json, "loads", lambda text: {1: "bad"})
    with pytest.raises(ValueError, match="string keys"):
        load_external_validation_environment_lock(path)
