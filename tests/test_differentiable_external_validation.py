# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable external-validation tests
"""Tests for differentiable external-validation package manifests."""

from __future__ import annotations

from pathlib import Path

from scpn_quantum_control.differentiable_external_validation import (
    EXTERNAL_VALIDATION_ENVIRONMENT_LOCK_SCHEMA,
    EnvironmentLockfileSummary,
    ExternalValidationEnvironmentLock,
    build_external_validation_environment_lock,
    load_external_validation_environment_lock,
    render_external_validation_environment_lock_markdown,
    summarize_environment_lockfile,
    validate_external_validation_environment_lock,
)


def test_build_external_validation_environment_lock_records_exact_lockfiles() -> None:
    manifest = build_external_validation_environment_lock()

    paths = {lockfile.path for lockfile in manifest.lockfiles}
    assert manifest.schema == EXTERNAL_VALIDATION_ENVIRONMENT_LOCK_SCHEMA
    assert manifest.classification == "functional_non_isolated"
    assert "isolated_affinity benchmark claims" in manifest.claim_boundary
    assert "pyproject.toml" in paths
    assert "requirements-ci-py310-linux.txt" in paths
    assert "requirements-ci-py313-linux.txt" in paths
    assert (
        "data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/"
        "framework_overlay_freeze.txt"
    ) in paths
    assert all(len(lockfile.sha256) == 64 for lockfile in manifest.lockfiles)
    assert all(lockfile.size_bytes > 0 for lockfile in manifest.lockfiles)


def test_committed_external_validation_environment_lock_matches_files() -> None:
    manifest = load_external_validation_environment_lock()
    validation = validate_external_validation_environment_lock(manifest)

    assert validation.passed
    assert not validation.errors
    assert "pyproject.toml" in validation.checked_paths
    assert "requirements-ci-py312-linux.txt" in validation.checked_paths


def test_environment_lock_validation_rejects_hash_drift() -> None:
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


def test_summarize_environment_lockfile_counts_pinned_packages(tmp_path: Path) -> None:
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


def test_environment_lock_markdown_lists_claim_boundary() -> None:
    markdown = render_external_validation_environment_lock_markdown(
        build_external_validation_environment_lock()
    )

    assert "# Differentiable External-Validation Environment Lock" in markdown
    assert "functional_non_isolated" in markdown
    assert "pyproject.toml" in markdown
    assert "isolated_affinity benchmark claims" in markdown
