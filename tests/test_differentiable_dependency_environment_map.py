# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable dependency environment map tests
"""Tests for differentiable dependency and environment evidence governance."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

import scpn_quantum_control as scpn
from scpn_quantum_control import (
    DifferentiableDependencyEnvironmentEvidence,
    DifferentiableDependencyEnvironmentMap,
    DifferentiableDependencyEnvironmentMapValidation,
    DifferentiableDependencyEnvironmentProfile,
    DifferentiableDependencyEnvironmentStatus,
    build_differentiable_dependency_environment_evidence,
    differentiable_api,
    differentiable_module_hardening_registry,
    render_differentiable_dependency_environment_map_markdown,
    run_differentiable_dependency_environment_map,
    validate_differentiable_dependency_environment_map,
)
from scpn_quantum_control.differentiable_dependency_environment_map import (
    DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_MAP_CLAIM_BOUNDARY,
    DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_VALIDATION_CLAIM_BOUNDARY,
)
from scpn_quantum_control.differentiable_external_validation import (
    build_external_validation_environment_lock,
)


def _valid_profile() -> DifferentiableDependencyEnvironmentProfile:
    """Return a minimal valid public dependency profile.

    Returns
    -------
    DifferentiableDependencyEnvironmentProfile
        Locked profile with one checksummed dependency path.
    """
    return _profile_record()


def _profile_record(
    *,
    profile_id: object = "runtime_baseline",
    title: object = "Runtime baseline",
    role: object = "Test dependency environment.",
    lockfile_paths: object = ("requirements.txt",),
    evidence_paths: object = ("requirements.txt",),
    pinned_package_count: object = 1,
    checksum_count: object = 1,
    evidence_status: object = "locked",
    blockers: object = (),
    claim_boundary: object = DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_MAP_CLAIM_BOUNDARY,
) -> DifferentiableDependencyEnvironmentProfile:
    """Construct one profile through its typed public contract.

    Parameters
    ----------
    profile_id : str, optional
        Profile identifier to validate.
    title : str, optional
        Reviewer-facing title to validate.
    role : str, optional
        Profile role to validate.
    lockfile_paths : tuple[str, ...], optional
        Lockfile paths to validate.
    evidence_paths : tuple[str, ...], optional
        Evidence paths to validate.
    pinned_package_count : int, optional
        Pinned-package count to validate.
    checksum_count : int, optional
        Checksum count to validate.
    evidence_status : DifferentiableDependencyEnvironmentStatus, optional
        Evidence classification to validate.
    claim_boundary : str, optional
        Claim boundary to validate.

    Returns
    -------
    DifferentiableDependencyEnvironmentProfile
        Profile created by the production dataclass constructor.
    """
    return DifferentiableDependencyEnvironmentProfile(
        profile_id=cast(str, profile_id),
        title=cast(str, title),
        role=cast(str, role),
        lockfile_paths=cast(tuple[str, ...], lockfile_paths),
        evidence_paths=cast(tuple[str, ...], evidence_paths),
        pinned_package_count=cast(int, pinned_package_count),
        checksum_count=cast(int, checksum_count),
        evidence_status=cast(DifferentiableDependencyEnvironmentStatus, evidence_status),
        blockers=cast(tuple[str, ...], blockers),
        claim_boundary=cast(str, claim_boundary),
    )


def _map_record(
    *,
    schema: object = "test-schema",
    artifact_id: object = "test-artifact",
    profiles: object | None = None,
    environment_ready: object = True,
    ready_profile_count: object = 1,
    total_profile_count: object = 1,
    evidence_records: object | None = None,
    ready_evidence_count: object | None = None,
    total_evidence_count: object | None = None,
    claim_boundary: object = DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_MAP_CLAIM_BOUNDARY,
) -> DifferentiableDependencyEnvironmentMap:
    """Construct aggregate evidence while permitting malformed test inputs."""
    loaded_profiles = (_valid_profile(),) if profiles is None else profiles
    loaded_evidence = (
        build_differentiable_dependency_environment_evidence()
        if evidence_records is None
        else evidence_records
    )
    ready_evidence = (
        sum(
            1
            for record in cast(
                tuple[DifferentiableDependencyEnvironmentEvidence, ...],
                loaded_evidence,
            )
            if record.environment_ready
        )
        if ready_evidence_count is None
        else ready_evidence_count
    )
    evidence_count = len(cast(tuple[object, ...], loaded_evidence))
    if total_evidence_count is not None:
        evidence_count = cast(int, total_evidence_count)
    return DifferentiableDependencyEnvironmentMap(
        schema=cast(str, schema),
        artifact_id=cast(str, artifact_id),
        profiles=cast(tuple[DifferentiableDependencyEnvironmentProfile, ...], loaded_profiles),
        environment_ready=cast(bool, environment_ready),
        ready_profile_count=cast(int, ready_profile_count),
        total_profile_count=cast(int, total_profile_count),
        evidence_records=cast(
            tuple[DifferentiableDependencyEnvironmentEvidence, ...], loaded_evidence
        ),
        ready_evidence_count=cast(int, ready_evidence),
        total_evidence_count=evidence_count,
        claim_boundary=cast(str, claim_boundary),
    )


def _validation_record(
    *,
    passed: object = True,
    errors: object = (),
    checked_profile_ids: object = ("runtime_baseline",),
    checked_evidence_ids: object = ("python_versions",),
    checked_paths: object = ("requirements.txt",),
    checked_lockfile_count: object = 1,
    checked_pinned_package_count: object = 1,
    claim_boundary: object = DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_VALIDATION_CLAIM_BOUNDARY,
) -> DifferentiableDependencyEnvironmentMapValidation:
    """Construct validation evidence while permitting malformed test inputs."""
    return DifferentiableDependencyEnvironmentMapValidation(
        passed=cast(bool, passed),
        errors=cast(tuple[str, ...], errors),
        checked_profile_ids=cast(tuple[str, ...], checked_profile_ids),
        checked_evidence_ids=cast(tuple[str, ...], checked_evidence_ids),
        checked_paths=cast(tuple[str, ...], checked_paths),
        checked_lockfile_count=cast(int, checked_lockfile_count),
        checked_pinned_package_count=cast(int, checked_pinned_package_count),
        claim_boundary=cast(str, claim_boundary),
    )


def test_dependency_environment_map_records_required_profiles() -> None:
    """The map must bind every differentiable runtime profile to lock evidence."""
    environment_map = run_differentiable_dependency_environment_map()

    assert environment_map.schema == "scpn_qc_differentiable_dependency_environment_map_v2"
    assert environment_map.environment_ready is False
    assert environment_map.total_profile_count == len(environment_map.profiles)
    assert environment_map.ready_profile_count < environment_map.total_profile_count
    assert {profile.profile_id for profile in environment_map.profiles} == {
        "runtime_baseline",
        "development_verification",
        "ci_python_matrix",
        "framework_overlay_cpu",
        "enzyme_runner_py39",
    }
    assert "no dependency or benchmark promotion" in environment_map.claim_boundary
    assert environment_map.total_evidence_count == len(environment_map.evidence_records)
    assert {record.evidence_id for record in environment_map.evidence_records} == {
        "python_versions",
        "rust_crates",
        "jax_cpu",
        "pytorch_cpu",
        "tensorflow_cpu",
        "pennylane_cpu",
        "qiskit",
        "catalyst",
        "enzyme_llvm_mlir",
        "gpu_overlay",
        "local_cpu",
        "jarvislabs_cloud",
        "provider_execution",
        "hardware_ticket",
        "gtx1060_workstation",
        "ml350_isolated",
    }


def test_dependency_environment_profiles_are_lockfile_backed() -> None:
    """Each profile must cite real lockfiles and explicit blocker state."""
    environment_map = run_differentiable_dependency_environment_map()
    profiles = {profile.profile_id: profile for profile in environment_map.profiles}

    runtime = profiles["runtime_baseline"]
    assert runtime.lockfile_paths == ("requirements.txt",)
    assert runtime.pinned_package_count > 0
    assert runtime.evidence_status == "locked"
    assert runtime.blockers == ()

    ci_matrix = profiles["ci_python_matrix"]
    assert {
        "requirements-ci-py311-linux.txt",
        "requirements-ci-py312-linux.txt",
        "requirements-ci-py313-linux.txt",
    } <= set(ci_matrix.lockfile_paths)
    assert ci_matrix.evidence_status == "locked"

    enzyme = profiles["enzyme_runner_py39"]
    assert enzyme.evidence_status == "hard_gap"
    assert any("toolchain" in blocker for blocker in enzyme.blockers)

    validation = validate_differentiable_dependency_environment_map(environment_map)
    assert validation.passed, validation.errors
    assert (
        "data/differentiable_phase_qnode/differentiable_dependency_environment_map_20260627.md"
        in (validation.checked_paths)
    )


def test_dependency_environment_committed_artifacts_match_live_lock_evidence() -> None:
    """Committed JSON and Markdown must exactly match the live deterministic map."""
    environment_map = run_differentiable_dependency_environment_map()
    artifact_root = Path("data/differentiable_phase_qnode")
    committed_json = json.loads(
        (artifact_root / "differentiable_dependency_environment_map_20260627.json").read_text(
            encoding="utf-8"
        )
    )
    committed_markdown = (
        artifact_root / "differentiable_dependency_environment_map_20260627.md"
    ).read_text(encoding="utf-8")

    assert committed_json == environment_map.to_dict()
    assert committed_markdown == render_differentiable_dependency_environment_map_markdown(
        environment_map
    )


def test_dependency_environment_map_validation_rejects_stale_lock_paths(
    tmp_path: Path,
) -> None:
    """Validation must fail closed when lockfile evidence is stale or over-promoted."""
    invalid_profile = DifferentiableDependencyEnvironmentProfile(
        profile_id="invalid_profile",
        title="Invalid profile",
        role="Invalid dependency evidence.",
        lockfile_paths=("missing-requirements.txt",),
        evidence_paths=("missing-requirements.txt",),
        pinned_package_count=0,
        checksum_count=0,
        evidence_status="locked",
        blockers=(),
        claim_boundary=DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_MAP_CLAIM_BOUNDARY,
    )
    environment_map = type(run_differentiable_dependency_environment_map())(
        schema="scpn_qc_differentiable_dependency_environment_map_v2",
        artifact_id="test-dependency-environment-map",
        profiles=(invalid_profile,),
        environment_ready=True,
        ready_profile_count=1,
        total_profile_count=1,
        evidence_records=run_differentiable_dependency_environment_map().evidence_records,
        ready_evidence_count=(
            run_differentiable_dependency_environment_map().ready_evidence_count
        ),
        total_evidence_count=(
            run_differentiable_dependency_environment_map().total_evidence_count
        ),
        claim_boundary=DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_MAP_CLAIM_BOUNDARY,
    )

    validation = validate_differentiable_dependency_environment_map(
        environment_map,
        environment_lock=build_external_validation_environment_lock(),
        repo_root=tmp_path,
    )

    assert not validation.passed
    assert any("evidence path does not exist" in error for error in validation.errors)
    assert any(
        "locked profiles must record pinned packages" in error for error in validation.errors
    )


def test_dependency_environment_map_markdown_unified_api_and_exports() -> None:
    """The map must render, dispatch, and export through public package surfaces."""
    environment_map = run_differentiable_dependency_environment_map()
    markdown = render_differentiable_dependency_environment_map_markdown(environment_map)
    result = differentiable_api("dependency_environment_map")

    assert "# Differentiable Dependency and Environment Evidence Map" in markdown
    assert "framework_overlay_cpu" in markdown
    assert result.operation == "dependency_environment_map"
    assert result.supported is False
    assert result.payload["total_profile_count"] == environment_map.total_profile_count
    assert "run_differentiable_dependency_environment_map" in scpn.__all__
    assert (
        scpn.run_differentiable_dependency_environment_map
        is run_differentiable_dependency_environment_map
    )
    registry_paths = {record.module_path for record in differentiable_module_hardening_registry()}
    assert (
        "src/scpn_quantum_control/differentiable_dependency_environment_map.py" in registry_paths
    )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        pytest.param(
            lambda: _profile_record(profile_id=" "),
            "profile_id must be a non-empty string",
            id="empty-profile-id",
        ),
        pytest.param(
            lambda: _profile_record(profile_id=7),
            "profile_id must be a non-empty string",
            id="non-string-profile-id",
        ),
        pytest.param(
            lambda: _profile_record(title=""),
            "title must be a non-empty string",
            id="empty-title",
        ),
        pytest.param(
            lambda: _profile_record(role="\t"),
            "role must be a non-empty string",
            id="empty-role",
        ),
        pytest.param(
            lambda: _profile_record(claim_boundary="test-only dependency evidence"),
            "claim_boundary must match the canonical dependency boundary",
            id="noncanonical-claim-boundary",
        ),
        pytest.param(
            lambda: _profile_record(lockfile_paths=()),
            "lockfile_paths must be non-empty",
            id="empty-lockfile-paths",
        ),
        pytest.param(
            lambda: _profile_record(lockfile_paths=["requirements.txt"]),
            "lockfile_paths must be a tuple",
            id="list-lockfile-paths",
        ),
        pytest.param(
            lambda: _profile_record(lockfile_paths=("requirements.txt", "requirements.txt")),
            "lockfile_paths must contain unique entries",
            id="duplicate-lockfile-paths",
        ),
        pytest.param(
            lambda: _profile_record(evidence_paths=("",)),
            "evidence_paths must contain non-empty strings",
            id="blank-evidence-path",
        ),
        pytest.param(
            lambda: _profile_record(pinned_package_count=-1),
            "pinned_package_count must be a non-negative integer",
            id="negative-pinned-count",
        ),
        pytest.param(
            lambda: _profile_record(pinned_package_count=True),
            "pinned_package_count must be a non-negative integer",
            id="boolean-pinned-count",
        ),
        pytest.param(
            lambda: _profile_record(checksum_count=-1),
            "checksum_count must be a non-negative integer",
            id="negative-checksum-count",
        ),
        pytest.param(
            lambda: _profile_record(
                evidence_status=cast(
                    DifferentiableDependencyEnvironmentStatus,
                    "unknown",
                )
            ),
            "evidence_status must be locked or hard_gap",
            id="unknown-status",
        ),
        pytest.param(
            lambda: _profile_record(blockers=("unexpected blocker",)),
            "locked profiles must not carry blockers",
            id="locked-with-blocker",
        ),
        pytest.param(
            lambda: _profile_record(evidence_status="hard_gap"),
            "hard_gap profiles must list blockers",
            id="hard-gap-without-blocker",
        ),
        pytest.param(
            lambda: _profile_record(checksum_count=2),
            "checksum_count must not exceed lockfile count",
            id="excess-checksum-count",
        ),
        pytest.param(
            lambda: _profile_record(blockers=["gap"]),
            "blockers must be a tuple",
            id="list-blockers",
        ),
        pytest.param(
            lambda: _profile_record(
                evidence_status="hard_gap",
                blockers=("gap", "gap"),
            ),
            "blockers must contain unique entries",
            id="duplicate-blockers",
        ),
    ],
)
def test_dependency_profile_rejects_invalid_public_fields(
    factory: Callable[[], DifferentiableDependencyEnvironmentProfile],
    message: str,
) -> None:
    """The public profile record must reject every malformed field family."""
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        pytest.param(
            lambda: _map_record(profiles=[]),
            "profiles must be a non-empty tuple",
            id="list-profiles",
        ),
        pytest.param(
            lambda: _map_record(profiles=()),
            "profiles must be a non-empty tuple",
            id="empty-profiles",
        ),
        pytest.param(
            lambda: _map_record(profiles=(object(),)),
            "profiles must contain dependency environment profiles",
            id="wrong-profile-type",
        ),
        pytest.param(
            lambda: _map_record(environment_ready=1),
            "environment_ready must be a bool",
            id="integer-readiness",
        ),
        pytest.param(
            lambda: _map_record(ready_profile_count=True),
            "ready_profile_count must be a non-negative integer",
            id="boolean-ready-count",
        ),
        pytest.param(
            lambda: _map_record(claim_boundary="test-only boundary"),
            "claim_boundary must match the canonical dependency boundary",
            id="noncanonical-map-boundary",
        ),
        pytest.param(
            lambda: _map_record(evidence_records=[]),
            "evidence_records must be a non-empty tuple",
            id="list-evidence-records",
        ),
        pytest.param(
            lambda: _map_record(
                evidence_records=(object(),),
                ready_evidence_count=0,
            ),
            "evidence_records must contain dependency environment evidence",
            id="wrong-evidence-record-type",
        ),
    ],
)
def test_dependency_environment_map_rejects_malformed_structure(
    factory: Callable[[], DifferentiableDependencyEnvironmentMap],
    message: str,
) -> None:
    """Aggregate evidence must reject malformed containers and scalar types."""
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        pytest.param(
            lambda: _validation_record(passed=1),
            "passed must be a bool",
            id="integer-passed",
        ),
        pytest.param(
            lambda: _validation_record(errors=[]),
            "errors must be a tuple",
            id="list-errors",
        ),
        pytest.param(
            lambda: _validation_record(checked_profile_ids=()),
            "checked_profile_ids must be non-empty",
            id="empty-profile-ids",
        ),
        pytest.param(
            lambda: _validation_record(
                checked_profile_ids=("runtime_baseline", "runtime_baseline")
            ),
            "checked_profile_ids must contain unique entries",
            id="duplicate-profile-ids",
        ),
        pytest.param(
            lambda: _validation_record(passed=True, errors=("failure",)),
            "passed must be true exactly when errors is empty",
            id="passed-with-errors",
        ),
        pytest.param(
            lambda: _validation_record(claim_boundary="test-only boundary"),
            "claim_boundary must match the canonical validation boundary",
            id="noncanonical-validation-boundary",
        ),
    ],
)
def test_dependency_environment_validation_rejects_malformed_structure(
    factory: Callable[[], DifferentiableDependencyEnvironmentMapValidation],
    message: str,
) -> None:
    """Validation records must be typed and internally coherent."""
    with pytest.raises(ValueError, match=message):
        factory()


def test_dependency_environment_validation_serializes_structural_errors() -> None:
    """Validation must report stale schema and aggregate-count metadata."""
    environment_lock = build_external_validation_environment_lock()
    environment_map = run_differentiable_dependency_environment_map(
        environment_lock=environment_lock
    )
    invalid_map = replace(
        environment_map,
        schema="stale-dependency-environment-map",
        total_profile_count=environment_map.total_profile_count + 1,
    )

    validation = validate_differentiable_dependency_environment_map(
        invalid_map,
        environment_lock=environment_lock,
    )
    payload = validation.to_dict()

    assert not validation.passed
    assert (
        "unexpected dependency-environment-map schema: stale-dependency-environment-map"
        in validation.errors
    )
    assert "total_profile_count does not match profile count" in validation.errors
    assert payload["passed"] is False
    assert payload["errors"] == list(validation.errors)
    assert payload["checked_profile_ids"] == list(validation.checked_profile_ids)
    assert payload["checked_evidence_ids"] == list(validation.checked_evidence_ids)
    assert payload["checked_paths"] == list(validation.checked_paths)


def test_dependency_environment_validation_rejects_stale_counts_and_readiness() -> None:
    """Aggregate and profile counts must exactly match the current lock evidence."""
    environment_lock = build_external_validation_environment_lock()
    environment_map = run_differentiable_dependency_environment_map(
        environment_lock=environment_lock
    )
    runtime = environment_map.profiles[0]
    stale_runtime = replace(
        runtime,
        pinned_package_count=runtime.pinned_package_count + 1,
        checksum_count=0,
    )
    stale_map = replace(
        environment_map,
        profiles=(stale_runtime, *environment_map.profiles[1:]),
        ready_profile_count=environment_map.ready_profile_count - 1,
        environment_ready=True,
    )

    validation = validate_differentiable_dependency_environment_map(
        stale_map,
        environment_lock=environment_lock,
    )

    assert not validation.passed
    assert "ready_profile_count does not match ready profiles" in validation.errors
    assert "environment_ready does not match profile and lock readiness" in validation.errors
    assert any("pinned_package_count does not match" in error for error in validation.errors)
    assert any("checksum_count does not match" in error for error in validation.errors)


def test_dependency_environment_validation_rejects_stale_evidence_inventory() -> None:
    """Evidence IDs, classifications, counts, digests, and pins must stay canonical."""
    environment_lock = build_external_validation_environment_lock()
    environment_map = run_differentiable_dependency_environment_map(
        environment_lock=environment_lock
    )
    first, second, third, fourth, *remaining = environment_map.evidence_records
    stale_records = (
        replace(first, evidence_id="unexpected_evidence"),
        replace(second, classification="wrong_classification"),
        replace(third, evidence_sha256=("0" * 64,)),
        replace(fourth, version_pins=("missing-constraint>=99",)),
        *remaining,
    )
    stale_map = replace(
        environment_map,
        evidence_records=stale_records,
        ready_evidence_count=environment_map.ready_evidence_count - 1,
        total_evidence_count=environment_map.total_evidence_count + 1,
    )

    validation = validate_differentiable_dependency_environment_map(
        stale_map,
        environment_lock=environment_lock,
    )

    assert not validation.passed
    assert "ready_evidence_count does not match ready evidence records" in validation.errors
    assert "total_evidence_count does not match evidence record count" in validation.errors
    assert any("evidence IDs must match" in error for error in validation.errors)
    assert any("unexpected evidence classification" in error for error in validation.errors)
    assert any("evidence SHA-256 mismatch" in error for error in validation.errors)
    assert any(
        "uncited version pin: missing-constraint>=99" in error for error in validation.errors
    )


def test_dependency_environment_validation_rejects_unsafe_evidence_record_paths() -> None:
    """Toolchain and route sources must be regular repository-contained files."""
    environment_lock = build_external_validation_environment_lock()
    environment_map = run_differentiable_dependency_environment_map(
        environment_lock=environment_lock
    )
    first, second, *remaining = environment_map.evidence_records
    invalid_records = (
        replace(
            first,
            evidence_paths=("../escape.txt",),
            evidence_sha256=("0" * 64,),
        ),
        replace(
            second,
            evidence_paths=("missing-evidence.txt",),
            evidence_sha256=("0" * 64,),
        ),
        *remaining,
    )
    invalid_map = replace(environment_map, evidence_records=invalid_records)

    validation = validate_differentiable_dependency_environment_map(
        invalid_map,
        environment_lock=environment_lock,
    )

    assert not validation.passed
    assert any("evidence path is unsafe: ../escape.txt" in error for error in validation.errors)
    assert any(
        "evidence path does not exist: missing-evidence.txt" in error
        for error in validation.errors
    )


@pytest.mark.parametrize(
    "unsafe_path",
    [
        pytest.param(" ../escape.md", id="leading-space"),
        pytest.param("locks\\escape.md", id="backslash"),
        pytest.param("/tmp/escape.md", id="absolute"),
        pytest.param("locks//escape.md", id="noncanonical"),
        pytest.param("../escape.md", id="parent-traversal"),
    ],
)
def test_dependency_environment_validation_rejects_unsafe_paths(
    unsafe_path: str,
) -> None:
    """Evidence paths must be canonical repository-contained paths."""
    environment_lock = build_external_validation_environment_lock()
    environment_map = run_differentiable_dependency_environment_map(
        environment_lock=environment_lock
    )
    runtime = environment_map.profiles[0]
    unsafe_runtime = replace(runtime, evidence_paths=(unsafe_path,))
    unsafe_map = replace(
        environment_map,
        profiles=(unsafe_runtime, *environment_map.profiles[1:]),
    )

    validation = validate_differentiable_dependency_environment_map(
        unsafe_map,
        environment_lock=environment_lock,
    )

    assert not validation.passed
    assert any(f"evidence path is unsafe: {unsafe_path}" in error for error in validation.errors)
    if unsafe_path.endswith(".md"):
        assert any(
            f"dependency environment evidence path is unsafe: {unsafe_path}" == error
            for error in validation.errors
        )


def test_dependency_environment_validation_rejects_directory_and_symlink_escape(
    tmp_path: Path,
) -> None:
    """Existing directories and symlinks escaping the root are not file evidence."""
    environment_lock = build_external_validation_environment_lock()
    (tmp_path / "requirements.txt").mkdir()
    external_file = tmp_path.parent / "external-lock.txt"
    external_file.write_text("outside\n", encoding="utf-8")
    (tmp_path / "escape.txt").symlink_to(external_file)
    profile = _profile_record(
        evidence_paths=("requirements.txt", "escape.txt"),
    )
    environment_map = _map_record(profiles=(profile,))

    validation = validate_differentiable_dependency_environment_map(
        environment_map,
        environment_lock=environment_lock,
        repo_root=tmp_path,
    )

    assert not validation.passed
    assert any(
        "evidence path does not exist: requirements.txt" in error for error in validation.errors
    )
    assert any("evidence path is unsafe: escape.txt" in error for error in validation.errors)


def test_dependency_environment_validation_rejects_duplicate_profile_ids() -> None:
    """The aggregate record must reject duplicate identities at construction."""
    environment_lock = build_external_validation_environment_lock()
    environment_map = run_differentiable_dependency_environment_map(
        environment_lock=environment_lock
    )
    profiles = environment_map.profiles
    duplicate = replace(profiles[1], profile_id=profiles[0].profile_id)
    with pytest.raises(ValueError, match="profiles must contain unique profile_id values"):
        replace(
            environment_map,
            profiles=(profiles[0], duplicate, *profiles[2:]),
        )


def test_dependency_environment_map_rejects_duplicate_evidence_ids() -> None:
    """The aggregate record must reject duplicate toolchain or route identities."""
    environment_map = run_differentiable_dependency_environment_map()
    records = environment_map.evidence_records
    duplicate = replace(records[1], evidence_id=records[0].evidence_id)

    with pytest.raises(ValueError, match="unique evidence_id values"):
        replace(
            environment_map,
            evidence_records=(records[0], duplicate, *records[2:]),
        )


def test_dependency_environment_map_rejects_absent_checksum_at_source() -> None:
    """Injected lock evidence cannot omit a checksum before map construction."""
    environment_lock = build_external_validation_environment_lock()
    target_path = "requirements-ci-cross-platform-smoke.txt"
    target = next(
        lockfile for lockfile in environment_lock.lockfiles if lockfile.path == target_path
    )

    with pytest.raises(ValueError, match="lockfile sha256 must be a lowercase SHA-256 digest"):
        replace(target, sha256="")


def test_dependency_environment_markdown_escapes_profile_cells() -> None:
    """The public renderer must neutralise newlines and table separators."""
    profile = replace(
        _valid_profile(),
        lockfile_paths=("locks/runtime\nvariant|cpu.txt",),
    )
    environment_map = DifferentiableDependencyEnvironmentMap(
        schema="test-dependency-environment-map",
        artifact_id="test-dependency-environment-map",
        profiles=(profile,),
        environment_ready=True,
        ready_profile_count=1,
        total_profile_count=1,
        evidence_records=build_differentiable_dependency_environment_evidence(),
        ready_evidence_count=sum(
            1
            for record in build_differentiable_dependency_environment_evidence()
            if record.environment_ready
        ),
        total_evidence_count=len(build_differentiable_dependency_environment_evidence()),
        claim_boundary=DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_MAP_CLAIM_BOUNDARY,
    )

    markdown = render_differentiable_dependency_environment_map_markdown(environment_map)

    assert "locks/runtime variant\\|cpu.txt" in markdown
