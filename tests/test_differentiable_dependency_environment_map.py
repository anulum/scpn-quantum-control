# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable dependency environment map tests
"""Tests for differentiable dependency and environment evidence governance."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

import scpn_quantum_control as scpn
from scpn_quantum_control import (
    DifferentiableDependencyEnvironmentMap,
    DifferentiableDependencyEnvironmentProfile,
    DifferentiableDependencyEnvironmentStatus,
    differentiable_api,
    differentiable_module_hardening_registry,
    render_differentiable_dependency_environment_map_markdown,
    run_differentiable_dependency_environment_map,
    validate_differentiable_dependency_environment_map,
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
    profile_id: str = "runtime_baseline",
    title: str = "Runtime baseline",
    role: str = "Test dependency environment.",
    lockfile_paths: tuple[str, ...] = ("requirements.txt",),
    evidence_paths: tuple[str, ...] = ("requirements.txt",),
    pinned_package_count: int = 1,
    checksum_count: int = 1,
    evidence_status: DifferentiableDependencyEnvironmentStatus = "locked",
    claim_boundary: str = "test-only dependency evidence",
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
        profile_id=profile_id,
        title=title,
        role=role,
        lockfile_paths=lockfile_paths,
        evidence_paths=evidence_paths,
        pinned_package_count=pinned_package_count,
        checksum_count=checksum_count,
        evidence_status=evidence_status,
        blockers=(),
        claim_boundary=claim_boundary,
    )


def test_dependency_environment_map_records_required_profiles() -> None:
    """The map must bind every differentiable runtime profile to lock evidence."""
    environment_map = run_differentiable_dependency_environment_map()

    assert environment_map.schema == "scpn_qc_differentiable_dependency_environment_map_v1"
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
        blockers=("unexpected blocker",),
        claim_boundary="test-only invalid profile",
    )
    environment_map = type(run_differentiable_dependency_environment_map())(
        schema="scpn_qc_differentiable_dependency_environment_map_v1",
        artifact_id="test-dependency-environment-map",
        profiles=(invalid_profile,),
        environment_ready=True,
        ready_profile_count=1,
        total_profile_count=1,
        claim_boundary="test-only invalid environment map",
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
    assert any(
        "ready environment profiles must not carry blockers" in error
        for error in validation.errors
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
            "profile_id must be non-empty",
            id="empty-profile-id",
        ),
        pytest.param(
            lambda: _profile_record(title=""),
            "title must be non-empty",
            id="empty-title",
        ),
        pytest.param(
            lambda: _profile_record(role="\t"),
            "role must be non-empty",
            id="empty-role",
        ),
        pytest.param(
            lambda: _profile_record(claim_boundary=""),
            "claim_boundary must be non-empty",
            id="empty-claim-boundary",
        ),
        pytest.param(
            lambda: _profile_record(lockfile_paths=()),
            "lockfile_paths must contain non-empty entries",
            id="empty-lockfile-paths",
        ),
        pytest.param(
            lambda: _profile_record(evidence_paths=("",)),
            "evidence_paths must contain non-empty entries",
            id="blank-evidence-path",
        ),
        pytest.param(
            lambda: _profile_record(pinned_package_count=-1),
            "pinned_package_count must be non-negative",
            id="negative-pinned-count",
        ),
        pytest.param(
            lambda: _profile_record(checksum_count=-1),
            "checksum_count must be non-negative",
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
    ],
)
def test_dependency_profile_rejects_invalid_public_fields(
    factory: Callable[[], DifferentiableDependencyEnvironmentProfile],
    message: str,
) -> None:
    """The public profile record must reject every malformed field family."""
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
    assert payload["checked_paths"] == list(validation.checked_paths)


def test_dependency_environment_validation_rejects_duplicate_profile_ids() -> None:
    """Validation must expose duplicate IDs instead of collapsing profiles."""
    environment_lock = build_external_validation_environment_lock()
    environment_map = run_differentiable_dependency_environment_map(
        environment_lock=environment_lock
    )
    profiles = environment_map.profiles
    duplicate = replace(profiles[1], profile_id=profiles[0].profile_id)
    invalid_map = replace(
        environment_map,
        profiles=(profiles[0], duplicate, *profiles[2:]),
    )

    validation = validate_differentiable_dependency_environment_map(
        invalid_map,
        environment_lock=environment_lock,
    )

    assert not validation.passed
    assert "duplicate dependency environment profile_id: runtime_baseline" in validation.errors
    assert any("profile IDs must match" in error for error in validation.errors)


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
        claim_boundary="test-only dependency evidence",
    )

    markdown = render_differentiable_dependency_environment_map_markdown(environment_map)

    assert "locks/runtime variant\\|cpu.txt" in markdown
