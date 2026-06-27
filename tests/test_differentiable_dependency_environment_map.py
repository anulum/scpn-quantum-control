# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable dependency environment map tests
"""Tests for differentiable dependency and environment evidence governance."""

from __future__ import annotations

from pathlib import Path

import scpn_quantum_control as scpn
from scpn_quantum_control import (
    DifferentiableDependencyEnvironmentProfile,
    differentiable_api,
    differentiable_module_hardening_registry,
    render_differentiable_dependency_environment_map_markdown,
    run_differentiable_dependency_environment_map,
    validate_differentiable_dependency_environment_map,
)
from scpn_quantum_control.differentiable_external_validation import (
    build_external_validation_environment_lock,
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
