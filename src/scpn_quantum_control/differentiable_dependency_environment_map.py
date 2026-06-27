# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable dependency environment map.
"""Dependency and environment evidence map for differentiable-programming governance."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .differentiable_claim_ledger import REPO_ROOT
from .differentiable_external_validation import (
    EnvironmentLockfileSummary,
    ExternalValidationEnvironmentLock,
    build_external_validation_environment_lock,
    validate_external_validation_environment_lock,
)

DifferentiableDependencyEnvironmentProfileId = Literal[
    "runtime_baseline",
    "development_verification",
    "ci_python_matrix",
    "framework_overlay_cpu",
    "enzyme_runner_py39",
]
DifferentiableDependencyEnvironmentStatus = Literal["locked", "hard_gap"]

DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_MAP_SCHEMA = (
    "scpn_qc_differentiable_dependency_environment_map_v1"
)
DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_MAP_ARTIFACT_ID = "diff-dependency-environment-map-20260627"
DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_MAP_CLAIM_BOUNDARY = (
    "Differentiable dependency and environment evidence map only; no dependency "
    "or benchmark promotion, framework parity promotion, provider execution, "
    "hardware execution, GPU execution, Enzyme promotion, or isolated benchmark "
    "claim is implied."
)
REQUIRED_DEPENDENCY_ENVIRONMENT_PROFILE_IDS: tuple[
    DifferentiableDependencyEnvironmentProfileId, ...
] = (
    "runtime_baseline",
    "development_verification",
    "ci_python_matrix",
    "framework_overlay_cpu",
    "enzyme_runner_py39",
)


@dataclass(frozen=True)
class DifferentiableDependencyEnvironmentProfile:
    """One differentiable dependency profile tied to lockfile evidence."""

    profile_id: DifferentiableDependencyEnvironmentProfileId | str
    title: str
    role: str
    lockfile_paths: tuple[str, ...]
    evidence_paths: tuple[str, ...]
    pinned_package_count: int
    checksum_count: int
    evidence_status: DifferentiableDependencyEnvironmentStatus
    blockers: tuple[str, ...]
    claim_boundary: str

    def __post_init__(self) -> None:
        """Validate profile fields before emitting dependency evidence."""
        for field_name in ("profile_id", "title", "role", "evidence_status", "claim_boundary"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")
        for field_name in ("lockfile_paths", "evidence_paths"):
            value = getattr(self, field_name)
            if not value or any(not str(item).strip() for item in value):
                raise ValueError(f"{field_name} must contain non-empty entries")
        if self.pinned_package_count < 0:
            raise ValueError("pinned_package_count must be non-negative")
        if self.checksum_count < 0:
            raise ValueError("checksum_count must be non-negative")
        if self.evidence_status not in {"locked", "hard_gap"}:
            raise ValueError("evidence_status must be locked or hard_gap")

    @property
    def environment_ready(self) -> bool:
        """Return whether this dependency profile can support promotion."""
        return self.evidence_status == "locked" and not self.blockers

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready dependency environment profile."""
        return {
            "profile_id": self.profile_id,
            "title": self.title,
            "role": self.role,
            "lockfile_paths": list(self.lockfile_paths),
            "evidence_paths": list(self.evidence_paths),
            "pinned_package_count": self.pinned_package_count,
            "checksum_count": self.checksum_count,
            "evidence_status": self.evidence_status,
            "blockers": list(self.blockers),
            "environment_ready": self.environment_ready,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class DifferentiableDependencyEnvironmentMap:
    """Deterministic map of dependency profiles and environment evidence."""

    schema: str
    artifact_id: str
    profiles: tuple[DifferentiableDependencyEnvironmentProfile, ...]
    environment_ready: bool
    ready_profile_count: int
    total_profile_count: int
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready dependency environment map."""
        return {
            "schema": self.schema,
            "artifact_id": self.artifact_id,
            "environment_ready": self.environment_ready,
            "ready_profile_count": self.ready_profile_count,
            "total_profile_count": self.total_profile_count,
            "claim_boundary": self.claim_boundary,
            "profiles": [profile.to_dict() for profile in self.profiles],
        }


@dataclass(frozen=True)
class DifferentiableDependencyEnvironmentMapValidation:
    """Validation result for a differentiable dependency environment map."""

    passed: bool
    errors: tuple[str, ...]
    checked_profile_ids: tuple[str, ...]
    checked_paths: tuple[str, ...]
    checked_lockfile_count: int
    checked_pinned_package_count: int
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready dependency-environment validation evidence."""
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "checked_profile_ids": list(self.checked_profile_ids),
            "checked_paths": list(self.checked_paths),
            "checked_lockfile_count": self.checked_lockfile_count,
            "checked_pinned_package_count": self.checked_pinned_package_count,
            "claim_boundary": self.claim_boundary,
        }


def run_differentiable_dependency_environment_map(
    *,
    environment_lock: ExternalValidationEnvironmentLock | None = None,
) -> DifferentiableDependencyEnvironmentMap:
    """Build the dependency and environment map from committed lockfile evidence."""
    loaded_lock = (
        build_external_validation_environment_lock()
        if environment_lock is None
        else environment_lock
    )
    profiles = _default_dependency_profiles(loaded_lock)
    ready_count = sum(1 for profile in profiles if profile.environment_ready)
    return DifferentiableDependencyEnvironmentMap(
        schema=DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_MAP_SCHEMA,
        artifact_id=DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_MAP_ARTIFACT_ID,
        profiles=profiles,
        environment_ready=ready_count == len(profiles) and loaded_lock.classification == "locked",
        ready_profile_count=ready_count,
        total_profile_count=len(profiles),
        claim_boundary=DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_MAP_CLAIM_BOUNDARY,
    )


def validate_differentiable_dependency_environment_map(
    environment_map: DifferentiableDependencyEnvironmentMap,
    *,
    environment_lock: ExternalValidationEnvironmentLock | None = None,
    repo_root: Path = REPO_ROOT,
) -> DifferentiableDependencyEnvironmentMapValidation:
    """Validate dependency profiles, paths, checksums, and readiness invariants."""
    loaded_lock = (
        build_external_validation_environment_lock(repo_root=repo_root)
        if environment_lock is None
        else environment_lock
    )
    lock_validation = validate_external_validation_environment_lock(
        loaded_lock,
        repo_root=repo_root,
    )
    errors = [f"environment-lock validation failed: {error}" for error in lock_validation.errors]
    lockfile_paths = {lockfile.path for lockfile in loaded_lock.lockfiles}
    profile_ids = tuple(str(profile.profile_id) for profile in environment_map.profiles)
    checked_paths: set[str] = {
        "data/differentiable_phase_qnode/differentiable_dependency_environment_map_20260627.md"
    }

    if environment_map.schema != DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_MAP_SCHEMA:
        errors.append(f"unexpected dependency-environment-map schema: {environment_map.schema}")
    if environment_map.total_profile_count != len(environment_map.profiles):
        errors.append("total_profile_count does not match profile count")
    ready_count = sum(1 for profile in environment_map.profiles if profile.environment_ready)
    if environment_map.ready_profile_count != ready_count:
        errors.append("ready_profile_count does not match ready profiles")
    expected_ready = (
        ready_count == len(environment_map.profiles) and loaded_lock.classification == "locked"
    )
    if environment_map.environment_ready != expected_ready:
        errors.append("environment_ready does not match profile and lock readiness")
    if profile_ids != tuple(REQUIRED_DEPENDENCY_ENVIRONMENT_PROFILE_IDS):
        errors.append(
            "dependency environment profile IDs must match "
            "REQUIRED_DEPENDENCY_ENVIRONMENT_PROFILE_IDS exactly"
        )
    for profile_id in _duplicates(profile_ids):
        errors.append(f"duplicate dependency environment profile_id: {profile_id}")

    for profile in environment_map.profiles:
        if profile.evidence_status == "locked" and profile.pinned_package_count < 1:
            errors.append(f"{profile.profile_id}: locked profiles must record pinned packages")
        if environment_map.environment_ready and profile.blockers:
            errors.append(
                f"{profile.profile_id}: ready environment profiles must not carry blockers"
            )
        for path in (*profile.lockfile_paths, *profile.evidence_paths):
            checked_paths.add(path)
            if path not in lockfile_paths:
                errors.append(f"{profile.profile_id}: path is not in environment lock: {path}")
            if not (repo_root / path).exists():
                errors.append(f"{profile.profile_id}: evidence path does not exist: {path}")

    for path in tuple(checked_paths):
        if path.endswith(".md") and not (repo_root / path).exists():
            errors.append(f"dependency environment evidence path does not exist: {path}")

    return DifferentiableDependencyEnvironmentMapValidation(
        passed=not errors,
        errors=tuple(errors),
        checked_profile_ids=profile_ids,
        checked_paths=tuple(sorted(checked_paths)),
        checked_lockfile_count=len(lockfile_paths),
        checked_pinned_package_count=sum(
            lockfile.pinned_package_count for lockfile in loaded_lock.lockfiles
        ),
        claim_boundary=(
            "Dependency-environment validation only; validates lockfile paths, "
            "checksums, pinned-package counts, and hard-gap blockers without "
            "promoting framework, Enzyme, provider, hardware, GPU, or isolated "
            "benchmark claims."
        ),
    )


def render_differentiable_dependency_environment_map_markdown(
    environment_map: DifferentiableDependencyEnvironmentMap,
) -> str:
    """Render a reviewer-facing Markdown summary of the dependency map."""
    lines = [
        "<!--",
        "SPDX-License-Identifier: AGPL-3.0-or-later",
        "Commercial license available",
        "© Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "© Code 2020–2026 Miroslav Šotek. All rights reserved.",
        "ORCID: 0009-0009-3560-0851",
        "Contact: www.anulum.li | protoscience@anulum.li",
        "SCPN Quantum Control — Differentiable Dependency and Environment Evidence Map",
        "-->",
        "",
        "# Differentiable Dependency and Environment Evidence Map",
        "",
        f"- Schema: `{environment_map.schema}`",
        f"- Artifact ID: `{environment_map.artifact_id}`",
        f"- Environment ready: `{environment_map.environment_ready}`",
        f"- Ready profiles: `{environment_map.ready_profile_count}/{environment_map.total_profile_count}`",
        f"- Claim boundary: {environment_map.claim_boundary}",
        "",
        "| Profile | Status | Lockfiles | Pinned packages | Blockers |",
        "|---|---|---|---|---|",
    ]
    for profile in environment_map.profiles:
        lines.append(
            "| `{profile}` | `{status}` | {lockfiles} | {pinned} | {blockers} |".format(
                profile=profile.profile_id,
                status=profile.evidence_status,
                lockfiles=_markdown_cell("<br>".join(profile.lockfile_paths)),
                pinned=profile.pinned_package_count,
                blockers=_markdown_cell("<br>".join(profile.blockers) or "none"),
            )
        )
    lines.append("")
    lines.append(
        "This map is dependency and environment evidence only. It does not promote "
        "framework parity, Enzyme/MLIR parity, provider execution, hardware "
        "execution, GPU execution, performance, or isolated benchmark claims."
    )
    lines.append("")
    return "\n".join(lines)


def _default_dependency_profiles(
    environment_lock: ExternalValidationEnvironmentLock,
) -> tuple[DifferentiableDependencyEnvironmentProfile, ...]:
    lockfiles = {lockfile.path: lockfile for lockfile in environment_lock.lockfiles}
    return (
        _profile(
            profile_id="runtime_baseline",
            title="Runtime baseline",
            role="Runtime dependency baseline for bounded differentiable imports.",
            paths=("requirements.txt",),
            lockfiles=lockfiles,
            status="locked",
            blockers=(),
        ),
        _profile(
            profile_id="development_verification",
            title="Development verification",
            role="Developer verification lock input for local quality gates.",
            paths=("requirements-dev.txt",),
            lockfiles=lockfiles,
            status="locked",
            blockers=(),
        ),
        _profile(
            profile_id="ci_python_matrix",
            title="CI Python matrix",
            role="Cross-platform smoke and Linux Python 3.11/3.12/3.13 CI locks.",
            paths=(
                "requirements-ci-cross-platform-smoke.txt",
                "requirements-ci-py311-linux.txt",
                "requirements-ci-py312-linux.txt",
                "requirements-ci-py313-linux.txt",
            ),
            lockfiles=lockfiles,
            status="locked",
            blockers=(),
        ),
        _profile(
            profile_id="framework_overlay_cpu",
            title="CPU framework overlay",
            role="CPU-only framework overlay freeze for JAX, PyTorch, TensorFlow, and PennyLane rows.",
            paths=(
                "data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/"
                "framework_overlay_freeze.txt",
            ),
            lockfiles=lockfiles,
            status="locked",
            blockers=(),
        ),
        _profile(
            profile_id="enzyme_runner_py39",
            title="Python 3.9 Enzyme runner",
            role="Optional Enzyme/JAX runner freeze for installed-toolchain hard-gap evidence.",
            paths=(
                "data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/"
                "enzyme_py39_freeze.txt",
            ),
            lockfiles=lockfiles,
            status="hard_gap",
            blockers=(
                "Enzyme/JAX runner lockfiles exist, but native Enzyme/LLVM/MLIR "
                "toolchain execution remains explicit hard-gap evidence until "
                "configured runner artefacts pass.",
            ),
        ),
    )


def _profile(
    *,
    profile_id: DifferentiableDependencyEnvironmentProfileId,
    title: str,
    role: str,
    paths: tuple[str, ...],
    lockfiles: dict[str, EnvironmentLockfileSummary],
    status: DifferentiableDependencyEnvironmentStatus,
    blockers: tuple[str, ...],
) -> DifferentiableDependencyEnvironmentProfile:
    pinned_count = 0
    checksum_count = 0
    for path in paths:
        lockfile = lockfiles[path]
        pinned_count += lockfile.pinned_package_count
        if lockfile.sha256:
            checksum_count += 1
    return DifferentiableDependencyEnvironmentProfile(
        profile_id=profile_id,
        title=title,
        role=role,
        lockfile_paths=paths,
        evidence_paths=paths,
        pinned_package_count=pinned_count,
        checksum_count=checksum_count,
        evidence_status=status,
        blockers=blockers,
        claim_boundary=DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_MAP_CLAIM_BOUNDARY,
    )


def _duplicates(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return tuple(sorted(duplicates))


def _markdown_cell(value: str) -> str:
    return value.replace("\n", " ").replace("|", "\\|")


__all__ = [
    "DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_MAP_ARTIFACT_ID",
    "DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_MAP_CLAIM_BOUNDARY",
    "DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_MAP_SCHEMA",
    "REQUIRED_DEPENDENCY_ENVIRONMENT_PROFILE_IDS",
    "DifferentiableDependencyEnvironmentMap",
    "DifferentiableDependencyEnvironmentMapValidation",
    "DifferentiableDependencyEnvironmentProfile",
    "DifferentiableDependencyEnvironmentProfileId",
    "DifferentiableDependencyEnvironmentStatus",
    "render_differentiable_dependency_environment_map_markdown",
    "run_differentiable_dependency_environment_map",
    "validate_differentiable_dependency_environment_map",
]
