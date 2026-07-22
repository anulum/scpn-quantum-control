# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable dependency environment map.
"""Dependency and environment evidence map for differentiable-programming governance."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .differentiable_claim_ledger import REPO_ROOT
from .differentiable_dependency_environment_evidence import (
    REQUIRED_DEPENDENCY_ENVIRONMENT_EVIDENCE_CLASSIFICATIONS,
    REQUIRED_DEPENDENCY_ENVIRONMENT_EVIDENCE_IDS,
    DifferentiableDependencyEnvironmentEvidence,
    build_differentiable_dependency_environment_evidence,
)
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
    "scpn_qc_differentiable_dependency_environment_map_v2"
)
DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_MAP_ARTIFACT_ID = "diff-dependency-environment-map-20260627"
DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_MAP_CLAIM_BOUNDARY = (
    "Differentiable dependency and environment evidence map only; no dependency "
    "or benchmark promotion, framework parity promotion, provider execution, "
    "hardware execution, GPU execution, Enzyme promotion, or isolated benchmark "
    "claim is implied."
)
DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_VALIDATION_CLAIM_BOUNDARY = (
    "Dependency-environment validation only; validates lockfile paths, "
    "checksums, pinned-package counts, and hard-gap blockers without "
    "promoting framework, Enzyme, provider, hardware, GPU, or isolated "
    "benchmark claims."
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
    """Describe one differentiable profile tied to lockfile evidence.

    Parameters
    ----------
    profile_id : DifferentiableDependencyEnvironmentProfileId or str
        Stable identifier for the runtime or verification profile.
    title : str
        Reviewer-facing profile title.
    role : str
        Purpose of the profile within differentiable validation.
    lockfile_paths : tuple[str, ...]
        Repository-relative lockfiles that define the environment.
    evidence_paths : tuple[str, ...]
        Repository-relative files reviewers must be able to inspect.
    pinned_package_count : int
        Total pinned package entries across the profile lockfiles.
    checksum_count : int
        Number of profile lockfiles carrying a non-empty checksum.
    evidence_status : DifferentiableDependencyEnvironmentStatus
        Whether the profile is locked or remains a hard gap.
    blockers : tuple[str, ...]
        Explicit reasons the profile cannot support promotion.
    claim_boundary : str
        Non-promotional interpretation attached to the evidence.

    """

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
        """Validate profile fields before emitting dependency evidence.

        Raises
        ------
        ValueError
            If a required text or path is empty, a count is negative, or the
            evidence status is outside the locked/hard-gap contract.

        """
        for field_name in ("profile_id", "title", "role"):
            _require_nonblank_text(getattr(self, field_name), field_name)
        _require_string_tuple(self.lockfile_paths, "lockfile_paths", require_nonempty=True)
        _require_string_tuple(self.evidence_paths, "evidence_paths", require_nonempty=True)
        _require_nonnegative_int(self.pinned_package_count, "pinned_package_count")
        _require_nonnegative_int(self.checksum_count, "checksum_count")
        _require_string_tuple(self.blockers, "blockers", require_nonempty=False)
        if self.evidence_status not in {"locked", "hard_gap"}:
            raise ValueError("evidence_status must be locked or hard_gap")
        if self.evidence_status == "locked" and self.blockers:
            raise ValueError("locked profiles must not carry blockers")
        if self.evidence_status == "hard_gap" and not self.blockers:
            raise ValueError("hard_gap profiles must list blockers")
        if self.checksum_count > len(self.lockfile_paths):
            raise ValueError("checksum_count must not exceed lockfile count")
        if self.claim_boundary != DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_MAP_CLAIM_BOUNDARY:
            raise ValueError("claim_boundary must match the canonical dependency boundary")

    @property
    def environment_ready(self) -> bool:
        """Return whether this dependency profile can support promotion.

        Returns
        -------
        bool
            ``True`` only for a locked profile without blockers.

        """
        return self.evidence_status == "locked" and not self.blockers

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready dependency environment profile.

        Returns
        -------
        dict[str, object]
            Profile fields with tuple values materialised as JSON-ready lists.

        """
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
    """Aggregate deterministic dependency profiles and environment evidence.

    Parameters
    ----------
    schema : str
        Versioned schema identifier for the emitted map.
    artifact_id : str
        Stable identifier for the committed evidence artefact.
    profiles : tuple[DifferentiableDependencyEnvironmentProfile, ...]
        Ordered dependency profiles governed by the map.
    environment_ready : bool
        Whether every profile and the underlying lock permit promotion.
    ready_profile_count : int
        Number of profiles whose locked evidence has no blockers.
    total_profile_count : int
        Number of profiles represented in the map.
    evidence_records : tuple[DifferentiableDependencyEnvironmentEvidence, ...]
        Ordered version-pin and execution-route evidence inventory.
    ready_evidence_count : int
        Number of evidence rows whose cited sources are locked.
    total_evidence_count : int
        Number of evidence rows represented in the map.
    claim_boundary : str
        Non-promotional interpretation attached to the map.

    """

    schema: str
    artifact_id: str
    profiles: tuple[DifferentiableDependencyEnvironmentProfile, ...]
    environment_ready: bool
    ready_profile_count: int
    total_profile_count: int
    evidence_records: tuple[DifferentiableDependencyEnvironmentEvidence, ...]
    ready_evidence_count: int
    total_evidence_count: int
    claim_boundary: str

    def __post_init__(self) -> None:
        """Reject malformed aggregate evidence before validation or rendering."""
        _require_nonblank_text(self.schema, "schema")
        _require_nonblank_text(self.artifact_id, "artifact_id")
        if type(self.profiles) is not tuple or not self.profiles:
            raise ValueError("profiles must be a non-empty tuple")
        if any(
            not isinstance(profile, DifferentiableDependencyEnvironmentProfile)
            for profile in self.profiles
        ):
            raise ValueError("profiles must contain dependency environment profiles")
        profile_ids = tuple(profile.profile_id for profile in self.profiles)
        if len(profile_ids) != len(set(profile_ids)):
            raise ValueError("profiles must contain unique profile_id values")
        if type(self.environment_ready) is not bool:
            raise ValueError("environment_ready must be a bool")
        _require_nonnegative_int(self.ready_profile_count, "ready_profile_count")
        _require_nonnegative_int(self.total_profile_count, "total_profile_count")
        if type(self.evidence_records) is not tuple or not self.evidence_records:
            raise ValueError("evidence_records must be a non-empty tuple")
        if any(
            not isinstance(record, DifferentiableDependencyEnvironmentEvidence)
            for record in self.evidence_records
        ):
            raise ValueError("evidence_records must contain dependency environment evidence")
        evidence_ids = tuple(record.evidence_id for record in self.evidence_records)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence_records must contain unique evidence_id values")
        _require_nonnegative_int(self.ready_evidence_count, "ready_evidence_count")
        _require_nonnegative_int(self.total_evidence_count, "total_evidence_count")
        if self.claim_boundary != DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_MAP_CLAIM_BOUNDARY:
            raise ValueError("claim_boundary must match the canonical dependency boundary")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready dependency environment map.

        Returns
        -------
        dict[str, object]
            Map metadata and nested JSON-ready profile dictionaries.

        """
        return {
            "schema": self.schema,
            "artifact_id": self.artifact_id,
            "environment_ready": self.environment_ready,
            "ready_profile_count": self.ready_profile_count,
            "total_profile_count": self.total_profile_count,
            "ready_evidence_count": self.ready_evidence_count,
            "total_evidence_count": self.total_evidence_count,
            "claim_boundary": self.claim_boundary,
            "profiles": [profile.to_dict() for profile in self.profiles],
            "evidence_records": [record.to_dict() for record in self.evidence_records],
        }


@dataclass(frozen=True)
class DifferentiableDependencyEnvironmentMapValidation:
    """Record validation evidence for a dependency environment map.

    Parameters
    ----------
    passed : bool
        Whether every structural and filesystem invariant passed.
    errors : tuple[str, ...]
        Deterministic validation errors in discovery order.
    checked_profile_ids : tuple[str, ...]
        Profile identifiers encountered during validation.
    checked_evidence_ids : tuple[str, ...]
        Toolchain and execution-route identifiers encountered during validation.
    checked_paths : tuple[str, ...]
        Sorted repository-relative evidence paths checked.
    checked_lockfile_count : int
        Number of distinct lockfile paths in the environment lock.
    checked_pinned_package_count : int
        Aggregate pinned-package count in the environment lock.
    claim_boundary : str
        Non-promotional interpretation attached to validation evidence.

    """

    passed: bool
    errors: tuple[str, ...]
    checked_profile_ids: tuple[str, ...]
    checked_evidence_ids: tuple[str, ...]
    checked_paths: tuple[str, ...]
    checked_lockfile_count: int
    checked_pinned_package_count: int
    claim_boundary: str

    def __post_init__(self) -> None:
        """Reject malformed or internally contradictory validation evidence."""
        if type(self.passed) is not bool:
            raise ValueError("passed must be a bool")
        _require_string_tuple(
            self.errors,
            "errors",
            require_nonempty=False,
            require_unique=False,
        )
        _require_string_tuple(
            self.checked_profile_ids,
            "checked_profile_ids",
            require_nonempty=True,
        )
        _require_string_tuple(
            self.checked_evidence_ids,
            "checked_evidence_ids",
            require_nonempty=True,
        )
        _require_string_tuple(self.checked_paths, "checked_paths", require_nonempty=True)
        _require_nonnegative_int(self.checked_lockfile_count, "checked_lockfile_count")
        _require_nonnegative_int(
            self.checked_pinned_package_count,
            "checked_pinned_package_count",
        )
        if self.passed == bool(self.errors):
            raise ValueError("passed must be true exactly when errors is empty")
        if self.claim_boundary != DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_VALIDATION_CLAIM_BOUNDARY:
            raise ValueError("claim_boundary must match the canonical validation boundary")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready dependency-environment validation evidence.

        Returns
        -------
        dict[str, object]
            Validation metadata with tuple values materialised as lists.

        """
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "checked_profile_ids": list(self.checked_profile_ids),
            "checked_evidence_ids": list(self.checked_evidence_ids),
            "checked_paths": list(self.checked_paths),
            "checked_lockfile_count": self.checked_lockfile_count,
            "checked_pinned_package_count": self.checked_pinned_package_count,
            "claim_boundary": self.claim_boundary,
        }


def run_differentiable_dependency_environment_map(
    *,
    environment_lock: ExternalValidationEnvironmentLock | None = None,
) -> DifferentiableDependencyEnvironmentMap:
    """Build the dependency and environment map from lockfile evidence.

    Parameters
    ----------
    environment_lock : ExternalValidationEnvironmentLock, optional
        Prebuilt lock evidence. When omitted, the current repository lockfiles
        are summarised through the external-validation environment builder.

    Returns
    -------
    DifferentiableDependencyEnvironmentMap
        Ordered profile evidence and aggregate readiness without promotion.

    """
    loaded_lock = (
        build_external_validation_environment_lock()
        if environment_lock is None
        else environment_lock
    )
    profiles = _default_dependency_profiles(loaded_lock)
    evidence_records = build_differentiable_dependency_environment_evidence()
    ready_count = sum(1 for profile in profiles if profile.environment_ready)
    ready_evidence_count = sum(1 for record in evidence_records if record.environment_ready)
    return DifferentiableDependencyEnvironmentMap(
        schema=DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_MAP_SCHEMA,
        artifact_id=DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_MAP_ARTIFACT_ID,
        profiles=profiles,
        environment_ready=(
            ready_count == len(profiles)
            and ready_evidence_count == len(evidence_records)
            and loaded_lock.classification == "locked"
        ),
        ready_profile_count=ready_count,
        total_profile_count=len(profiles),
        evidence_records=evidence_records,
        ready_evidence_count=ready_evidence_count,
        total_evidence_count=len(evidence_records),
        claim_boundary=DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_MAP_CLAIM_BOUNDARY,
    )


def validate_differentiable_dependency_environment_map(
    environment_map: DifferentiableDependencyEnvironmentMap,
    *,
    environment_lock: ExternalValidationEnvironmentLock | None = None,
    repo_root: Path = REPO_ROOT,
) -> DifferentiableDependencyEnvironmentMapValidation:
    """Validate profiles, paths, checksums, and readiness invariants.

    Parameters
    ----------
    environment_map : DifferentiableDependencyEnvironmentMap
        Candidate map whose schema, ordering, counts, paths, and blockers are
        validated.
    environment_lock : ExternalValidationEnvironmentLock, optional
        Prebuilt environment lock. When omitted, lock evidence is rebuilt from
        ``repo_root``.
    repo_root : pathlib.Path, optional
        Repository root used to resolve every cited evidence path.

    Returns
    -------
    DifferentiableDependencyEnvironmentMapValidation
        Fail-closed validation evidence containing every discovered error.

    """
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
    lockfiles = {lockfile.path: lockfile for lockfile in loaded_lock.lockfiles}
    lockfile_paths = set(lockfiles)
    profile_ids = tuple(profile.profile_id for profile in environment_map.profiles)
    evidence_ids = tuple(record.evidence_id for record in environment_map.evidence_records)
    checked_paths: set[str] = {
        "data/differentiable_phase_qnode/differentiable_dependency_environment_map_20260627.md"
    }

    if environment_map.schema != DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_MAP_SCHEMA:
        errors.append(f"unexpected dependency-environment-map schema: {environment_map.schema}")
    if environment_map.artifact_id != DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_MAP_ARTIFACT_ID:
        errors.append(
            f"unexpected dependency-environment-map artifact_id: {environment_map.artifact_id}"
        )
    if environment_map.total_profile_count != len(environment_map.profiles):
        errors.append("total_profile_count does not match profile count")
    ready_count = sum(1 for profile in environment_map.profiles if profile.environment_ready)
    if environment_map.ready_profile_count != ready_count:
        errors.append("ready_profile_count does not match ready profiles")
    ready_evidence_count = sum(
        1 for record in environment_map.evidence_records if record.environment_ready
    )
    expected_ready = (
        ready_count == len(environment_map.profiles)
        and ready_evidence_count == len(environment_map.evidence_records)
        and loaded_lock.classification == "locked"
    )
    if environment_map.environment_ready != expected_ready:
        errors.append("environment_ready does not match profile and lock readiness")
    if profile_ids != tuple(REQUIRED_DEPENDENCY_ENVIRONMENT_PROFILE_IDS):
        errors.append(
            "dependency environment profile IDs must match "
            "REQUIRED_DEPENDENCY_ENVIRONMENT_PROFILE_IDS exactly"
        )
    if environment_map.ready_evidence_count != ready_evidence_count:
        errors.append("ready_evidence_count does not match ready evidence records")
    if environment_map.total_evidence_count != len(environment_map.evidence_records):
        errors.append("total_evidence_count does not match evidence record count")
    if evidence_ids != tuple(REQUIRED_DEPENDENCY_ENVIRONMENT_EVIDENCE_IDS):
        errors.append(
            "dependency environment evidence IDs must match "
            "REQUIRED_DEPENDENCY_ENVIRONMENT_EVIDENCE_IDS exactly"
        )
    for profile in environment_map.profiles:
        if profile.evidence_status == "locked" and profile.pinned_package_count < 1:
            errors.append(f"{profile.profile_id}: locked profiles must record pinned packages")
        profile_lockfiles = [
            lockfiles[path] for path in profile.lockfile_paths if path in lockfiles
        ]
        expected_pinned_count = sum(
            lockfile.pinned_package_count for lockfile in profile_lockfiles
        )
        if profile.pinned_package_count != expected_pinned_count:
            errors.append(
                f"{profile.profile_id}: pinned_package_count does not match lockfile evidence"
            )
        if profile.checksum_count != len(profile_lockfiles):
            errors.append(f"{profile.profile_id}: checksum_count does not match lockfile evidence")
        for path in (*profile.lockfile_paths, *profile.evidence_paths):
            checked_paths.add(path)
            if path not in lockfile_paths:
                errors.append(f"{profile.profile_id}: path is not in environment lock: {path}")
            if not _is_contained_regular_file(repo_root, path):
                if not _is_contained_repo_path(repo_root, path):
                    errors.append(f"{profile.profile_id}: evidence path is unsafe: {path}")
                    continue
                errors.append(f"{profile.profile_id}: evidence path does not exist: {path}")

    for record in environment_map.evidence_records:
        expected_classification = REQUIRED_DEPENDENCY_ENVIRONMENT_EVIDENCE_CLASSIFICATIONS.get(
            record.evidence_id
        )
        if record.classification != expected_classification:
            errors.append(f"{record.evidence_id}: unexpected evidence classification")
        source_texts: list[str] = []
        for path, expected_sha256 in zip(
            record.evidence_paths,
            record.evidence_sha256,
            strict=True,
        ):
            checked_paths.add(path)
            if not _is_contained_regular_file(repo_root, path):
                if not _is_contained_repo_path(repo_root, path):
                    errors.append(f"{record.evidence_id}: evidence path is unsafe: {path}")
                    continue
                errors.append(f"{record.evidence_id}: evidence path does not exist: {path}")
                continue
            evidence_path = repo_root.resolve() / path
            payload = evidence_path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected_sha256:
                errors.append(f"{record.evidence_id}: evidence SHA-256 mismatch: {path}")
            source_texts.append(payload.decode("utf-8", errors="replace"))
        combined_sources = "\n".join(source_texts)
        for version_pin in record.version_pins:
            if not _version_pin_is_cited(version_pin, combined_sources):
                errors.append(f"{record.evidence_id}: uncited version pin: {version_pin}")

    for path in tuple(checked_paths):
        if path.endswith(".md") and not _is_contained_regular_file(repo_root, path):
            if not _is_contained_repo_path(repo_root, path):
                errors.append(f"dependency environment evidence path is unsafe: {path}")
                continue
            errors.append(f"dependency environment evidence path does not exist: {path}")

    return DifferentiableDependencyEnvironmentMapValidation(
        passed=not errors,
        errors=tuple(errors),
        checked_profile_ids=profile_ids,
        checked_evidence_ids=evidence_ids,
        checked_paths=tuple(sorted(checked_paths)),
        checked_lockfile_count=len(lockfile_paths),
        checked_pinned_package_count=sum(
            lockfile.pinned_package_count for lockfile in loaded_lock.lockfiles
        ),
        claim_boundary=DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_VALIDATION_CLAIM_BOUNDARY,
    )


def render_differentiable_dependency_environment_map_markdown(
    environment_map: DifferentiableDependencyEnvironmentMap,
) -> str:
    """Render a reviewer-facing Markdown summary of the dependency map.

    Parameters
    ----------
    environment_map : DifferentiableDependencyEnvironmentMap
        Dependency map to render without changing its readiness classification.

    Returns
    -------
    str
        SPDX-prefixed Markdown with aggregate readiness, profile rows, and the
        non-promotional claim boundary.

    """
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
        f"- Ready evidence rows: `{environment_map.ready_evidence_count}/{environment_map.total_evidence_count}`",
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
    lines.extend(
        (
            "| Evidence | Category | Classification | Status | Versions/constraints | Blockers |",
            "|---|---|---|---|---|---|",
        )
    )
    for record in environment_map.evidence_records:
        lines.append(
            "| `{evidence}` | `{category}` | `{classification}` | `{status}` | {pins} | {blockers} |".format(
                evidence=record.evidence_id,
                category=record.category,
                classification=record.classification,
                status=record.evidence_status,
                pins=_markdown_cell("<br>".join(record.version_pins) or "n/a"),
                blockers=_markdown_cell("<br>".join(record.blockers) or "none"),
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
    """Build the required ordered profiles from an environment lock.

    Parameters
    ----------
    environment_lock : ExternalValidationEnvironmentLock
        Lockfile summaries keyed by the repository paths required by each
        profile.

    Returns
    -------
    tuple[DifferentiableDependencyEnvironmentProfile, ...]
        Runtime, development, CI, framework-overlay, and Enzyme profiles in
        canonical order.

    """
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
    """Aggregate selected lockfiles into one validated profile.

    Parameters
    ----------
    profile_id : DifferentiableDependencyEnvironmentProfileId
        Canonical profile identifier.
    title : str
        Reviewer-facing profile title.
    role : str
        Purpose of the dependency environment.
    paths : tuple[str, ...]
        Ordered lockfile paths included in the profile.
    lockfiles : dict[str, EnvironmentLockfileSummary]
        Environment-lock summaries keyed by repository-relative path.
    status : DifferentiableDependencyEnvironmentStatus
        Locked or hard-gap evidence classification.
    blockers : tuple[str, ...]
        Explicit promotion blockers attached to the profile.

    Returns
    -------
    DifferentiableDependencyEnvironmentProfile
        Validated profile with aggregate pin and checksum counts.

    """
    pinned_count = 0
    for path in paths:
        lockfile = lockfiles[path]
        pinned_count += lockfile.pinned_package_count
    return DifferentiableDependencyEnvironmentProfile(
        profile_id=profile_id,
        title=title,
        role=role,
        lockfile_paths=paths,
        evidence_paths=paths,
        pinned_package_count=pinned_count,
        checksum_count=len(paths),
        evidence_status=status,
        blockers=blockers,
        claim_boundary=DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_MAP_CLAIM_BOUNDARY,
    )


def _require_nonblank_text(value: object, field_name: str) -> None:
    """Require an exact non-blank string without coercing caller input."""
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_string_tuple(
    value: object,
    field_name: str,
    *,
    require_nonempty: bool,
    require_unique: bool = True,
) -> None:
    """Require a tuple of unique, non-blank strings."""
    if type(value) is not tuple:
        raise ValueError(f"{field_name} must be a tuple")
    if require_nonempty and not value:
        raise ValueError(f"{field_name} must be non-empty")
    if any(type(item) is not str or not item.strip() for item in value):
        raise ValueError(f"{field_name} must contain non-empty strings")
    if require_unique and len(value) != len(set(value)):
        raise ValueError(f"{field_name} must contain unique entries")


def _require_nonnegative_int(value: object, field_name: str) -> None:
    """Require an exact non-negative integer, excluding booleans."""
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _is_safe_repo_relative_path(path: str) -> bool:
    """Return whether a path is canonical, relative, and lexically contained."""
    candidate = Path(path)
    return (
        path == path.strip()
        and "\\" not in path
        and not candidate.is_absolute()
        and candidate.as_posix() == path
        and ".." not in candidate.parts
    )


def _is_contained_regular_file(repo_root: Path, path: str) -> bool:
    """Return whether a safe path resolves to a regular file inside the root."""
    if not _is_contained_repo_path(repo_root, path):
        return False
    resolved_root = repo_root.resolve()
    resolved_path = (resolved_root / path).resolve()
    return resolved_path.is_file()


def _is_contained_repo_path(repo_root: Path, path: str) -> bool:
    """Return whether a canonical path resolves inside the repository root."""
    if not _is_safe_repo_relative_path(path):
        return False
    resolved_root = repo_root.resolve()
    return (resolved_root / path).resolve().is_relative_to(resolved_root)


def _version_pin_is_cited(version_pin: str, source_text: str) -> bool:
    """Return whether cited sources contain one declared version or constraint."""
    if version_pin in source_text:
        return True
    if "==" not in version_pin:
        return False
    package, version = version_pin.split("==", maxsplit=1)
    lowered_source = source_text.lower().replace("-", " ").replace("_", " ")
    package_tokens = package.lower().replace("-", " ").replace("_", " ").split()
    return (
        bool(version)
        and version in source_text
        and all(token in lowered_source for token in package_tokens)
    )


def _markdown_cell(value: str) -> str:
    """Escape one value for safe insertion into a Markdown table cell.

    Parameters
    ----------
    value : str
        Unescaped text that may contain newlines or table separators.

    Returns
    -------
    str
        Single-line text with vertical bars escaped.

    """
    return value.replace("\n", " ").replace("|", "\\|")


__all__ = [
    "DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_MAP_ARTIFACT_ID",
    "DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_MAP_CLAIM_BOUNDARY",
    "DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_MAP_SCHEMA",
    "DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_VALIDATION_CLAIM_BOUNDARY",
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
