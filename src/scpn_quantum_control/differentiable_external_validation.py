# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable external-validation package.
"""External-validation package manifests for differentiable evidence."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_VALIDATION_ENVIRONMENT_LOCK_SCHEMA = (
    "scpn_qc_differentiable_external_validation_environment_lock_v1"
)
DEFAULT_EXTERNAL_VALIDATION_ENVIRONMENT_LOCK_PATH = (
    REPO_ROOT
    / "data"
    / "differentiable_phase_qnode"
    / "external_validation_environment_lock_20260616.json"
)
DEFAULT_EXTERNAL_VALIDATION_ARTIFACT_BUNDLE_PATH = (
    REPO_ROOT
    / "data"
    / "differentiable_phase_qnode"
    / "external_validation_artifact_bundle_20260616.json"
)
DEFAULT_ENVIRONMENT_LOCK_INPUTS: tuple[tuple[str, str], ...] = (
    ("pyproject.toml", "Package metadata and bounded dependency ranges"),
    ("requirements.txt", "Runtime dependency lock input"),
    ("requirements-dev.txt", "Developer verification dependency lock input"),
    ("requirements-ci-cross-platform-smoke.txt", "Cross-platform smoke CI lockfile"),
    ("requirements-ci-py310-linux.txt", "Python 3.10 Linux CI lockfile"),
    ("requirements-ci-py311-linux.txt", "Python 3.11 Linux CI lockfile"),
    ("requirements-ci-py312-linux.txt", "Python 3.12 Linux CI lockfile"),
    ("requirements-ci-py313-linux.txt", "Python 3.13 Linux CI lockfile"),
    (
        "data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/"
        "framework_overlay_freeze.txt",
        "CPU framework overlay freeze used for JAX, PyTorch, TensorFlow, and PennyLane rows",
    ),
    (
        "data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/enzyme_py39_freeze.txt",
        "Python 3.9 Enzyme/JAX runner freeze used for installed-toolchain hard-gap evidence",
    ),
)
DEFAULT_ARTIFACT_BUNDLE_INPUTS: tuple[tuple[str, str], ...] = (
    ("data/differentiable_phase_qnode/claim_ledger.json", "Claim-ledger source of truth"),
    ("data/differentiable_phase_qnode/claim_ledger.md", "Reviewer claim-ledger summary"),
    (
        "data/differentiable_phase_qnode/public_claim_table_20260616.md",
        "Public-safe claim wording table",
    ),
    (
        "data/differentiable_phase_qnode/external_validation_environment_lock_20260616.json",
        "Exact environment-lock manifest",
    ),
    (
        "data/differentiable_phase_qnode/external_validation_environment_lock_20260616.md",
        "Environment-lock reviewer summary",
    ),
    (
        "data/differentiable_phase_qnode/domain_benchmark_dataset_closure_20260616.json",
        "Exact-answer domain dataset closure artefact",
    ),
    (
        "data/differentiable_phase_qnode/identical_circuit_gradient_comparison_20260616.json",
        "Identical-circuit Qiskit/PennyLane gradient comparison artefact",
    ),
    (
        "data/differentiable_phase_qnode/torch_maturity_audit_20260616.json",
        "PyTorch maturity audit artefact",
    ),
    (
        "data/differentiable_phase_qnode/enzyme_mlir_maturity_audit_20260616.json",
        "Enzyme/MLIR maturity audit artefact with runtime-gap and correctness evidence",
    ),
    (
        "data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/README.md",
        "Local benchmark evidence README and hardware boundary",
    ),
    (
        "data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/"
        "diff-qnode-ci-evidence-schema-v1.json",
        "Functional non-isolated CI evidence bundle",
    ),
    (
        "data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/"
        "diff-qnode-external-comparison.json",
        "Functional non-isolated external comparison rows",
    ),
    (
        "data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/phase_qnode_affinity.json",
        "Phase-QNode affinity benchmark artefact",
    ),
    (
        "data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/host_readiness.json",
        "Host-readiness blocker metadata",
    ),
)
PINNED_REQUIREMENT_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?==[^#;\s]+")


@dataclass(frozen=True)
class EnvironmentLockfileSummary:
    """Checksum and package-entry summary for one lockfile."""

    path: str
    role: str
    sha256: str
    size_bytes: int
    line_count: int
    pinned_package_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready lockfile summary."""
        return {
            "path": self.path,
            "role": self.role,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "line_count": self.line_count,
            "pinned_package_count": self.pinned_package_count,
        }


@dataclass(frozen=True)
class ExternalValidationEnvironmentLock:
    """Exact lockfile manifest for external differentiable validation."""

    artifact_id: str
    schema: str
    python_version: str
    platform: str
    lockfiles: tuple[EnvironmentLockfileSummary, ...]
    classification: str
    claim_boundary: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready external-validation environment manifest."""
        return {
            "schema": self.schema,
            "artifact_id": self.artifact_id,
            "python_version": self.python_version,
            "platform": self.platform,
            "classification": self.classification,
            "claim_boundary": self.claim_boundary,
            "lockfiles": [lockfile.to_dict() for lockfile in self.lockfiles],
        }


@dataclass(frozen=True)
class ExternalValidationEnvironmentLockValidation:
    """Validation result for external-validation environment lockfiles."""

    passed: bool
    errors: tuple[str, ...]
    checked_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready validation metadata."""
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "checked_paths": list(self.checked_paths),
        }


@dataclass(frozen=True)
class ExternalValidationArtifactEntry:
    """Checksum summary for one external-validation package artefact."""

    path: str
    role: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready artefact entry."""
        return {
            "path": self.path,
            "role": self.role,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class ExternalValidationArtifactBundle:
    """Reproducible manifest over committed differentiable validation artefacts."""

    artifact_id: str
    schema: str
    entries: tuple[ExternalValidationArtifactEntry, ...]
    classification: str
    claim_boundary: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready artefact-bundle manifest."""
        return {
            "schema": self.schema,
            "artifact_id": self.artifact_id,
            "classification": self.classification,
            "claim_boundary": self.claim_boundary,
            "entries": [entry.to_dict() for entry in self.entries],
        }


def summarize_environment_lockfile(
    path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    role: str,
) -> EnvironmentLockfileSummary:
    """Summarize one repository-relative lockfile with a SHA-256 digest."""
    resolved = path if path.is_absolute() else repo_root / path
    if not resolved.exists():
        raise FileNotFoundError(f"environment lockfile is missing: {path}")
    data = resolved.read_bytes()
    text = data.decode("utf-8")
    rel_path = resolved.relative_to(repo_root).as_posix()
    pinned_count = sum(
        1 for line in text.splitlines() if PINNED_REQUIREMENT_PATTERN.match(line.strip())
    )
    return EnvironmentLockfileSummary(
        path=rel_path,
        role=role,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        line_count=len(text.splitlines()),
        pinned_package_count=pinned_count,
    )


def summarize_artifact_entry(
    path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    role: str,
) -> ExternalValidationArtifactEntry:
    """Summarize one external-validation artefact with a SHA-256 digest."""
    resolved = path if path.is_absolute() else repo_root / path
    if not resolved.exists():
        raise FileNotFoundError(f"external-validation artefact is missing: {path}")
    data = resolved.read_bytes()
    return ExternalValidationArtifactEntry(
        path=resolved.relative_to(repo_root).as_posix(),
        role=role,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
    )


def build_external_validation_environment_lock(
    *,
    repo_root: Path = REPO_ROOT,
    artifact_id: str = "diff-external-validation-environment-lock-20260616",
    inputs: tuple[tuple[str, str], ...] = DEFAULT_ENVIRONMENT_LOCK_INPUTS,
) -> ExternalValidationEnvironmentLock:
    """Build the exact environment-lock manifest for the differentiable package."""
    summaries = tuple(
        summarize_environment_lockfile(repo_root / lockfile_path, repo_root=repo_root, role=role)
        for lockfile_path, role in inputs
    )
    return ExternalValidationEnvironmentLock(
        artifact_id=artifact_id,
        schema=EXTERNAL_VALIDATION_ENVIRONMENT_LOCK_SCHEMA,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        platform=platform.platform(),
        lockfiles=summaries,
        classification="functional_non_isolated",
        claim_boundary=(
            "Exact environment lockfile manifest for reviewer reproduction only; "
            "it does not promote performance, provider, QPU, GPU, hardware, or "
            "isolated_affinity benchmark claims."
        ),
    )


def build_external_validation_artifact_bundle(
    *,
    repo_root: Path = REPO_ROOT,
    artifact_id: str = "diff-external-validation-artifact-bundle-20260616",
    inputs: tuple[tuple[str, str], ...] = DEFAULT_ARTIFACT_BUNDLE_INPUTS,
) -> ExternalValidationArtifactBundle:
    """Build the reproducible external-validation artefact bundle manifest."""
    entries = tuple(
        summarize_artifact_entry(repo_root / artefact_path, repo_root=repo_root, role=role)
        for artefact_path, role in inputs
    )
    return ExternalValidationArtifactBundle(
        artifact_id=artifact_id,
        schema="scpn_qc_differentiable_external_validation_artifact_bundle_v1",
        entries=entries,
        classification="functional_non_isolated",
        claim_boundary=(
            "Reproducible artefact-bundle manifest only; it records committed "
            "evidence file checksums and does not promote performance, provider, "
            "QPU, GPU, hardware, or isolated_affinity benchmark claims."
        ),
    )


def load_external_validation_environment_lock(
    path: Path = DEFAULT_EXTERNAL_VALIDATION_ENVIRONMENT_LOCK_PATH,
) -> ExternalValidationEnvironmentLock:
    """Load a committed external-validation environment manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ExternalValidationEnvironmentLock(
        artifact_id=str(payload["artifact_id"]),
        schema=str(payload["schema"]),
        python_version=str(payload["python_version"]),
        platform=str(payload["platform"]),
        lockfiles=tuple(
            EnvironmentLockfileSummary(
                path=str(lockfile["path"]),
                role=str(lockfile["role"]),
                sha256=str(lockfile["sha256"]),
                size_bytes=int(lockfile["size_bytes"]),
                line_count=int(lockfile["line_count"]),
                pinned_package_count=int(lockfile["pinned_package_count"]),
            )
            for lockfile in payload["lockfiles"]
        ),
        classification=str(payload["classification"]),
        claim_boundary=str(payload["claim_boundary"]),
    )


def load_external_validation_artifact_bundle(
    path: Path = DEFAULT_EXTERNAL_VALIDATION_ARTIFACT_BUNDLE_PATH,
) -> ExternalValidationArtifactBundle:
    """Load a committed external-validation artefact-bundle manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ExternalValidationArtifactBundle(
        artifact_id=str(payload["artifact_id"]),
        schema=str(payload["schema"]),
        entries=tuple(
            ExternalValidationArtifactEntry(
                path=str(entry["path"]),
                role=str(entry["role"]),
                sha256=str(entry["sha256"]),
                size_bytes=int(entry["size_bytes"]),
            )
            for entry in payload["entries"]
        ),
        classification=str(payload["classification"]),
        claim_boundary=str(payload["claim_boundary"]),
    )


def validate_external_validation_environment_lock(
    manifest: ExternalValidationEnvironmentLock | None = None,
    *,
    repo_root: Path = REPO_ROOT,
    path: Path = DEFAULT_EXTERNAL_VALIDATION_ENVIRONMENT_LOCK_PATH,
) -> ExternalValidationEnvironmentLockValidation:
    """Validate that committed environment-lock hashes match repository files."""
    candidate = manifest or load_external_validation_environment_lock(path)
    errors: list[str] = []
    checked_paths: list[str] = []
    if candidate.schema != EXTERNAL_VALIDATION_ENVIRONMENT_LOCK_SCHEMA:
        errors.append(f"unexpected schema: {candidate.schema}")
    if candidate.classification != "functional_non_isolated":
        errors.append("environment lock manifest must remain functional_non_isolated")
    if "isolated_affinity benchmark claims" not in candidate.claim_boundary:
        errors.append("environment lock manifest claim boundary is not explicit enough")
    for lockfile in candidate.lockfiles:
        checked_paths.append(lockfile.path)
        resolved = repo_root / lockfile.path
        if not resolved.exists():
            errors.append(f"missing lockfile: {lockfile.path}")
            continue
        current = summarize_environment_lockfile(resolved, repo_root=repo_root, role=lockfile.role)
        if current.sha256 != lockfile.sha256:
            errors.append(f"sha256 mismatch: {lockfile.path}")
        if current.size_bytes != lockfile.size_bytes:
            errors.append(f"size mismatch: {lockfile.path}")
        if current.line_count != lockfile.line_count:
            errors.append(f"line-count mismatch: {lockfile.path}")
        if current.pinned_package_count != lockfile.pinned_package_count:
            errors.append(f"pinned-package-count mismatch: {lockfile.path}")
    return ExternalValidationEnvironmentLockValidation(
        passed=not errors,
        errors=tuple(errors),
        checked_paths=tuple(checked_paths),
    )


def validate_external_validation_artifact_bundle(
    bundle: ExternalValidationArtifactBundle | None = None,
    *,
    repo_root: Path = REPO_ROOT,
    path: Path = DEFAULT_EXTERNAL_VALIDATION_ARTIFACT_BUNDLE_PATH,
) -> ExternalValidationEnvironmentLockValidation:
    """Validate that the artefact-bundle manifest matches repository files."""
    candidate = bundle or load_external_validation_artifact_bundle(path)
    errors: list[str] = []
    checked_paths: list[str] = []
    if candidate.schema != "scpn_qc_differentiable_external_validation_artifact_bundle_v1":
        errors.append(f"unexpected schema: {candidate.schema}")
    if candidate.classification != "functional_non_isolated":
        errors.append("artifact bundle must remain functional_non_isolated")
    if "isolated_affinity benchmark claims" not in candidate.claim_boundary:
        errors.append("artifact bundle claim boundary is not explicit enough")
    for entry in candidate.entries:
        checked_paths.append(entry.path)
        resolved = repo_root / entry.path
        if not resolved.exists():
            errors.append(f"missing artefact: {entry.path}")
            continue
        current = summarize_artifact_entry(resolved, repo_root=repo_root, role=entry.role)
        if current.sha256 != entry.sha256:
            errors.append(f"sha256 mismatch: {entry.path}")
        if current.size_bytes != entry.size_bytes:
            errors.append(f"size mismatch: {entry.path}")
    return ExternalValidationEnvironmentLockValidation(
        passed=not errors,
        errors=tuple(errors),
        checked_paths=tuple(checked_paths),
    )


def render_external_validation_environment_lock_markdown(
    manifest: ExternalValidationEnvironmentLock,
) -> str:
    """Render a reviewer-facing Markdown summary for the lockfile manifest."""
    lines = [
        "<!--",
        "SPDX-License-Identifier: AGPL-3.0-or-later",
        "Commercial license available",
        "© Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "© Code 2020–2026 Miroslav Šotek. All rights reserved.",
        "ORCID: 0009-0009-3560-0851",
        "Contact: www.anulum.li | protoscience@anulum.li",
        "SCPN Quantum Control — Differentiable external-validation environment lock",
        "-->",
        "",
        "# Differentiable External-Validation Environment Lock",
        "",
        f"- Artefact ID: `{manifest.artifact_id}`",
        f"- Classification: `{manifest.classification}`",
        f"- Python: `{manifest.python_version}`",
        f"- Platform: `{manifest.platform}`",
        f"- Claim boundary: {manifest.claim_boundary}",
        "",
        "| Lockfile | Role | SHA-256 | Pinned packages |",
        "|---|---|---|---|",
    ]
    for lockfile in manifest.lockfiles:
        lines.append(
            f"| `{lockfile.path}` | {lockfile.role} | `{lockfile.sha256}` | "
            f"{lockfile.pinned_package_count} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_external_validation_artifact_bundle_markdown(
    bundle: ExternalValidationArtifactBundle,
) -> str:
    """Render a reviewer-facing Markdown summary for the artefact bundle."""
    lines = [
        "<!--",
        "SPDX-License-Identifier: AGPL-3.0-or-later",
        "Commercial license available",
        "© Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "© Code 2020–2026 Miroslav Šotek. All rights reserved.",
        "ORCID: 0009-0009-3560-0851",
        "Contact: www.anulum.li | protoscience@anulum.li",
        "SCPN Quantum Control — Differentiable external-validation artefact bundle",
        "-->",
        "",
        "# Differentiable External-Validation Artefact Bundle",
        "",
        f"- Artefact ID: `{bundle.artifact_id}`",
        f"- Classification: `{bundle.classification}`",
        f"- Claim boundary: {bundle.claim_boundary}",
        "",
        "| Artefact | Role | SHA-256 | Size bytes |",
        "|---|---|---|---:|",
    ]
    for entry in bundle.entries:
        lines.append(f"| `{entry.path}` | {entry.role} | `{entry.sha256}` | {entry.size_bytes} |")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_ENVIRONMENT_LOCK_INPUTS",
    "DEFAULT_ARTIFACT_BUNDLE_INPUTS",
    "DEFAULT_EXTERNAL_VALIDATION_ARTIFACT_BUNDLE_PATH",
    "DEFAULT_EXTERNAL_VALIDATION_ENVIRONMENT_LOCK_PATH",
    "EXTERNAL_VALIDATION_ENVIRONMENT_LOCK_SCHEMA",
    "EnvironmentLockfileSummary",
    "ExternalValidationArtifactBundle",
    "ExternalValidationArtifactEntry",
    "ExternalValidationEnvironmentLock",
    "ExternalValidationEnvironmentLockValidation",
    "build_external_validation_artifact_bundle",
    "build_external_validation_environment_lock",
    "load_external_validation_artifact_bundle",
    "load_external_validation_environment_lock",
    "render_external_validation_artifact_bundle_markdown",
    "render_external_validation_environment_lock_markdown",
    "summarize_artifact_entry",
    "summarize_environment_lockfile",
    "validate_external_validation_artifact_bundle",
    "validate_external_validation_environment_lock",
]
