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
        "data/differentiable_phase_qnode/differentiable_support_surface_alignment_20260627.json",
        "Differentiable support-surface alignment rerun artefact",
    ),
    (
        "data/differentiable_phase_qnode/differentiable_support_surface_alignment_20260627.md",
        "Differentiable support-surface alignment reviewer summary",
    ),
    (
        "data/differentiable_phase_qnode/differentiable_baseline_scorecard_20260620.json",
        "Differentiable baseline category scorecard artefact",
    ),
    (
        "data/differentiable_phase_qnode/differentiable_baseline_scorecard_20260620.md",
        "Differentiable baseline category scorecard reviewer summary",
    ),
    (
        "data/differentiable_phase_qnode/differentiable_competitive_baseline_refresh_20260627.json",
        "Differentiable competitive-baseline freshness artefact",
    ),
    (
        "data/differentiable_phase_qnode/differentiable_competitive_baseline_refresh_20260627.md",
        "Differentiable competitive-baseline freshness reviewer summary",
    ),
    (
        "data/differentiable_phase_qnode/differentiable_rust_python_inventory_20260620.json",
        "Differentiable Rust/Python rustification surface inventory artefact",
    ),
    (
        "data/differentiable_phase_qnode/differentiable_rust_python_inventory_20260620.md",
        "Differentiable Rust/Python rustification surface inventory reviewer summary",
    ),
    (
        "data/differentiable_phase_qnode/differentiable_architecture_map_20260627.json",
        "Differentiable architecture and Rustification routing map artefact",
    ),
    (
        "data/differentiable_phase_qnode/differentiable_architecture_map_20260627.md",
        "Differentiable architecture and Rustification routing map reviewer summary",
    ),
    (
        "data/differentiable_phase_qnode/differentiable_dependency_environment_map_20260627.json",
        "Differentiable dependency and environment evidence map artefact",
    ),
    (
        "data/differentiable_phase_qnode/differentiable_dependency_environment_map_20260627.md",
        "Differentiable dependency and environment evidence map reviewer summary",
    ),
    (
        "data/differentiable_phase_qnode/differentiable_isolated_benchmark_plan_20260627.json",
        "Reserved-host isolated benchmark batch plan artefact",
    ),
    (
        "data/differentiable_phase_qnode/differentiable_isolated_benchmark_plan_20260627.md",
        "Reserved-host isolated benchmark batch plan reviewer summary",
    ),
    (
        "data/differentiable_phase_qnode/provider_gradient_boundary_20260705.json",
        "No-submit provider-gradient boundary evidence artefact",
    ),
    (
        "data/differentiable_phase_qnode/provider_gradient_boundary_20260705.md",
        "No-submit provider-gradient boundary reviewer summary",
    ),
    (
        "data/differentiable_phase_qnode/compiler_evidence_boundary_20260705.json",
        "LLVM/JIT and Enzyme/MLIR compiler evidence boundary artefact",
    ),
    (
        "data/differentiable_phase_qnode/compiler_evidence_boundary_20260705.md",
        "LLVM/JIT and Enzyme/MLIR compiler evidence boundary reviewer summary",
    ),
    (
        "data/differentiable_phase_qnode/compiler_alias_activity_evidence_20260706.json",
        "Program AD compiler alias-activity evidence artefact",
    ),
    (
        "data/differentiable_phase_qnode/compiler_alias_activity_evidence_20260706.md",
        "Program AD compiler alias-activity evidence reviewer summary",
    ),
    (
        "data/differentiable_phase_qnode/compiler_promotion_batch_20260706.json",
        "Non-promotional compiler evidence promotion-batch artefact",
    ),
    (
        "data/differentiable_phase_qnode/compiler_promotion_batch_20260706.md",
        "Non-promotional compiler evidence promotion-batch reviewer summary",
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
        "data/differentiable_phase_qnode/enzyme_mlir_compiler_ad_breadth_artifact_20260706.json",
        "Enzyme/MLIR raw 11-case compiler-AD breadth artefact",
    ),
    (
        "data/differentiable_phase_qnode/enzyme_mlir_compiler_ad_breadth_artifact_20260706.md",
        "Enzyme/MLIR raw 11-case compiler-AD breadth reviewer summary",
    ),
    (
        "data/differentiable_phase_qnode/native_whole_program_ad_execution_evidence_20260622.json",
        "Native LLVM/JIT whole-program AD execution evidence",
    ),
    (
        "data/differentiable_phase_qnode/native_whole_program_ad_execution_evidence_20260622.md",
        "Native LLVM/JIT whole-program AD execution reviewer summary",
    ),
    (
        "data/differentiable_phase_qnode/llvm_jit_claim_gate_20260704.json",
        "Native LLVM/JIT promotion claim gate artefact",
    ),
    (
        "data/differentiable_phase_qnode/llvm_jit_claim_gate_20260704.md",
        "Native LLVM/JIT promotion claim gate reviewer summary",
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

    def __post_init__(self) -> None:
        """Validate immutable lockfile evidence without type coercion."""
        _require_nonblank(self.path, "lockfile path")
        _require_nonblank(self.role, "lockfile role")
        _require_sha256(self.sha256, "lockfile sha256")
        for field_name in ("size_bytes", "line_count", "pinned_package_count"):
            _require_nonnegative_int(getattr(self, field_name), f"lockfile {field_name}")

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

    def __post_init__(self) -> None:
        """Validate environment-manifest identity and unique lockfile paths."""
        for field_name in (
            "artifact_id",
            "schema",
            "python_version",
            "platform",
            "classification",
            "claim_boundary",
        ):
            _require_nonblank(getattr(self, field_name), f"environment manifest {field_name}")
        _require_typed_tuple(
            self.lockfiles,
            EnvironmentLockfileSummary,
            "environment manifest lockfiles",
        )
        _require_unique_paths(
            tuple(lockfile.path for lockfile in self.lockfiles),
            "environment manifest lockfiles",
        )

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

    def __post_init__(self) -> None:
        """Validate pass/error coherence and checked-path metadata."""
        if type(self.passed) is not bool:
            raise ValueError("external-validation passed must be boolean")
        _require_string_tuple(self.errors, "external-validation errors")
        _require_string_tuple(self.checked_paths, "external-validation checked_paths")
        if self.passed == bool(self.errors):
            raise ValueError(
                "external-validation passed must be true exactly when errors are empty"
            )

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

    def __post_init__(self) -> None:
        """Validate immutable artefact evidence without type coercion."""
        _require_nonblank(self.path, "artefact path")
        _require_nonblank(self.role, "artefact role")
        _require_sha256(self.sha256, "artefact sha256")
        _require_nonnegative_int(self.size_bytes, "artefact size_bytes")

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

    def __post_init__(self) -> None:
        """Validate bundle identity and unique artefact paths."""
        for field_name in ("artifact_id", "schema", "classification", "claim_boundary"):
            _require_nonblank(getattr(self, field_name), f"artifact bundle {field_name}")
        _require_typed_tuple(
            self.entries,
            ExternalValidationArtifactEntry,
            "artifact bundle entries",
        )
        _require_unique_paths(
            tuple(entry.path for entry in self.entries),
            "artifact bundle entries",
        )

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
    resolved = _contained_evidence_file(
        path,
        repo_root=repo_root,
        context="environment lockfile",
    )
    data = resolved.read_bytes()
    text = data.decode("utf-8")
    rel_path = resolved.relative_to(repo_root.resolve()).as_posix()
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
    resolved = _contained_evidence_file(
        path,
        repo_root=repo_root,
        context="external-validation artefact",
    )
    data = resolved.read_bytes()
    return ExternalValidationArtifactEntry(
        path=resolved.relative_to(repo_root.resolve()).as_posix(),
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
    payload = _json_object(path, "environment manifest")
    lockfiles = _object_list(payload, "lockfiles", "environment manifest")
    return ExternalValidationEnvironmentLock(
        artifact_id=_string_field(payload, "artifact_id", "environment manifest"),
        schema=_string_field(payload, "schema", "environment manifest"),
        python_version=_string_field(payload, "python_version", "environment manifest"),
        platform=_string_field(payload, "platform", "environment manifest"),
        lockfiles=tuple(
            EnvironmentLockfileSummary(
                path=_string_field(lockfile, "path", "environment lockfile"),
                role=_string_field(lockfile, "role", "environment lockfile"),
                sha256=_string_field(lockfile, "sha256", "environment lockfile"),
                size_bytes=_integer_field(lockfile, "size_bytes", "environment lockfile"),
                line_count=_integer_field(lockfile, "line_count", "environment lockfile"),
                pinned_package_count=_integer_field(
                    lockfile,
                    "pinned_package_count",
                    "environment lockfile",
                ),
            )
            for lockfile in lockfiles
        ),
        classification=_string_field(payload, "classification", "environment manifest"),
        claim_boundary=_string_field(payload, "claim_boundary", "environment manifest"),
    )


def load_external_validation_artifact_bundle(
    path: Path = DEFAULT_EXTERNAL_VALIDATION_ARTIFACT_BUNDLE_PATH,
) -> ExternalValidationArtifactBundle:
    """Load a committed external-validation artefact-bundle manifest."""
    payload = _json_object(path, "artifact bundle")
    entries = _object_list(payload, "entries", "artifact bundle")
    return ExternalValidationArtifactBundle(
        artifact_id=_string_field(payload, "artifact_id", "artifact bundle"),
        schema=_string_field(payload, "schema", "artifact bundle"),
        entries=tuple(
            ExternalValidationArtifactEntry(
                path=_string_field(entry, "path", "artifact bundle entry"),
                role=_string_field(entry, "role", "artifact bundle entry"),
                sha256=_string_field(entry, "sha256", "artifact bundle entry"),
                size_bytes=_integer_field(entry, "size_bytes", "artifact bundle entry"),
            )
            for entry in entries
        ),
        classification=_string_field(payload, "classification", "artifact bundle"),
        claim_boundary=_string_field(payload, "claim_boundary", "artifact bundle"),
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
        try:
            current = summarize_environment_lockfile(
                Path(lockfile.path),
                repo_root=repo_root,
                role=lockfile.role,
            )
        except ValueError:
            errors.append(f"unsafe lockfile path: {lockfile.path}")
            continue
        except FileNotFoundError:
            errors.append(f"missing lockfile: {lockfile.path}")
            continue
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
        try:
            current = summarize_artifact_entry(
                Path(entry.path),
                repo_root=repo_root,
                role=entry.role,
            )
        except ValueError:
            errors.append(f"unsafe artefact path: {entry.path}")
            continue
        except FileNotFoundError:
            errors.append(f"missing artefact: {entry.path}")
            continue
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
    return "\n".join(lines)


def _require_nonblank(value: object, field_name: str) -> None:
    """Require an exact non-blank string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _contained_evidence_file(path: Path, *, repo_root: Path, context: str) -> Path:
    """Resolve a regular evidence file without permitting repository escape."""
    resolved_root = repo_root.resolve()
    candidate = (path if path.is_absolute() else resolved_root / path).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{context} escapes repository: {path}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"{context} is missing: {path}")
    return candidate


def _require_sha256(value: object, field_name: str) -> None:
    """Require one lowercase SHA-256 hexadecimal digest."""
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _require_nonnegative_int(value: object, field_name: str) -> None:
    """Require an exact non-negative integer, excluding booleans."""
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _require_typed_tuple(value: object, item_type: type[object], field_name: str) -> None:
    """Require an exact tuple containing only the requested runtime type."""
    if not isinstance(value, tuple) or any(not isinstance(item, item_type) for item in value):
        raise ValueError(f"{field_name} must be a tuple of {item_type.__name__}")


def _require_string_tuple(value: object, field_name: str) -> None:
    """Require an exact tuple containing non-blank strings."""
    if not isinstance(value, tuple) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{field_name} must contain non-empty strings")


def _require_unique_paths(paths: tuple[str, ...], field_name: str) -> None:
    """Reject ambiguous duplicate evidence identities."""
    if len(set(paths)) != len(paths):
        raise ValueError(f"{field_name} must have unique paths")


def _json_object(path: Path, context: str) -> dict[str, object]:
    """Load one JSON object without accepting scalar or list roots."""
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ValueError(f"{context} must be a JSON object with string keys")
    return payload


def _object_list(
    payload: dict[str, object],
    key: str,
    context: str,
) -> tuple[dict[str, object], ...]:
    """Read a required list of JSON objects with string keys."""
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{context} {key} must be a list")
    rows: list[dict[str, object]] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict) or any(not isinstance(item, str) for item in row):
            raise ValueError(f"{context} {key}[{index}] must be an object with string keys")
        rows.append(row)
    return tuple(rows)


def _string_field(payload: dict[str, object], key: str, context: str) -> str:
    """Read one required exact non-blank string field."""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} {key} must be a non-empty string")
    return value


def _integer_field(payload: dict[str, object], key: str, context: str) -> int:
    """Read one required exact non-negative integer field."""
    value = payload.get(key)
    if type(value) is not int or value < 0:
        raise ValueError(f"{context} {key} must be a non-negative integer")
    return value


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
