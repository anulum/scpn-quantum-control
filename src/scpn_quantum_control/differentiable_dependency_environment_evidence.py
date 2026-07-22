# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable environment evidence records.
"""Version-pin and execution-route evidence for differentiable environments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .differentiable_claim_ledger import REPO_ROOT

DifferentiableDependencyEnvironmentEvidenceCategory = Literal[
    "toolchain",
    "execution_route",
]
DifferentiableDependencyEnvironmentEvidenceStatus = Literal["locked", "hard_gap"]
DifferentiableDependencyEnvironmentEvidenceId = Literal[
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
]

DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_EVIDENCE_CLAIM_BOUNDARY = (
    "Version-pin and execution-route evidence only; locked rows prove cited "
    "dependency provenance or bounded local execution, while hard-gap rows "
    "remain non-promotional until their named artefacts exist."
)
REQUIRED_DEPENDENCY_ENVIRONMENT_EVIDENCE_IDS: tuple[
    DifferentiableDependencyEnvironmentEvidenceId, ...
] = (
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
)
REQUIRED_DEPENDENCY_ENVIRONMENT_EVIDENCE_CLASSIFICATIONS: dict[str, str] = {
    "python_versions": "locked_versions",
    "rust_crates": "locked_versions",
    "jax_cpu": "locked_versions",
    "pytorch_cpu": "locked_versions",
    "tensorflow_cpu": "locked_versions",
    "pennylane_cpu": "locked_versions",
    "qiskit": "locked_versions",
    "catalyst": "locked_versions",
    "enzyme_llvm_mlir": "locked_versions",
    "gpu_overlay": "declared_unlocked",
    "local_cpu": "locally_runnable",
    "jarvislabs_cloud": "cloud_only",
    "provider_execution": "provider_only",
    "hardware_ticket": "hardware_ticket_only",
    "gtx1060_workstation": "unsupported_local_hardware",
    "ml350_isolated": "isolated_host_only",
}


@dataclass(frozen=True)
class DifferentiableDependencyEnvironmentEvidence:
    """Describe one version-pin or execution-route evidence row.

    Parameters
    ----------
    evidence_id : DifferentiableDependencyEnvironmentEvidenceId or str
        Stable identifier governed by the required evidence inventory.
    title : str
        Reviewer-facing row title.
    category : DifferentiableDependencyEnvironmentEvidenceCategory
        Whether the row describes a toolchain or an execution route.
    classification : str
        Required toolchain or route classification for the row.
    version_pins : tuple[str, ...]
        Exact pins or, for a declared-unlocked hard gap, version constraints.
    evidence_paths : tuple[str, ...]
        Repository-relative source files supporting the row.
    evidence_sha256 : tuple[str, ...]
        SHA-256 digests aligned one-to-one with ``evidence_paths``.
    evidence_status : DifferentiableDependencyEnvironmentEvidenceStatus
        Locked evidence or an explicit hard gap.
    blockers : tuple[str, ...]
        Promotion blockers; empty only for locked evidence.
    claim_boundary : str
        Canonical non-promotional interpretation.

    """

    evidence_id: DifferentiableDependencyEnvironmentEvidenceId | str
    title: str
    category: DifferentiableDependencyEnvironmentEvidenceCategory
    classification: str
    version_pins: tuple[str, ...]
    evidence_paths: tuple[str, ...]
    evidence_sha256: tuple[str, ...]
    evidence_status: DifferentiableDependencyEnvironmentEvidenceStatus
    blockers: tuple[str, ...]
    claim_boundary: str

    def __post_init__(self) -> None:
        """Reject malformed, contradictory, or unbounded evidence rows."""
        for field_name in ("evidence_id", "title", "classification"):
            _require_nonblank_text(getattr(self, field_name), field_name)
        if self.category not in {"toolchain", "execution_route"}:
            raise ValueError("category must be toolchain or execution_route")
        _require_string_tuple(
            self.version_pins,
            "version_pins",
            require_nonempty=self.category == "toolchain",
        )
        _require_string_tuple(self.evidence_paths, "evidence_paths", require_nonempty=True)
        _require_string_tuple(self.evidence_sha256, "evidence_sha256", require_nonempty=True)
        if len(self.evidence_sha256) != len(self.evidence_paths):
            raise ValueError("evidence_sha256 must align with evidence_paths")
        if any(len(digest) != 64 or not _is_lower_hex(digest) for digest in self.evidence_sha256):
            raise ValueError("evidence_sha256 must contain lowercase SHA-256 digests")
        _require_string_tuple(self.blockers, "blockers", require_nonempty=False)
        if self.evidence_status not in {"locked", "hard_gap"}:
            raise ValueError("evidence_status must be locked or hard_gap")
        if self.evidence_status == "locked" and self.blockers:
            raise ValueError("locked evidence must not carry blockers")
        if self.evidence_status == "hard_gap" and not self.blockers:
            raise ValueError("hard-gap evidence must list blockers")
        if self.claim_boundary != DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_EVIDENCE_CLAIM_BOUNDARY:
            raise ValueError("claim_boundary must match the canonical evidence boundary")

    @property
    def environment_ready(self) -> bool:
        """Return whether the evidence row is locked without blockers."""
        return self.evidence_status == "locked" and not self.blockers

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready evidence record."""
        return {
            "evidence_id": self.evidence_id,
            "title": self.title,
            "category": self.category,
            "classification": self.classification,
            "version_pins": list(self.version_pins),
            "evidence_paths": list(self.evidence_paths),
            "evidence_sha256": list(self.evidence_sha256),
            "evidence_status": self.evidence_status,
            "blockers": list(self.blockers),
            "environment_ready": self.environment_ready,
            "claim_boundary": self.claim_boundary,
        }


def build_differentiable_dependency_environment_evidence(
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[DifferentiableDependencyEnvironmentEvidence, ...]:
    """Build the required version-pin and execution-route evidence inventory.

    Parameters
    ----------
    repo_root : pathlib.Path, optional
        Repository root containing every cited evidence source.

    Returns
    -------
    tuple[DifferentiableDependencyEnvironmentEvidence, ...]
        Canonically ordered toolchain and execution-route evidence.

    """
    framework_freeze = (
        "data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/"
        "framework_overlay_freeze.txt"
    )
    enzyme_freeze = (
        "data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/enzyme_py39_freeze.txt"
    )
    catalyst_evidence = (
        "data/differentiable_phase_qnode/ml350_full_framework_catalyst_baseline_20260705/"
        "diff-qnode-external-comparison.json"
    )
    maturity_evidence = "data/differentiable_phase_qnode/enzyme_mlir_maturity_audit_20260616.json"
    records = (
        _record(
            repo_root,
            "python_versions",
            "Python versions",
            "toolchain",
            "locked_versions",
            ("python==3.11", "python==3.12", "python==3.13", "enzyme-runner-python==3.9"),
            (
                "pyproject.toml",
                "requirements-ci-py311-linux.txt",
                "requirements-ci-py312-linux.txt",
                "requirements-ci-py313-linux.txt",
                "data/differentiable_phase_qnode/external_validation_environment_lock_20260616.json",
            ),
            "locked",
        ),
        _record(
            repo_root,
            "rust_crates",
            "Rust crate lock",
            "toolchain",
            "locked_versions",
            (
                "anyhow==1.0.104",
                "chrono==0.4.45",
                "clap==4.6.3",
                "criterion==0.5.1",
                "scpn-quantum-engine==0.2.0",
                "pyo3==0.29.0",
                "ndarray==0.16.1",
                "numpy==0.29.0",
                "nalgebra==0.35.0",
                "num-complex==0.4.6",
                "rand==0.10.2",
                "rayon==1.11.0",
                "reqwest==0.13.4",
                "scpn-quantum-program-ad-replay==0.1.0",
                "serde==1.0.229",
                "serde_json==1.0.151",
            ),
            ("scpn_quantum_engine/Cargo.lock",),
            "locked",
        ),
        _record(
            repo_root,
            "jax_cpu",
            "JAX CPU overlay",
            "toolchain",
            "locked_versions",
            ("jax==0.10.1", "jaxlib==0.10.1"),
            (framework_freeze,),
            "locked",
        ),
        _record(
            repo_root,
            "pytorch_cpu",
            "PyTorch CPU overlay",
            "toolchain",
            "locked_versions",
            ("torch==2.12.0+cpu",),
            (framework_freeze,),
            "locked",
        ),
        _record(
            repo_root,
            "tensorflow_cpu",
            "TensorFlow CPU overlay",
            "toolchain",
            "locked_versions",
            ("tensorflow_cpu==2.21.0",),
            (framework_freeze,),
            "locked",
        ),
        _record(
            repo_root,
            "pennylane_cpu",
            "PennyLane CPU overlay",
            "toolchain",
            "locked_versions",
            ("pennylane==0.45.0", "pennylane_lightning==0.45.0"),
            (framework_freeze,),
            "locked",
        ),
        _record(
            repo_root,
            "qiskit",
            "Qiskit runtime and provider tooling",
            "toolchain",
            "locked_versions",
            (
                "qiskit==2.5.0",
                "qiskit-aer==0.17.2",
                "qiskit-qasm3-import==0.6.0",
                "qiskit-ibm-runtime==0.47.0",
            ),
            ("requirements.txt", "requirements-dev.txt"),
            "locked",
        ),
        _record(
            repo_root,
            "catalyst",
            "Catalyst bounded ML350 tooling",
            "toolchain",
            "locked_versions",
            ("pennylane-catalyst==0.15.0", "catalyst==0.15.0"),
            (catalyst_evidence,),
            "locked",
        ),
        _record(
            repo_root,
            "enzyme_llvm_mlir",
            "Enzyme, LLVM, and MLIR tooling",
            "toolchain",
            "locked_versions",
            ("enzyme-ad==0.0.6", "Enzyme LLVM plugin 0.0.79", "LLVM==18.1.3", "MLIR==18.1.3"),
            (enzyme_freeze, maturity_evidence),
            "locked",
        ),
        _record(
            repo_root,
            "gpu_overlay",
            "Optional GPU overlay",
            "toolchain",
            "declared_unlocked",
            ("cupy-cuda12x>=13.0", "jax[cuda12]>=0.4.30", "torch>=2.2,<3.0"),
            (
                "pyproject.toml",
                "src/scpn_quantum_control/phase/jax_maturity.py",
                "src/scpn_quantum_control/phase/torch_maturity.py",
            ),
            "hard_gap",
            (
                "The optional CUDA requirements are constrained but have no exact GPU lock or compatible modern-GPU execution artefact.",
            ),
        ),
        _record(
            repo_root,
            "local_cpu",
            "Local CPU execution",
            "execution_route",
            "locally_runnable",
            (),
            (
                "data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/diff-qnode-external-comparison.json",
            ),
            "locked",
        ),
        _record(
            repo_root,
            "jarvislabs_cloud",
            "JarvisLabs cloud execution",
            "execution_route",
            "cloud_only",
            (),
            (
                "src/scpn_quantum_control/phase/jax_maturity.py",
                "src/scpn_quantum_control/phase/torch_maturity.py",
                "docs/quantum_gradients.md",
            ),
            "hard_gap",
            (
                "JarvisLabs is a dispatch plan only until returned CUDA, XLA, pmap, device, and isolated-benchmark artefacts validate.",
            ),
        ),
        _record(
            repo_root,
            "provider_execution",
            "Provider execution",
            "execution_route",
            "provider_only",
            (),
            ("src/scpn_quantum_control/phase/qiskit_runtime.py", "docs/differentiable_api.md"),
            "hard_gap",
            (
                "Provider execution requires an approved provider workflow and captured provider artefacts.",
            ),
        ),
        _record(
            repo_root,
            "hardware_ticket",
            "Ticketed hardware execution",
            "execution_route",
            "hardware_ticket_only",
            (),
            ("src/scpn_quantum_control/phase/qiskit_runtime.py", "docs/differentiable_api.md"),
            "hard_gap",
            (
                "Live hardware execution requires an owner-approved ticket, job metadata, raw counts, calibration, and replay evidence.",
            ),
        ),
        _record(
            repo_root,
            "gtx1060_workstation",
            "GTX 1060 workstation",
            "execution_route",
            "unsupported_local_hardware",
            (),
            ("src/scpn_quantum_control/phase/jax_maturity.py", "docs/differentiable_api.md"),
            "hard_gap",
            (
                "The GTX 1060 is explicitly incompatible with the modern CUDA and multi-device promotion route.",
            ),
        ),
        _record(
            repo_root,
            "ml350_isolated",
            "ML350 isolated execution",
            "execution_route",
            "isolated_host_only",
            (),
            (
                "docs/differentiable_reviewer_evidence.md",
                "data/differentiable_phase_qnode/ml350_full_framework_catalyst_baseline_20260705/phase_qnode_affinity_validation_isolated_required.json",
            ),
            "hard_gap",
            (
                "ML350 bounded CPU evidence exists, but promotion-grade isolated reruns remain blocked on the host-readiness and RAM gate.",
            ),
        ),
    )
    return records


def _record(
    repo_root: Path,
    evidence_id: DifferentiableDependencyEnvironmentEvidenceId,
    title: str,
    category: DifferentiableDependencyEnvironmentEvidenceCategory,
    classification: str,
    version_pins: tuple[str, ...],
    evidence_paths: tuple[str, ...],
    evidence_status: DifferentiableDependencyEnvironmentEvidenceStatus,
    blockers: tuple[str, ...] = (),
) -> DifferentiableDependencyEnvironmentEvidence:
    """Build one evidence row and bind each cited source by SHA-256."""
    evidence_sha256 = tuple(
        hashlib.sha256((repo_root / path).read_bytes()).hexdigest() for path in evidence_paths
    )
    return DifferentiableDependencyEnvironmentEvidence(
        evidence_id=evidence_id,
        title=title,
        category=category,
        classification=classification,
        version_pins=version_pins,
        evidence_paths=evidence_paths,
        evidence_sha256=evidence_sha256,
        evidence_status=evidence_status,
        blockers=blockers,
        claim_boundary=DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_EVIDENCE_CLAIM_BOUNDARY,
    )


def _require_nonblank_text(value: object, field_name: str) -> None:
    """Require an exact non-blank string."""
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_string_tuple(value: object, field_name: str, *, require_nonempty: bool) -> None:
    """Require a tuple of unique non-blank strings."""
    if type(value) is not tuple:
        raise ValueError(f"{field_name} must be a tuple")
    if require_nonempty and not value:
        raise ValueError(f"{field_name} must be non-empty")
    if any(type(item) is not str or not item.strip() for item in value):
        raise ValueError(f"{field_name} must contain non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{field_name} must contain unique entries")


def _is_lower_hex(value: str) -> bool:
    """Return whether a value contains lowercase hexadecimal digits only."""
    return all(character in "0123456789abcdef" for character in value)


__all__ = [
    "DIFFERENTIABLE_DEPENDENCY_ENVIRONMENT_EVIDENCE_CLAIM_BOUNDARY",
    "REQUIRED_DEPENDENCY_ENVIRONMENT_EVIDENCE_CLASSIFICATIONS",
    "REQUIRED_DEPENDENCY_ENVIRONMENT_EVIDENCE_IDS",
    "DifferentiableDependencyEnvironmentEvidence",
    "DifferentiableDependencyEnvironmentEvidenceCategory",
    "DifferentiableDependencyEnvironmentEvidenceId",
    "DifferentiableDependencyEnvironmentEvidenceStatus",
    "build_differentiable_dependency_environment_evidence",
]
