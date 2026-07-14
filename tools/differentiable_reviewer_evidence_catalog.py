# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable reviewer-evidence catalogue
"""Typed immutable catalogue for differentiable reviewer evidence.

This module owns the DP-015 criticism rows, DP-030 evidence-package rows,
public open-gap definitions, and capability-manifest requirements. Validation,
rendering, file access, and command-line behavior belong to
``differentiable_reviewer_evidence_page``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

REVIEWER_EVIDENCE_PAGE_SCHEMA: Final[str] = "scpn_qc_differentiable_reviewer_evidence_page_v1"
CAPABILITY_MANIFEST_SCHEMA: Final[str] = "capability-manifest.v1"
GENERATED_BY: Final[str] = "python tools/differentiable_reviewer_evidence_page.py --write"

EvidenceCategory: TypeAlias = Literal["reviewer_criticism", "evidence_package"]
EvidenceStatus: TypeAlias = Literal["implemented", "bounded", "open"]


def _require_non_blank_fields(label: str, fields: Mapping[str, str]) -> None:
    """Reject blank dataclass fields with a precise label."""
    for name, value in fields.items():
        if not value.strip():
            raise ValueError(f"{label} {name} must be non-blank")


def _require_unique_non_blank(values: Sequence[str], label: str) -> None:
    """Reject blank or duplicate tuple values."""
    if any(not value.strip() for value in values):
        raise ValueError(f"{label} must contain only non-blank strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicates")


REQUIRED_PACKAGE_EXPORTS: Final[tuple[str, ...]] = (
    "PhaseVQE",
    "batch_parameter_shift_gradient",
    "program_ad_registry_dispatch_coverage_report",
    "run_differentiable_programming_benchmark_suite",
    "run_quantum_gradient_benchmark_suite",
)
REQUIRED_SOURCE_MODULES: Final[tuple[str, ...]] = (
    "src/scpn_quantum_control/benchmarks/differentiable_programming.py",
    "src/scpn_quantum_control/differentiable_parameter_shift.py",
    "src/scpn_quantum_control/phase/gradient_support_matrix.py",
    "src/scpn_quantum_control/phase/phase_vqe.py",
    "src/scpn_quantum_control/phase/qnn_optimizer_benchmark.py",
    "src/scpn_quantum_control/phase/qnode_framework_parity.py",
)
REQUIRED_TEST_FILES: Final[tuple[str, ...]] = (
    "tests/test_differentiable_reviewer_evidence_page.py",
    "tests/test_differentiable_parameter_shift.py",
    "tests/test_differentiable_programming_benchmarks.py",
    "tests/test_phase_gradient_support_matrix.py",
    "tests/test_phase_qnn_framework_bridge_matrix.py",
    "tests/test_phase_qnn_optimizer_benchmark.py",
    "tests/test_phase_qnode_framework_parity.py",
    "tests/test_phase_vqe.py",
)
REQUIRED_PUBLIC_PAGES: Final[tuple[str, ...]] = (
    "docs/differentiable_programming.md",
    "docs/differentiable_reviewer_evidence.md",
    "docs/differentiable_roadmap.md",
    "docs/differentiable_support_matrix.md",
    "docs/quantum_gradients.md",
)


@dataclass(frozen=True)
class OpenEvidenceGap:
    """Public pointer for evidence that remains unpromoted.

    Parameters
    ----------
    pointer
        Stable public gap identifier.
    title
        Short human-readable gap name.
    roadmap_marker
        Exact marker required in the public differentiable roadmap.
    detail
        Bounded description of the evidence still required.

    Raises
    ------
    ValueError
        If a field is blank or the pointer is not canonical.

    """

    pointer: str
    title: str
    roadmap_marker: str
    detail: str

    def __post_init__(self) -> None:
        """Reject malformed public gap definitions."""
        if not self.pointer.startswith("DIFF-OPEN-"):
            raise ValueError("open-gap pointers must start with DIFF-OPEN-")
        _require_non_blank_fields(
            "open gap",
            {
                "pointer": self.pointer,
                "title": self.title,
                "roadmap_marker": self.roadmap_marker,
                "detail": self.detail,
            },
        )

    def to_dict(self) -> dict[str, str]:
        """Return a deterministic JSON-ready gap record."""
        return {
            "pointer": self.pointer,
            "title": self.title,
            "roadmap_marker": self.roadmap_marker,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ReviewerEvidenceRow:
    """One reviewer criticism or DP-030 evidence-package row.

    Parameters
    ----------
    row_id
        Stable DP-015 or DP-030 row identifier.
    category
        Whether the row answers a criticism or inventories the package.
    criticism
        Reviewer concern or evidence-package requirement.
    status
        Implemented, bounded, or open evidence classification.
    evidence_summary
        Factual summary of the current repository evidence.
    commands
        Scoped commands that exercise the named production surfaces.
    evidence_paths
        Repository-relative paths supporting the row.
    open_gap_refs
        Public roadmap gap identifiers, when promotion remains open.
    changelog_markers
        Exact changelog substrings required for historical traceability.
    claim_boundary
        Explicit statement of what the row does not promote.

    Raises
    ------
    ValueError
        If fields, classifications, or evidence routes are inconsistent.

    """

    row_id: str
    category: EvidenceCategory
    criticism: str
    status: EvidenceStatus
    evidence_summary: str
    commands: tuple[str, ...]
    evidence_paths: tuple[str, ...]
    open_gap_refs: tuple[str, ...]
    changelog_markers: tuple[str, ...]
    claim_boundary: str

    def __post_init__(self) -> None:
        """Reject incomplete or contradictory row definitions."""
        _require_non_blank_fields(
            "reviewer evidence row",
            {
                "row_id": self.row_id,
                "criticism": self.criticism,
                "evidence_summary": self.evidence_summary,
                "claim_boundary": self.claim_boundary,
            },
        )
        allowed_categories = {"reviewer_criticism", "evidence_package"}
        allowed_statuses = {"implemented", "bounded", "open"}
        if self.category not in allowed_categories:
            raise ValueError(f"unsupported reviewer-evidence category: {self.category}")
        if self.status not in allowed_statuses:
            raise ValueError(f"unsupported reviewer-evidence status: {self.status}")
        prefix = "DP-015-" if self.category == "reviewer_criticism" else "DP-030-"
        if not self.row_id.startswith(prefix):
            raise ValueError(f"{self.category} row IDs must start with {prefix}")
        _require_unique_non_blank(self.commands, f"{self.row_id} commands")
        _require_unique_non_blank(self.evidence_paths, f"{self.row_id} evidence paths")
        _require_unique_non_blank(self.open_gap_refs, f"{self.row_id} open-gap references")
        _require_unique_non_blank(self.changelog_markers, f"{self.row_id} changelog markers")
        if not self.commands and not self.open_gap_refs:
            raise ValueError(f"{self.row_id} must name a command or open-gap pointer")
        if self.status == "open" and not self.open_gap_refs:
            raise ValueError(f"open row {self.row_id} must name an open-gap pointer")
        if self.status == "implemented" and self.open_gap_refs:
            raise ValueError(f"implemented row {self.row_id} cannot carry open-gap pointers")
        if not self.evidence_paths:
            raise ValueError(f"{self.row_id} must name at least one evidence path")

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-ready evidence row."""
        return {
            "row_id": self.row_id,
            "category": self.category,
            "criticism": self.criticism,
            "status": self.status,
            "evidence_summary": self.evidence_summary,
            "commands": list(self.commands),
            "evidence_paths": list(self.evidence_paths),
            "open_gap_refs": list(self.open_gap_refs),
            "changelog_markers": list(self.changelog_markers),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class SupportMatrixSummary:
    """Counts parsed from the generated differentiable support matrix.

    Parameters
    ----------
    registry_covered
        Number of complete Program AD registry rows.
    registry_total
        Number of declared Program AD registry rows.
    planner_supported
        Number of supported planner audit cases.
    planner_blocked
        Number of fail-closed planner audit cases.

    Raises
    ------
    ValueError
        If counts are non-positive or the registry is incomplete.

    """

    registry_covered: int
    registry_total: int
    planner_supported: int
    planner_blocked: int

    def __post_init__(self) -> None:
        """Reject stale-looking or internally inconsistent summaries."""
        counts = (
            self.registry_covered,
            self.registry_total,
            self.planner_supported,
            self.planner_blocked,
        )
        if any(isinstance(value, bool) or value <= 0 for value in counts):
            raise ValueError("support-matrix summary counts must be positive integers")
        if self.registry_covered != self.registry_total:
            raise ValueError("reviewer evidence requires complete declared registry coverage")

    def to_dict(self) -> dict[str, int]:
        """Return a deterministic JSON-ready support summary."""
        return {
            "registry_covered": self.registry_covered,
            "registry_total": self.registry_total,
            "planner_supported": self.planner_supported,
            "planner_blocked": self.planner_blocked,
        }


@dataclass(frozen=True)
class ReviewerEvidenceValidation:
    """Verdict for reviewer-evidence page source alignment.

    Parameters
    ----------
    passed
        Whether every row and referenced repository surface is valid.
    errors
        Human-readable mismatch descriptions, empty on success.

    Raises
    ------
    ValueError
        If the verdict and error list contradict one another.

    """

    passed: bool
    errors: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject contradictory validation verdicts."""
        if self.passed and self.errors:
            raise ValueError("a passed validation must not carry errors")
        if not self.passed and not self.errors:
            raise ValueError("a failed validation must explain its errors")


OPEN_GAPS: Final[tuple[OpenEvidenceGap, ...]] = (
    OpenEvidenceGap(
        "DIFF-OPEN-01",
        "Framework-native and accelerator promotion",
        "DIFF-OPEN-01 — Framework-native and accelerator promotion",
        "Broader native framework transforms, incompatible accelerator routes, and "
        "cross-runtime persistence require dedicated artefacts.",
    ),
    OpenEvidenceGap(
        "DIFF-OPEN-02",
        "Provider and hardware gradient evidence",
        "DIFF-OPEN-02 — Provider and hardware gradient evidence",
        "Live-ticket provider jobs, raw counts, calibration, and simulator comparison "
        "remain approval-gated.",
    ),
    OpenEvidenceGap(
        "DIFF-OPEN-03",
        "Isolated benchmark promotion",
        "DIFF-OPEN-03 — Isolated benchmark promotion",
        "Performance claims require reserved-host isolated-affinity artefacts and "
        "validated comparison metadata.",
    ),
    OpenEvidenceGap(
        "DIFF-OPEN-04",
        "Compiler AD promotion",
        "DIFF-OPEN-04 — Compiler AD promotion",
        "Broader Enzyme, MLIR, LLVM/JIT, and executable registry promotion requires "
        "native evidence plus isolated benchmark identifiers.",
    ),
    OpenEvidenceGap(
        "DIFF-OPEN-05",
        "Tutorial and notebook breadth",
        "DIFF-OPEN-05 — Tutorial and notebook breadth",
        "Public notebooks beyond the executable examples remain open until clean-"
        "environment replay and expected outputs are recorded.",
    ),
)

REVIEWER_EVIDENCE_ROWS: Final[tuple[ReviewerEvidenceRow, ...]] = (
    ReviewerEvidenceRow(
        "DP-015-01",
        "reviewer_criticism",
        "No parameter-shift implementation",
        "implemented",
        "Scalar, batched, generalised finite-spectrum, registered Phase-QNode, and "
        "finite-shot parameter-shift contracts have executable tests.",
        (
            "python -m pytest -q tests/test_differentiable_parameter_shift.py "
            "tests/test_phase_generalised_parameter_shift.py",
        ),
        (
            "src/scpn_quantum_control/differentiable_parameter_shift.py",
            "src/scpn_quantum_control/phase/generalised_parameter_shift.py",
            "tests/test_differentiable_parameter_shift.py",
            "tests/test_phase_generalised_parameter_shift.py",
            "docs/quantum_gradients.md",
        ),
        (),
        (
            "Added a registered local Phase-QNode circuit family",
            "analytic parameter-shift gradients",
        ),
        "Implemented for declared local and bounded callback routes; no universal "
        "gate, provider, or hardware-gradient claim is implied.",
    ),
    ReviewerEvidenceRow(
        "DP-015-02",
        "reviewer_criticism",
        "No quantum-gradient surface",
        "bounded",
        "The executable planner and Phase-QNode gradient backend cover supported "
        "statevector, finite-shot callback, and host-bridge cases while unsafe cells "
        "fail closed.",
        (
            "python -m pytest -q tests/test_phase_gradient_backend.py "
            "tests/test_phase_gradient_support_matrix.py",
        ),
        (
            "src/scpn_quantum_control/phase/gradient_backend.py",
            "src/scpn_quantum_control/phase/gradient_support_matrix.py",
            "tests/test_phase_gradient_backend.py",
            "tests/test_phase_gradient_support_matrix.py",
            "docs/differentiable_support_matrix.md",
        ),
        ("DIFF-OPEN-02",),
        ("finite-shot uncertainty records", "provider callback"),
        "Local and caller-supplied callback evidence does not promote live provider "
        "execution or unrestricted hardware gradients.",
    ),
    ReviewerEvidenceRow(
        "DP-015-03",
        "reviewer_criticism",
        "No autodiff evidence in the changelog",
        "implemented",
        "The changelog records parameter-shift, finite-shot, JAX, PyTorch, "
        "TensorFlow, Program AD, and compiler-AD additions after implementation.",
        ("python tools/differentiable_reviewer_evidence_page.py --check",),
        (
            "CHANGELOG.md",
            "tools/differentiable_reviewer_evidence_page.py",
            "tests/test_differentiable_reviewer_evidence_page.py",
        ),
        (),
        (
            "Added native bounded phase-QNN framework-gradient evidence",
            "Added Rust/PyO3 parity for materialised finite-shot",
            "Added a six-parameter sparse Ising-chain Hamiltonian",
        ),
        "Historical entries document landed bounded surfaces; they do not upgrade "
        "their recorded provider, hardware, or performance boundaries.",
    ),
    ReviewerEvidenceRow(
        "DP-015-04",
        "reviewer_criticism",
        "VQE uses classical or gradient-free optimisation only",
        "implemented",
        "PhaseVQE exposes parameter-shift gradients, gradient descent, natural-"
        "gradient paths, and convergence evidence alongside declared classical baselines.",
        (
            "python -m pytest -q tests/test_phase_vqe.py "
            "tests/test_phase_gradient_descent.py tests/test_phase_natural_gradient.py",
        ),
        (
            "src/scpn_quantum_control/phase/phase_vqe.py",
            "src/scpn_quantum_control/phase/gradient_descent.py",
            "src/scpn_quantum_control/phase/natural_gradient.py",
            "tests/test_phase_vqe.py",
            "tests/test_phase_gradient_descent.py",
            "tests/test_phase_natural_gradient.py",
            "docs/quantum_gradients.md",
        ),
        (),
        (
            "known-ground-state optimizer convergence certificates",
            "natural-gradient PhaseVQE",
        ),
        "Convergence evidence is bounded to declared local objectives and does not "
        "establish global optimiser superiority.",
    ),
    ReviewerEvidenceRow(
        "DP-015-05",
        "reviewer_criticism",
        "No ML-framework integration",
        "bounded",
        "JAX, PyTorch, TensorFlow, PennyLane, and Qiskit rows are exercised through "
        "the public framework bridge/parity contracts with explicit optional-dependency "
        "and unsupported-route classifications.",
        (
            "python -m pytest -q tests/test_phase_qnn_framework_bridge_matrix.py "
            "tests/test_phase_qnode_framework_parity.py",
        ),
        (
            "src/scpn_quantum_control/phase/qnn_framework_bridge_matrix.py",
            "src/scpn_quantum_control/phase/qnode_framework_parity.py",
            "tests/test_phase_qnn_framework_bridge_matrix.py",
            "tests/test_phase_qnode_framework_parity.py",
            "docs/differentiable_programming.md",
        ),
        ("DIFF-OPEN-01", "DIFF-OPEN-02"),
        (
            "Added native bounded phase-QNN framework-gradient evidence",
            "Corrected Phase-QNode framework-parity verification",
        ),
        "Installed local and compatibility routes do not imply arbitrary-simulator, "
        "accelerator, provider, or hardware framework autodiff.",
    ),
    ReviewerEvidenceRow(
        "DP-030-01",
        "evidence_package",
        "Finite-difference versus parameter-shift correctness and runtime",
        "bounded",
        "The differentiable benchmark suite records analytic agreement and local "
        "functional timing with explicit non-isolated classification.",
        ("python -m pytest -q tests/test_differentiable_programming_benchmarks.py",),
        (
            "src/scpn_quantum_control/benchmarks/differentiable_programming.py",
            "tests/test_differentiable_programming_benchmarks.py",
            "data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/"
            "diff-qnode-ci-evidence-schema-v1.json",
        ),
        ("DIFF-OPEN-03",),
        ("quantum-gradient benchmark suite",),
        "Local functional timing is not isolated performance evidence and supports no "
        "speedup claim.",
    ),
    ReviewerEvidenceRow(
        "DP-030-02",
        "evidence_package",
        "Batched shifted-circuit execution",
        "implemented",
        "Batched parameter-shift helpers and benchmark rows validate values, gradients, "
        "evaluation counts, masks, and malformed batch contracts.",
        (
            "python -m pytest -q tests/test_differentiable_batch_helpers.py "
            "tests/test_differentiable_parameter_shift.py",
        ),
        (
            "src/scpn_quantum_control/differentiable_batch_helpers.py",
            "src/scpn_quantum_control/differentiable_parameter_shift.py",
            "tests/test_differentiable_batch_helpers.py",
            "tests/test_differentiable_parameter_shift.py",
        ),
        (),
        ("batch parameter-shift",),
        "Batched local evaluation does not imply provider-side batching, QPU execution, "
        "or performance promotion.",
    ),
    ReviewerEvidenceRow(
        "DP-030-03",
        "evidence_package",
        "VQE and QNN optimiser convergence comparison",
        "bounded",
        "Deterministic multi-start and QNN optimiser suites compare parameter-shift, "
        "finite-difference, natural-gradient, Adam, L-BFGS-B, SPSA, and declared "
        "derivative-free routes.",
        (
            "python -m pytest -q tests/test_phase_qnn_optimizer_benchmark.py "
            "tests/test_phase_vqe.py",
        ),
        (
            "src/scpn_quantum_control/phase/qnn_optimizer_benchmark.py",
            "src/scpn_quantum_control/phase/phase_vqe.py",
            "tests/test_phase_qnn_optimizer_benchmark.py",
            "tests/test_phase_vqe.py",
            "data/differentiable_phase_qnode/ground_state_optimizer_convergence_20260709.json",
        ),
        ("DIFF-OPEN-03",),
        ("known-ground-state optimizer convergence certificates",),
        "Bounded local convergence comparisons do not prove general optimiser or "
        "application superiority.",
    ),
    ReviewerEvidenceRow(
        "DP-030-04",
        "evidence_package",
        "Framework adapter overhead benchmarks",
        "open",
        "Functional framework-overlay evidence exists, but promotable overhead ratios "
        "still require compatible accelerator and isolated-affinity artefacts.",
        (
            "python -m pytest -q tests/test_differentiable_framework_overlay.py "
            "tests/test_differentiable_benchmark_workflow.py",
        ),
        (
            "src/scpn_quantum_control/differentiable_framework_overlay.py",
            "tests/test_differentiable_framework_overlay.py",
            "tests/test_differentiable_benchmark_workflow.py",
            "data/differentiable_phase_qnode/ml350_framework_overlay_baseline_20260705/"
            "diff-qnode-external-comparison.json",
        ),
        ("DIFF-OPEN-01", "DIFF-OPEN-03"),
        ("framework overlay",),
        "Functional overlay rows are not isolated overhead or accelerator performance claims.",
    ),
    ReviewerEvidenceRow(
        "DP-030-05",
        "evidence_package",
        "Compiler-backed AD benchmark evidence",
        "bounded",
        "Native whole-program and Enzyme/MLIR evidence rows are executable where the "
        "toolchain is present and remain non-promotional without isolated artefact IDs.",
        (
            "python -m pytest -q tests/test_native_whole_program_ad_execution_evidence.py "
            "tests/test_enzyme_toolchain_execution_evidence.py "
            "tests/test_phase_qnode_compiler_lowering.py",
        ),
        (
            "src/scpn_quantum_control/compiler/mlir_enzyme_evidence.py",
            "tests/test_native_whole_program_ad_execution_evidence.py",
            "tests/test_enzyme_toolchain_execution_evidence.py",
            "tests/test_phase_qnode_compiler_lowering.py",
            "data/differentiable_phase_qnode/compiler_evidence_boundary_20260705.json",
        ),
        ("DIFF-OPEN-03", "DIFF-OPEN-04"),
        ("compiler promotion-batch",),
        "Toolchain execution rows do not promote broad compiler AD, Rust registry "
        "execution, LLVM/JIT, or performance claims.",
    ),
    ReviewerEvidenceRow(
        "DP-030-06",
        "evidence_package",
        "Noisy-gradient and shot-allocation evidence",
        "bounded",
        "Finite-shot parameter-shift, seeded SPSA, score-function, confidence-interval, "
        "shot-allocation, and failure-policy contracts have executable tests.",
        (
            "python -m pytest -q tests/test_stochastic_gradient_failure_policy.py "
            "tests/test_stochastic_gradient_rust_parity.py "
            "tests/test_phase_qnn_optimizer_benchmark.py",
        ),
        (
            "src/scpn_quantum_control/differentiable_stochastic_estimators.py",
            "src/scpn_quantum_control/differentiable_stochastic_policy.py",
            "tests/test_stochastic_gradient_failure_policy.py",
            "tests/test_stochastic_gradient_rust_parity.py",
            "tests/test_phase_qnn_optimizer_benchmark.py",
            "docs/quantum_gradients.md",
        ),
        ("DIFF-OPEN-02", "DIFF-OPEN-03"),
        ("stochastic-gradient confidence intervals",),
        "Seeded simulator and callback uncertainty evidence does not imply calibrated "
        "hardware robustness or performance promotion.",
    ),
    ReviewerEvidenceRow(
        "DP-030-07",
        "evidence_package",
        "Reviewer evidence page and synchronization gate",
        "implemented",
        "This generated page validates all row commands, paths, changelog markers, "
        "public open gaps, support-matrix counts, and capability-manifest surfaces.",
        (
            "python tools/differentiable_support_matrix_page.py --check",
            "python tools/differentiable_reviewer_evidence_page.py --check",
        ),
        (
            "tools/differentiable_support_matrix_page.py",
            "tools/differentiable_reviewer_evidence_page.py",
            "tests/test_differentiable_support_matrix_page.py",
            "tests/test_differentiable_reviewer_evidence_page.py",
            "docs/differentiable_support_matrix.md",
            "docs/differentiable_reviewer_evidence.md",
        ),
        (),
        ("generated differentiable reviewer-evidence page",),
        "The page inventories evidence and gaps; it does not itself prove numerical "
        "correctness, provider execution, or benchmark promotion.",
    ),
)

REQUIRED_ROW_IDS: Final[frozenset[str]] = frozenset(row.row_id for row in REVIEWER_EVIDENCE_ROWS)


__all__ = [
    "CAPABILITY_MANIFEST_SCHEMA",
    "GENERATED_BY",
    "OPEN_GAPS",
    "REQUIRED_PACKAGE_EXPORTS",
    "REQUIRED_PUBLIC_PAGES",
    "REQUIRED_ROW_IDS",
    "REQUIRED_SOURCE_MODULES",
    "REQUIRED_TEST_FILES",
    "REVIEWER_EVIDENCE_PAGE_SCHEMA",
    "REVIEWER_EVIDENCE_ROWS",
    "OpenEvidenceGap",
    "ReviewerEvidenceRow",
    "ReviewerEvidenceValidation",
    "SupportMatrixSummary",
]
