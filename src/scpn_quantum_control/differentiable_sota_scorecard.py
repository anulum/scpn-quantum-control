# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable SOTA scorecard.
"""State-of-art scorecard governance for differentiable computing claims."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .differentiable_claim_ledger import (
    DEFAULT_LEDGER_PATH,
    REPO_ROOT,
    ClaimLedger,
    ClaimLedgerRow,
    load_differentiable_claim_ledger,
)

DifferentiableSOTACategory = Literal[
    "jax_native_transforms",
    "pytorch_autograd_compile",
    "pennylane_qnode_device_plugin",
    "qiskit_runtime_provider_gradients",
    "catalyst_compiler_workflows",
    "enzyme_compiler_ad",
    "rust_native_program_ad",
    "provider_hardware_gradients",
    "benchmark_promotion",
    "docs_api_maintainability",
    "adoption_licensing",
]
DifferentiableSOTAStatus = Literal[
    "behind_baseline",
    "at_baseline",
    "exceeds_baseline",
    "not_comparable",
]

DIFFERENTIABLE_SOTA_SCORECARD_SCHEMA = "scpn_qc_differentiable_sota_scorecard_v1"
DIFFERENTIABLE_SOTA_SCORECARD_ARTIFACT_ID = "diff-sota-scorecard-20260620"
DIFFERENTIABLE_SOTA_SCORECARD_CLAIM_BOUNDARY = (
    "Differentiable state-of-art scorecard governance only; the lane remains "
    "SOTA-candidate until category rows have promoted claim-ledger evidence, "
    "isolated benchmark artefacts, and external baseline comparisons."
)
REQUIRED_SOTA_CATEGORIES: tuple[DifferentiableSOTACategory, ...] = (
    "jax_native_transforms",
    "pytorch_autograd_compile",
    "pennylane_qnode_device_plugin",
    "qiskit_runtime_provider_gradients",
    "catalyst_compiler_workflows",
    "enzyme_compiler_ad",
    "rust_native_program_ad",
    "provider_hardware_gradients",
    "benchmark_promotion",
    "docs_api_maintainability",
    "adoption_licensing",
)
READY_STATUSES: frozenset[DifferentiableSOTAStatus] = frozenset(
    {"at_baseline", "exceeds_baseline"}
)
DEFAULT_PUBLIC_SOTA_LANGUAGE_PATHS: tuple[str, ...] = (
    "README.md",
    "docs/differentiable_api.md",
    "docs/differentiable_programming.md",
    "docs/differentiable_external_validation_report.md",
    "docs/onboarding.md",
)
PROMOTIONAL_LANGUAGE_PHRASES: tuple[str, ...] = (
    "state-of-the-art",
    "state of the art",
    "world-class",
    "world-leading",
    "production performance",
    "promotion-ready",
    "promotion ready",
    "promotion_ready=true",
    "at_baseline",
    "at baseline",
    "exceeds_baseline",
    "exceeds baseline",
)
BOUNDED_PROMOTION_LANGUAGE_MARKERS: tuple[str, ...] = (
    "sota-candidate",
    "behind_baseline",
    "non-promotional",
    "not a promotion",
    "not yet suitable",
    "not suitable",
    "not claim",
    "no claim",
    "does not promote",
    "fails when",
    "fails unless",
    "without promoting",
    "without a matching",
    "keeps promotion blocked",
    "remain blocked",
    "remains blocked",
    "blocked until",
    "fail-closed",
)
_CATEGORY_LANGUAGE_MARKERS: Mapping[DifferentiableSOTACategory, tuple[str, ...]] = {
    "jax_native_transforms": ("jax", "native transforms", "pytree", "openxla"),
    "pytorch_autograd_compile": ("pytorch", "torch", "autograd", "torch.compile"),
    "pennylane_qnode_device_plugin": ("pennylane", "qnode", "device plugin"),
    "qiskit_runtime_provider_gradients": ("qiskit", "runtime", "estimator", "sampler"),
    "catalyst_compiler_workflows": ("catalyst", "qjit"),
    "enzyme_compiler_ad": ("enzyme", "compiler ad", "llvm", "mlir"),
    "rust_native_program_ad": ("rust", "program ad"),
    "provider_hardware_gradients": ("provider", "hardware", "qpu"),
    "benchmark_promotion": ("benchmark", "isolated affinity", "performance"),
    "docs_api_maintainability": ("docs", "api", "maintainability"),
    "adoption_licensing": ("adoption", "licensing", "license"),
}


@dataclass(frozen=True)
class DifferentiableSOTAScorecardRow:
    """One external-baseline category in the differentiable SOTA scorecard."""

    category: DifferentiableSOTACategory
    baseline: str
    current_evidence: str
    status: DifferentiableSOTAStatus
    claim_ids: tuple[str, ...]
    implementation_surface: tuple[str, ...]
    test_surface: tuple[str, ...]
    docs_surface: tuple[str, ...]
    benchmark_artifact_ids: tuple[str, ...]
    blockers: tuple[str, ...]
    next_hardening_rounds: tuple[str, ...]
    claim_boundary: str

    def __post_init__(self) -> None:
        """Validate category, status, and evidence invariants."""
        if self.category not in REQUIRED_SOTA_CATEGORIES:
            raise ValueError(f"unknown SOTA category: {self.category}")
        if self.status not in {
            "behind_baseline",
            "at_baseline",
            "exceeds_baseline",
            "not_comparable",
        }:
            raise ValueError(f"unknown SOTA status: {self.status}")
        for field_name in (
            "baseline",
            "current_evidence",
            "claim_boundary",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")
        for field_name in (
            "claim_ids",
            "implementation_surface",
            "test_surface",
            "docs_surface",
            "benchmark_artifact_ids",
            "next_hardening_rounds",
        ):
            value = getattr(self, field_name)
            if not value or any(not str(item).strip() for item in value):
                raise ValueError(f"{field_name} must contain non-empty entries")
        if self.status in READY_STATUSES and self.blockers:
            raise ValueError("ready SOTA rows must not carry blockers")
        if self.status == "behind_baseline" and not self.blockers:
            raise ValueError("behind-baseline SOTA rows must list blockers")

    @property
    def ready_for_promotion(self) -> bool:
        """Return whether this category is at or beyond the external baseline."""
        return self.status in READY_STATUSES

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready scorecard row."""
        return {
            "category": self.category,
            "baseline": self.baseline,
            "current_evidence": self.current_evidence,
            "status": self.status,
            "ready_for_promotion": self.ready_for_promotion,
            "claim_ids": list(self.claim_ids),
            "implementation_surface": list(self.implementation_surface),
            "test_surface": list(self.test_surface),
            "docs_surface": list(self.docs_surface),
            "benchmark_artifact_ids": list(self.benchmark_artifact_ids),
            "blockers": list(self.blockers),
            "next_hardening_rounds": list(self.next_hardening_rounds),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class DifferentiableSOTAScorecard:
    """Deterministic scorecard over differentiable external-baseline categories."""

    schema: str
    artifact_id: str
    rows: tuple[DifferentiableSOTAScorecardRow, ...]
    promotion_ready: bool
    ready_category_count: int
    total_category_count: int
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready scorecard payload."""
        return {
            "schema": self.schema,
            "artifact_id": self.artifact_id,
            "promotion_ready": self.promotion_ready,
            "ready_category_count": self.ready_category_count,
            "total_category_count": self.total_category_count,
            "claim_boundary": self.claim_boundary,
            "rows": [row.to_dict() for row in self.rows],
        }


@dataclass(frozen=True)
class DifferentiableSOTAScorecardValidation:
    """Validation result for a differentiable SOTA scorecard."""

    passed: bool
    errors: tuple[str, ...]
    checked_categories: tuple[DifferentiableSOTACategory, ...]
    checked_claim_ids: tuple[str, ...]
    checked_paths: tuple[str, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready validation evidence."""
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "checked_categories": list(self.checked_categories),
            "checked_claim_ids": list(self.checked_claim_ids),
            "checked_paths": list(self.checked_paths),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class DifferentiableSOTAPromotionLanguageAudit:
    """Audit result for public differentiable SOTA promotion wording."""

    passed: bool
    errors: tuple[str, ...]
    checked_paths: tuple[str, ...]
    checked_promotional_categories: tuple[DifferentiableSOTACategory, ...]
    checked_claim_ids: tuple[str, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready public-language audit evidence."""
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "checked_paths": list(self.checked_paths),
            "checked_promotional_categories": list(self.checked_promotional_categories),
            "checked_claim_ids": list(self.checked_claim_ids),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class _ScorecardRowSpec:
    category: DifferentiableSOTACategory
    baseline: str
    current_evidence: str
    claim_ids: tuple[str, ...]
    blockers: tuple[str, ...]
    next_hardening_rounds: tuple[str, ...]


def run_differentiable_sota_scorecard(
    *,
    ledger: ClaimLedger | None = None,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> DifferentiableSOTAScorecard:
    """Build the deterministic SOTA-category scorecard from committed evidence."""
    loaded_ledger = load_differentiable_claim_ledger(ledger_path) if ledger is None else ledger
    claim_rows = {row.claim_id: row for row in loaded_ledger.rows}
    rows = _default_scorecard_rows(claim_rows)
    ready_count = sum(1 for row in rows if row.ready_for_promotion)
    return DifferentiableSOTAScorecard(
        schema=DIFFERENTIABLE_SOTA_SCORECARD_SCHEMA,
        artifact_id=DIFFERENTIABLE_SOTA_SCORECARD_ARTIFACT_ID,
        rows=rows,
        promotion_ready=ready_count == len(rows),
        ready_category_count=ready_count,
        total_category_count=len(rows),
        claim_boundary=DIFFERENTIABLE_SOTA_SCORECARD_CLAIM_BOUNDARY,
    )


def validate_differentiable_sota_scorecard(
    scorecard: DifferentiableSOTAScorecard,
    *,
    ledger: ClaimLedger | None = None,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    repo_root: Path = REPO_ROOT,
) -> DifferentiableSOTAScorecardValidation:
    """Validate category coverage, path evidence, and promotion invariants."""
    loaded_ledger = load_differentiable_claim_ledger(ledger_path) if ledger is None else ledger
    claim_rows = {row.claim_id: row for row in loaded_ledger.rows}
    errors: list[str] = []
    checked_paths: set[str] = set()
    checked_claim_ids: set[str] = set()

    if scorecard.schema != DIFFERENTIABLE_SOTA_SCORECARD_SCHEMA:
        errors.append(f"unexpected scorecard schema: {scorecard.schema}")
    if scorecard.artifact_id != DIFFERENTIABLE_SOTA_SCORECARD_ARTIFACT_ID:
        errors.append(f"unexpected scorecard artifact_id: {scorecard.artifact_id}")
    categories = tuple(row.category for row in scorecard.rows)
    if categories != REQUIRED_SOTA_CATEGORIES:
        errors.append("scorecard categories must match REQUIRED_SOTA_CATEGORIES exactly")
    if scorecard.total_category_count != len(scorecard.rows):
        errors.append("total_category_count does not match row count")
    ready_count = sum(1 for row in scorecard.rows if row.ready_for_promotion)
    if scorecard.ready_category_count != ready_count:
        errors.append("ready_category_count does not match ready rows")
    if scorecard.promotion_ready != (ready_count == len(scorecard.rows)):
        errors.append("promotion_ready does not match ready row count")

    for row in scorecard.rows:
        checked_claim_ids.update(row.claim_ids)
        _validate_scorecard_row(row, claim_rows=claim_rows, errors=errors)
        for path in _row_paths(row):
            checked_paths.add(path)
            if not (repo_root / path).exists():
                errors.append(f"{row.category}: evidence path does not exist: {path}")

    return DifferentiableSOTAScorecardValidation(
        passed=not errors,
        errors=tuple(errors),
        checked_categories=categories,
        checked_claim_ids=tuple(sorted(checked_claim_ids)),
        checked_paths=tuple(sorted(checked_paths)),
        claim_boundary=(
            "SOTA scorecard validation only; validates category coverage and "
            "claim-ledger consistency without promoting performance, provider, "
            "hardware, QPU, GPU, or isolated_affinity claims"
        ),
    )


def audit_differentiable_sota_promotion_language(
    *,
    public_texts: Mapping[str, str] | None = None,
    public_paths: Iterable[str] = DEFAULT_PUBLIC_SOTA_LANGUAGE_PATHS,
    scorecard: DifferentiableSOTAScorecard | None = None,
    ledger: ClaimLedger | None = None,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    repo_root: Path = REPO_ROOT,
) -> DifferentiableSOTAPromotionLanguageAudit:
    """Reject public SOTA wording that lacks promoted scorecard evidence.

    Bounded governance language such as ``SOTA-candidate`` remains allowed. Any
    unbounded public wording that claims state-of-art, exceedance, or promotion
    readiness must reference categories whose scorecard rows are ready and whose
    claim-ledger rows are all promoted.
    """
    loaded_ledger = load_differentiable_claim_ledger(ledger_path) if ledger is None else ledger
    loaded_scorecard = (
        run_differentiable_sota_scorecard(ledger=loaded_ledger) if scorecard is None else scorecard
    )
    texts = (
        _load_public_sota_texts(public_paths=public_paths, repo_root=repo_root)
        if public_texts is None
        else dict(public_texts)
    )
    scorecard_validation = validate_differentiable_sota_scorecard(
        loaded_scorecard,
        ledger=loaded_ledger,
        repo_root=repo_root,
    )
    errors: list[str] = [
        f"scorecard validation failed: {error}" for error in scorecard_validation.errors
    ]
    promoted_claim_ids = {
        row.claim_id for row in loaded_ledger.rows if row.promotion_status == "promoted"
    }
    rows_by_category = {row.category: row for row in loaded_scorecard.rows}
    checked_categories: set[DifferentiableSOTACategory] = set()
    checked_claim_ids: set[str] = set()

    for path, text in texts.items():
        for line_number, line in enumerate(text.splitlines(), start=1):
            phrases = _promotional_phrases(line)
            if not phrases or _is_bounded_promotion_language(line):
                continue
            categories = _referenced_sota_categories(line)
            if not categories:
                categories = tuple(row.category for row in loaded_scorecard.rows)
            for category in categories:
                row = rows_by_category[category]
                checked_categories.add(category)
                checked_claim_ids.update(row.claim_ids)
                missing_promoted_claims = tuple(
                    claim_id for claim_id in row.claim_ids if claim_id not in promoted_claim_ids
                )
                if row.ready_for_promotion and not missing_promoted_claims:
                    continue
                errors.append(
                    f"{path}:{line_number}: public SOTA wording for {category} "
                    f"({', '.join(phrases)}) requires a ready scorecard row and "
                    f"promoted claim-ledger rows; status={row.status}, "
                    f"unpromoted_claim_ids={', '.join(missing_promoted_claims) or 'none'}"
                )

    return DifferentiableSOTAPromotionLanguageAudit(
        passed=not errors,
        errors=tuple(errors),
        checked_paths=tuple(sorted(texts)),
        checked_promotional_categories=tuple(sorted(checked_categories)),
        checked_claim_ids=tuple(sorted(checked_claim_ids)),
        claim_boundary=(
            "public SOTA promotion-language audit only; rejects unbounded "
            "state-of-art, exceedance, production-performance, or promotion-ready "
            "wording unless the referenced scorecard rows and claim-ledger rows "
            "are promoted"
        ),
    )


def render_differentiable_sota_scorecard_markdown(
    scorecard: DifferentiableSOTAScorecard,
) -> str:
    """Render a reviewer-facing Markdown summary of the scorecard."""
    lines = [
        "<!--",
        "SPDX-License-Identifier: AGPL-3.0-or-later",
        "Commercial license available",
        "© Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "© Code 2020–2026 Miroslav Šotek. All rights reserved.",
        "ORCID: 0009-0009-3560-0851",
        "Contact: www.anulum.li | protoscience@anulum.li",
        "SCPN Quantum Control — Differentiable SOTA Scorecard",
        "-->",
        "",
        "# Differentiable SOTA Scorecard",
        "",
        f"- Schema: `{scorecard.schema}`",
        f"- Artifact ID: `{scorecard.artifact_id}`",
        f"- Promotion ready: `{scorecard.promotion_ready}`",
        f"- Ready categories: `{scorecard.ready_category_count}/{scorecard.total_category_count}`",
        f"- Claim boundary: {scorecard.claim_boundary}",
        "",
        "| Category | Status | Baseline | Current evidence | Blockers | Next rounds |",
        "|---|---|---|---|---|---|",
    ]
    for row in scorecard.rows:
        lines.append(
            "| `{category}` | `{status}` | {baseline} | {evidence} | {blockers} | {rounds} |".format(
                category=row.category,
                status=row.status,
                baseline=_markdown_cell(row.baseline),
                evidence=_markdown_cell(row.current_evidence),
                blockers=_markdown_cell("<br>".join(row.blockers) or "none"),
                rounds=_markdown_cell("<br>".join(row.next_hardening_rounds)),
            )
        )
    lines.append("")
    lines.append(
        "Rows marked `behind_baseline` are explicit hardening work, not failure "
        "noise. No category may become `at_baseline` or `exceeds_baseline` "
        "unless the matching claim-ledger rows are promoted with artefact and "
        "benchmark evidence."
    )
    return "\n".join(lines)


def _default_scorecard_rows(
    claim_rows: Mapping[str, ClaimLedgerRow],
) -> tuple[DifferentiableSOTAScorecardRow, ...]:
    row_specs = (
        _row(
            "jax_native_transforms",
            "JAX composable transforms: grad/value_and_grad, jacfwd/jacrev, Hessian, "
            "JVP/VJP, jit, vmap, pmap/sharding, PyTrees, AOT/export, and OpenXLA devices.",
            "Registered deterministic local Phase-QNode JAX flat, PyTree, jit, vmap, "
            "pmap/sharding, and Hessian routes exist without host callbacks for bounded "
            "statevector circuits.",
            ("external_framework_comparison", "phase_qnode_claim_boundary"),
            (
                "provider/native arbitrary simulator lowering remains open",
                "finite-shot, dynamic-circuit, provider, hardware, and isolated benchmark rows remain blocked",
            ),
            ("Round 2 framework-native transform parity", "Round 6 benchmark promotion"),
        ),
        _row(
            "pytorch_autograd_compile",
            "PyTorch eager autograd, nn.Module, torch.func, torch.compile, AOTAutograd, "
            "dynamic shapes, CPU/GPU device maturity, and production training ergonomics.",
            "Bounded PyTorch nn.Module, torch.func, non-fullgraph torch.compile, local "
            "Phase-QNode statevector routes, and compile-boundary diagnostics exist with "
            "fail-closed fullgraph/dynamic-shape/AOTAutograd/CUDA/provider gaps.",
            ("external_framework_comparison", "phase_qnode_claim_boundary"),
            (
                "compatible CUDA/device evidence is scheduled for cloud hardware",
                "fullgraph compile, dynamic-shape promotion, AOTAutograd/export persistence, "
                "finite-shot, provider, hardware, and isolated benchmark rows remain blocked",
            ),
            ("Round 2 framework-native transform parity", "Round 6 benchmark promotion"),
        ),
        _row(
            "pennylane_qnode_device_plugin",
            "PennyLane QNode interfaces, diff-method routing, simulator adjoint/backprop, "
            "finite-shot methods, device gradients, provider plugins, and hardware-compatible routes.",
            "Bounded identical-circuit export/import and local default-qubit parity exist; provider-plugin "
            "execution, provider gradients, hardware plugins, and isolated benchmark evidence remain open.",
            ("external_framework_comparison", "phase_qnode_claim_boundary"),
            (
                "real provider-plugin execution artefacts are missing",
                "hardware-plugin and provider-gradient parity artefacts are missing",
            ),
            ("Round 3 quantum provider/plugin breadth", "Round 6 benchmark promotion"),
        ),
        _row(
            "qiskit_runtime_provider_gradients",
            "Qiskit Estimator/Sampler primitives, IBM Runtime sessions, backend/transpilation "
            "metadata, raw counts, calibration snapshots, and gradient workflows.",
            "No-submit Runtime/QPU evidence schemas and local Statevector parity exist; live Runtime/QPU "
            "execution, raw-count replay attachments, calibration comparison, and isolated evidence remain open.",
            ("phase_qnode_claim_boundary",),
            (
                "live-ticket-approved Runtime Estimator/Sampler execution is missing",
                "provider-gradient workflow evidence is not attached to live QPU runs",
            ),
            ("Round 3 quantum provider/plugin breadth", "Round 6 benchmark promotion"),
        ),
        _row(
            "catalyst_compiler_workflows",
            "Catalyst-style MLIR/LLVM/QIR compiled quantum-classical workflows, qjit, "
            "compiled control flow, compiled differentiation, and device support.",
            "SCPN has MLIR interchange, bounded compiler-AD metadata, and a dedicated "
            "Catalyst external-comparison row that remains hard-gap evidence until a "
            "configured Catalyst qjit/MLIR/QIR runner passes.",
            ("external_framework_comparison", "phase_qnode_claim_boundary"),
            (
                "Catalyst comparison runner evidence is missing",
                "compiled workflow, finite-shot, and device-support parity are unscored",
            ),
            ("Round 4 compiler AD and Program AD", "Round 7 documentation/API readiness"),
        ),
        _row(
            "enzyme_compiler_ad",
            "Enzyme LLVM/MLIR AD over statically analyzable programs, reverse-mode compiler AD, "
            "GPU-kernel AD evidence, and compiler-native benchmarks.",
            "The real Enzyme/LLVM toolchain now executes reverse-mode AD over scalar, vector and "
            "matrix C kernels with bit-exact gradients on the reference toolchain (2026-06-22 "
            "slice 2a, captured as gated evidence when the toolchain is absent), alongside the "
            "native LLVM scalar execution and MLIR maturity artefacts; operator breadth beyond "
            "those kernel families and isolated benchmark IDs remain open.",
            ("external_framework_comparison", "phase_qnode_claim_boundary"),
            (
                "compiler-AD breadth beyond scalar/vector/matrix kernels is incomplete",
                "isolated Enzyme/MLIR benchmark attachment is missing",
            ),
            ("Round 4 compiler AD and Program AD", "Round 6 benchmark promotion"),
        ),
        _row(
            "rust_native_program_ad",
            "Rust-native Program AD value and gradient replay, executed control-flow replay, "
            "array adjoints, registry metadata mirror, and safe PyO3 bindings.",
            "Rust Program AD now replays scalar primitives, view-aliasing (reshape, transpose, "
            "slice, matmul, matvec) and static linear algebra of arbitrary dimension (trace, "
            "determinant, inverse, solve; closed-form 2x2/3x3 and LU/Gauss-Jordan for 4x4 and up) "
            "through the unrolled scalar SSA, plus bounded elementwise shaped arrays with "
            "scalar-to-array broadcasting and static structural array opcodes for reshape, ravel, "
            "broadcast_to, reversed-axis transpose, static-axis concatenate/stack assembly, and "
            "static source-map indexing plus static-axis and scalar all-axis "
            "sum/mean/prod/var/std reductions. "
            "These paths are parity-verified against the "
            "Python/NumPy reference and CI-checked via cargo tests (2026-06-22 slices 2b-1..2b-6; "
            "2026-07-03 elementwise, structural array, structural assembly, static-axis reduction, "
            "static source-map indexing, static product reduction, and static variance/std "
            "reduction adjoints); a general "
            "structural ndarray adjoint engine, executable registry promotion and Rust-side LLVM/JIT "
            "lowering remain open.",
            ("phase_qnode_claim_boundary",),
            (
                "non-lowered dynamic indexing semantics, dynamic axes, ddof/correction metadata, zero-variance std gradients, selector/order-statistic reduction adjoints, and broad linalg array adjoints are missing",
                "executable registry promotion, Rust-side LLVM/JIT lowering, and isolated benchmark evidence are missing",
            ),
            ("Round 4 compiler AD and Program AD", "Round 5 Rustification readiness"),
        ),
        _row(
            "provider_hardware_gradients",
            "Ticketed provider/QPU gradients with backend allowlists, shot budgets, raw counts, "
            "calibration snapshots, reference simulators, and replayable hardware evidence.",
            "Provider/hardware preparation policies and no-submit artifacts exist; real hardware-gradient "
            "execution remains behind live-ticket and benchmark gates.",
            ("phase_qnode_claim_boundary",),
            (
                "live-ticket hardware-gradient evidence is missing",
                "raw-count, calibration, reference-simulator, and isolated benchmark attachments are missing",
            ),
            ("Round 3 quantum provider/plugin breadth", "Round 6 benchmark promotion"),
        ),
        _row(
            "benchmark_promotion",
            "Production performance claims require isolated affinity, host-load context, governor/frequency "
            "metadata, raw timing rows, memory rows, dependency versions, and reproducible artifacts.",
            "Local non-isolated functional evidence exists; isolated runner registration and first promotion "
            "artifact batch remain open.",
            ("ci_benchmark_evidence", "external_validation_artifact_bundle"),
            (
                "isolated benchmark raw artifacts are missing",
                "claim-ledger rows do not carry promoted isolated benchmark IDs",
            ),
            ("Round 6 benchmark promotion", "Round 8 release/promotion gate"),
        ),
        _row(
            "docs_api_maintainability",
            "Users need stable APIs, strict typing, full public docstrings, docs, generated manifests, "
            "module-specific tests, claim-ledger alignment, and public-language guards.",
            "Claim ledger, module-hardening audit, hardening-slice gate, generated manifest, and docs exist; "
            "the new scorecard keeps SOTA wording bounded until rows are promoted.",
            (
                "support_surface_alignment",
                "hardening_slice_gate",
                "module_hardening_audit",
                "public_claim_table",
            ),
            (
                "repository-wide strict mypy/docstring debt remains outside this slice",
                "external reviewer reproduction remains a release gate",
            ),
            (
                "Round 0 surface integrity and maintainability",
                "Round 7 documentation/API readiness",
            ),
        ),
        _row(
            "adoption_licensing",
            "A reusable differentiable computing stack needs clear licensing, install routes, contributor "
            "onboarding, permissive-core decision if needed, and hidden-state-free examples.",
            "AGPL/commercial route is documented and release gates exist; permissive core split and maintainer "
            "onboarding remain product decisions.",
            ("phase_qnode_claim_boundary", "public_claim_table"),
            (
                "permissive scpn-quantum-core split is undecided",
                "maintainer onboarding issues should wait until route classification stabilizes",
            ),
            ("Round 7 documentation/API readiness", "Round 8 release/promotion gate"),
        ),
    )
    return tuple(_attach_surfaces(spec, claim_rows) for spec in row_specs)


def _row(
    category: DifferentiableSOTACategory,
    baseline: str,
    current_evidence: str,
    claim_ids: tuple[str, ...],
    blockers: tuple[str, ...],
    rounds: tuple[str, ...],
) -> _ScorecardRowSpec:
    return _ScorecardRowSpec(
        category=category,
        baseline=baseline,
        current_evidence=current_evidence,
        claim_ids=claim_ids,
        blockers=blockers,
        next_hardening_rounds=rounds,
    )


def _attach_surfaces(
    spec: _ScorecardRowSpec,
    claim_rows: Mapping[str, ClaimLedgerRow],
) -> DifferentiableSOTAScorecardRow:
    claim_ids = spec.claim_ids
    referenced_rows = tuple(
        claim_rows[claim_id] for claim_id in claim_ids if claim_id in claim_rows
    )
    return DifferentiableSOTAScorecardRow(
        category=spec.category,
        baseline=spec.baseline,
        current_evidence=spec.current_evidence,
        status="behind_baseline",
        claim_ids=claim_ids,
        implementation_surface=_unique_paths(
            path for row in referenced_rows for path in row.implementation_surface
        ),
        test_surface=_unique_paths(path for row in referenced_rows for path in row.test_surface)
        + ("tests/test_differentiable_sota_scorecard.py",),
        docs_surface=_unique_paths(path for row in referenced_rows for path in row.docs_surface)
        + (
            "docs/differentiable_api.md",
            "docs/differentiable_programming.md",
            "data/differentiable_phase_qnode/differentiable_sota_scorecard_20260620.md",
        ),
        benchmark_artifact_ids=_unique_paths(
            artifact for row in referenced_rows for artifact in row.benchmark_artifact_ids
        )
        + (DIFFERENTIABLE_SOTA_SCORECARD_ARTIFACT_ID,),
        blockers=spec.blockers,
        next_hardening_rounds=spec.next_hardening_rounds,
        claim_boundary=DIFFERENTIABLE_SOTA_SCORECARD_CLAIM_BOUNDARY,
    )


def _validate_scorecard_row(
    row: DifferentiableSOTAScorecardRow,
    *,
    claim_rows: Mapping[str, ClaimLedgerRow],
    errors: list[str],
) -> None:
    missing_claims = tuple(claim_id for claim_id in row.claim_ids if claim_id not in claim_rows)
    for claim_id in missing_claims:
        errors.append(f"{row.category}: unknown claim-ledger row: {claim_id}")
    if row.ready_for_promotion:
        referenced = tuple(
            claim_rows[claim_id] for claim_id in row.claim_ids if claim_id in claim_rows
        )
        if not referenced or any(claim.promotion_status != "promoted" for claim in referenced):
            errors.append(f"{row.category}: ready status requires promoted ledger rows")
        if not row.benchmark_artifact_ids:
            errors.append(f"{row.category}: ready status requires benchmark artefact IDs")
    if row.status == "behind_baseline" and row.ready_for_promotion:
        errors.append(f"{row.category}: behind-baseline row cannot be promotion-ready")


def _row_paths(row: DifferentiableSOTAScorecardRow) -> Iterable[str]:
    yield from row.implementation_surface
    yield from row.test_surface
    yield from row.docs_surface


def _unique_paths(paths: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(path for path in paths if path))


def _markdown_cell(value: str) -> str:
    return value.replace("\n", " ").replace("|", "\\|")


def _load_public_sota_texts(
    *,
    public_paths: Iterable[str],
    repo_root: Path,
) -> dict[str, str]:
    texts: dict[str, str] = {}
    for relative_path in public_paths:
        path = repo_root / relative_path
        if path.exists():
            texts[relative_path] = path.read_text(encoding="utf-8")
    return texts


def _promotional_phrases(line: str) -> tuple[str, ...]:
    lowered = line.casefold()
    return tuple(phrase for phrase in PROMOTIONAL_LANGUAGE_PHRASES if phrase in lowered)


def _is_bounded_promotion_language(line: str) -> bool:
    lowered = line.casefold()
    return any(marker in lowered for marker in BOUNDED_PROMOTION_LANGUAGE_MARKERS)


def _referenced_sota_categories(line: str) -> tuple[DifferentiableSOTACategory, ...]:
    lowered = line.casefold()
    categories: list[DifferentiableSOTACategory] = []
    for category, markers in _CATEGORY_LANGUAGE_MARKERS.items():
        category_marker = category.replace("_", " ")
        if category in lowered or category_marker in lowered:
            categories.append(category)
            continue
        if any(marker in lowered for marker in markers):
            categories.append(category)
    return tuple(dict.fromkeys(categories))


__all__ = [
    "BOUNDED_PROMOTION_LANGUAGE_MARKERS",
    "DEFAULT_PUBLIC_SOTA_LANGUAGE_PATHS",
    "DIFFERENTIABLE_SOTA_SCORECARD_ARTIFACT_ID",
    "DIFFERENTIABLE_SOTA_SCORECARD_CLAIM_BOUNDARY",
    "DIFFERENTIABLE_SOTA_SCORECARD_SCHEMA",
    "PROMOTIONAL_LANGUAGE_PHRASES",
    "READY_STATUSES",
    "REQUIRED_SOTA_CATEGORIES",
    "DifferentiableSOTACategory",
    "DifferentiableSOTAPromotionLanguageAudit",
    "DifferentiableSOTAScorecard",
    "DifferentiableSOTAScorecardRow",
    "DifferentiableSOTAScorecardValidation",
    "DifferentiableSOTAStatus",
    "audit_differentiable_sota_promotion_language",
    "render_differentiable_sota_scorecard_markdown",
    "run_differentiable_sota_scorecard",
    "validate_differentiable_sota_scorecard",
]
