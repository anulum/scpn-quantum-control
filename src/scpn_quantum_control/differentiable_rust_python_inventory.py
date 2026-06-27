# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable Rust/Python surface inventory.
"""Rust/Python inventory governance for differentiable rustification planning."""

from __future__ import annotations

from collections import Counter
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

DifferentiableRustPythonInventoryClassification = Literal[
    "rust_backed",
    "python_reference",
    "metadata_only",
    "compiler_native_not_rust",
    "provider_blocked",
    "hardware_blocked",
    "deprecate_before_promotion",
]
DifferentiableRustPythonInventoryBenchmarkStatus = Literal[
    "not_applicable",
    "not_run",
    "functional_non_isolated",
    "isolated_required",
    "blocked",
]
DifferentiableRustPythonInventoryRustParityStatus = Literal[
    "complete",
    "partial",
    "missing",
    "not_applicable",
]
DifferentiableRustPythonInventoryPolyglotStatus = Literal[
    "complete",
    "partial",
    "missing",
    "not_applicable",
]

DIFFERENTIABLE_RUST_PYTHON_INVENTORY_SCHEMA = "scpn_qc_differentiable_rust_python_inventory_v1"
DIFFERENTIABLE_RUST_PYTHON_INVENTORY_ARTIFACT_ID = "diff-rust-python-inventory-20260620"
DIFFERENTIABLE_RUST_PYTHON_INVENTORY_CLAIM_BOUNDARY = (
    "Differentiable Rust/Python surface inventory for rustification planning; "
    "no broad rustification promotion, provider execution, hardware execution, "
    "LLVM/JIT execution, GPU execution, or isolated benchmark claim is implied."
)
REQUIRED_INVENTORY_CLASSIFICATIONS: tuple[DifferentiableRustPythonInventoryClassification, ...] = (
    "rust_backed",
    "python_reference",
    "metadata_only",
    "compiler_native_not_rust",
    "provider_blocked",
    "hardware_blocked",
    "deprecate_before_promotion",
)
READY_BENCHMARK_STATUSES: frozenset[DifferentiableRustPythonInventoryBenchmarkStatus] = frozenset(
    {"not_applicable", "isolated_required"}
)


@dataclass(frozen=True)
class DifferentiableRustPythonInventoryRow:
    """One classified differentiable surface in the Rust/Python inventory."""

    surface_id: str
    title: str
    classification: DifferentiableRustPythonInventoryClassification
    owner_module: str
    public_api: tuple[str, ...]
    python_surface: tuple[str, ...]
    rust_surface: tuple[str, ...]
    polyglot_surface: tuple[str, ...]
    test_surface: tuple[str, ...]
    docs_surface: tuple[str, ...]
    benchmark_surface: tuple[str, ...]
    claim_ids: tuple[str, ...]
    mypy_target: str
    docstring_status: str
    benchmark_status: DifferentiableRustPythonInventoryBenchmarkStatus
    rust_parity_status: DifferentiableRustPythonInventoryRustParityStatus
    polyglot_status: DifferentiableRustPythonInventoryPolyglotStatus
    blockers: tuple[str, ...]
    next_hardening_rounds: tuple[str, ...]
    claim_boundary: str

    def __post_init__(self) -> None:
        """Validate one inventory row before it can enter governance artefacts."""
        if self.classification not in REQUIRED_INVENTORY_CLASSIFICATIONS:
            raise ValueError(f"unknown inventory classification: {self.classification}")
        for field_name in (
            "surface_id",
            "title",
            "owner_module",
            "mypy_target",
            "docstring_status",
            "claim_boundary",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")
        for field_name in (
            "public_api",
            "python_surface",
            "rust_surface",
            "polyglot_surface",
            "test_surface",
            "docs_surface",
            "benchmark_surface",
            "claim_ids",
            "next_hardening_rounds",
        ):
            value = getattr(self, field_name)
            if not value or any(not str(item).strip() for item in value):
                raise ValueError(f"{field_name} must contain non-empty entries")

    @property
    def rustification_ready(self) -> bool:
        """Return whether this row can be used as a Rust promotion input."""
        return (
            self.classification == "rust_backed"
            and self.rust_parity_status == "complete"
            and self.polyglot_status in {"complete", "not_applicable"}
            and self.benchmark_status in READY_BENCHMARK_STATUSES
            and not self.blockers
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready inventory row."""
        return {
            "surface_id": self.surface_id,
            "title": self.title,
            "classification": self.classification,
            "owner_module": self.owner_module,
            "public_api": list(self.public_api),
            "python_surface": list(self.python_surface),
            "rust_surface": list(self.rust_surface),
            "polyglot_surface": list(self.polyglot_surface),
            "test_surface": list(self.test_surface),
            "docs_surface": list(self.docs_surface),
            "benchmark_surface": list(self.benchmark_surface),
            "claim_ids": list(self.claim_ids),
            "mypy_target": self.mypy_target,
            "docstring_status": self.docstring_status,
            "benchmark_status": self.benchmark_status,
            "rust_parity_status": self.rust_parity_status,
            "polyglot_status": self.polyglot_status,
            "blockers": list(self.blockers),
            "next_hardening_rounds": list(self.next_hardening_rounds),
            "rustification_ready": self.rustification_ready,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class DifferentiableRustPythonInventory:
    """Deterministic differentiable Rust/Python surface inventory."""

    schema: str
    artifact_id: str
    rows: tuple[DifferentiableRustPythonInventoryRow, ...]
    rustification_ready: bool
    ready_surface_count: int
    total_surface_count: int
    classification_counts: Mapping[str, int]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready inventory payload."""
        return {
            "schema": self.schema,
            "artifact_id": self.artifact_id,
            "rustification_ready": self.rustification_ready,
            "ready_surface_count": self.ready_surface_count,
            "total_surface_count": self.total_surface_count,
            "classification_counts": dict(self.classification_counts),
            "claim_boundary": self.claim_boundary,
            "rows": [row.to_dict() for row in self.rows],
        }


@dataclass(frozen=True)
class DifferentiableRustPythonInventoryValidation:
    """Validation result for a Rust/Python surface inventory."""

    passed: bool
    errors: tuple[str, ...]
    checked_surface_ids: tuple[str, ...]
    checked_claim_ids: tuple[str, ...]
    checked_paths: tuple[str, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready inventory validation evidence."""
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "checked_surface_ids": list(self.checked_surface_ids),
            "checked_claim_ids": list(self.checked_claim_ids),
            "checked_paths": list(self.checked_paths),
            "claim_boundary": self.claim_boundary,
        }


def run_differentiable_rust_python_inventory(
    *,
    ledger: ClaimLedger | None = None,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> DifferentiableRustPythonInventory:
    """Build the deterministic rustification inventory from committed surfaces."""
    loaded_ledger = load_differentiable_claim_ledger(ledger_path) if ledger is None else ledger
    claim_rows = {row.claim_id: row for row in loaded_ledger.rows}
    rows = _default_inventory_rows(claim_rows)
    ready_count = sum(1 for row in rows if row.rustification_ready)
    counts = Counter(row.classification for row in rows)
    classification_counts: dict[str, int] = {
        classification: counts.get(classification, 0)
        for classification in REQUIRED_INVENTORY_CLASSIFICATIONS
    }
    return DifferentiableRustPythonInventory(
        schema=DIFFERENTIABLE_RUST_PYTHON_INVENTORY_SCHEMA,
        artifact_id=DIFFERENTIABLE_RUST_PYTHON_INVENTORY_ARTIFACT_ID,
        rows=rows,
        rustification_ready=ready_count == len(rows),
        ready_surface_count=ready_count,
        total_surface_count=len(rows),
        classification_counts=classification_counts,
        claim_boundary=DIFFERENTIABLE_RUST_PYTHON_INVENTORY_CLAIM_BOUNDARY,
    )


def validate_differentiable_rust_python_inventory(
    inventory: DifferentiableRustPythonInventory,
    *,
    ledger: ClaimLedger | None = None,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    repo_root: Path = REPO_ROOT,
) -> DifferentiableRustPythonInventoryValidation:
    """Validate rustification inventory paths, claims, and readiness invariants."""
    loaded_ledger = load_differentiable_claim_ledger(ledger_path) if ledger is None else ledger
    claim_rows = {row.claim_id: row for row in loaded_ledger.rows}
    errors: list[str] = []
    checked_paths: set[str] = set()
    checked_claim_ids: set[str] = set()
    surface_ids = tuple(row.surface_id for row in inventory.rows)

    if inventory.schema != DIFFERENTIABLE_RUST_PYTHON_INVENTORY_SCHEMA:
        errors.append(f"unexpected inventory schema: {inventory.schema}")
    if inventory.total_surface_count != len(inventory.rows):
        errors.append("total_surface_count does not match row count")
    ready_count = sum(1 for row in inventory.rows if row.rustification_ready)
    if inventory.ready_surface_count != ready_count:
        errors.append("ready_surface_count does not match ready rows")
    if inventory.rustification_ready != (ready_count == len(inventory.rows)):
        errors.append("rustification_ready does not match ready row count")
    missing_classifications = tuple(
        classification
        for classification in REQUIRED_INVENTORY_CLASSIFICATIONS
        if classification not in inventory.classification_counts
    )
    for classification in missing_classifications:
        errors.append(f"classification_counts missing {classification}")

    for surface_id in _duplicates(surface_ids):
        errors.append(f"duplicate inventory surface_id: {surface_id}")
    for row in inventory.rows:
        checked_claim_ids.update(row.claim_ids)
        _validate_inventory_row(row, claim_rows=claim_rows, errors=errors)
        for path in _row_paths(row):
            checked_paths.add(path)
            if not (repo_root / path).exists():
                errors.append(f"{row.surface_id}: evidence path does not exist: {path}")

    return DifferentiableRustPythonInventoryValidation(
        passed=not errors,
        errors=tuple(errors),
        checked_surface_ids=surface_ids,
        checked_claim_ids=tuple(sorted(checked_claim_ids)),
        checked_paths=tuple(sorted(checked_paths)),
        claim_boundary=(
            "Rust/Python inventory validation only; validates declared surfaces, "
            "claim-ledger references, and readiness invariants without promoting "
            "provider, hardware, GPU, LLVM/JIT, or isolated benchmark claims."
        ),
    )


def render_differentiable_rust_python_inventory_markdown(
    inventory: DifferentiableRustPythonInventory,
) -> str:
    """Render a reviewer-facing Markdown summary of the Rust/Python inventory."""
    lines = [
        "<!--",
        "SPDX-License-Identifier: AGPL-3.0-or-later",
        "Commercial license available",
        "© Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "© Code 2020–2026 Miroslav Šotek. All rights reserved.",
        "ORCID: 0009-0009-3560-0851",
        "Contact: www.anulum.li | protoscience@anulum.li",
        "SCPN Quantum Control — Differentiable Rust/Python Surface Inventory",
        "-->",
        "",
        "# Differentiable Rust/Python Surface Inventory",
        "",
        f"- Schema: `{inventory.schema}`",
        f"- Artifact ID: `{inventory.artifact_id}`",
        f"- Rustification ready: `{inventory.rustification_ready}`",
        f"- Ready surfaces: `{inventory.ready_surface_count}/{inventory.total_surface_count}`",
        f"- Claim boundary: {inventory.claim_boundary}",
        "",
        "| Surface | Classification | Rust parity | Polyglot | Benchmark | Blockers |",
        "|---|---|---|---|---|---|",
    ]
    for row in inventory.rows:
        lines.append(
            "| `{surface}` | `{classification}` | `{rust}` | `{polyglot}` | "
            "`{benchmark}` | {blockers} |".format(
                surface=row.surface_id,
                classification=row.classification,
                rust=row.rust_parity_status,
                polyglot=row.polyglot_status,
                benchmark=row.benchmark_status,
                blockers=_markdown_cell("<br>".join(row.blockers) or "none"),
            )
        )
    lines.append("")
    lines.append(
        "Rows are planning evidence for the rustification queue. A row becoming "
        "`rustification_ready` does not by itself promote public performance, "
        "provider, hardware, GPU, LLVM/JIT, or isolated benchmark claims."
    )
    return "\n".join(lines)


def _default_inventory_rows(
    claim_rows: Mapping[str, ClaimLedgerRow],
) -> tuple[DifferentiableRustPythonInventoryRow, ...]:
    rows = (
        _inventory_row(
            "unified_differentiable_api",
            "Unified differentiable API and dashboard facade",
            "python_reference",
            "src/scpn_quantum_control/differentiable_api.py",
            ("differentiable_api", "differentiable_dashboard_status"),
            ("src/scpn_quantum_control/differentiable_api.py",),
            ("scpn_quantum_engine/src/lib.rs",),
            ("docs/differentiable_api.md",),
            ("tests/test_differentiable_api.py",),
            ("docs/differentiable_api.md", "docs/differentiable_programming.md"),
            ("benchmark_report",),
            ("support_surface_alignment", "phase_qnode_claim_boundary"),
            "functional_non_isolated",
            "partial",
            "partial",
            (
                "public orchestration remains Python-first",
                "dashboard rows still include metadata-only and blocked routes",
            ),
        ),
        _inventory_row(
            "rust_program_ad_ir",
            "Rust Program AD IR parser plus bounded scalar primitive, value+gradient, and executed branch replay",
            "rust_backed",
            "src/scpn_quantum_control/differentiable.py",
            (
                "program_ad_registry_dispatch_coverage_report",
                "compile_whole_program_frontend",
                "value_and_grad_program_ad_effect_ir_with_rust",
            ),
            ("src/scpn_quantum_control/differentiable.py", "src/scpn_quantum_engine.pyi"),
            (
                "scpn_quantum_engine/src/program_ad_ir.rs",
                "scpn_quantum_engine/tests/program_ad_ir.rs",
            ),
            ("docs/differentiable_api.md",),
            (
                "tests/test_phase_qnode_rust_parity.py",
                "tests/test_program_ad_effect_ir.py",
                "tests/test_program_ad_rust_bridge.py",
            ),
            ("docs/differentiable_api.md", "docs/differentiable_programming.md"),
            (
                "program_ad_rust_value_gradient_replay",
                "program_ad_rust_executed_branch_replay",
                "program_ad_rust_scalar_primitive_family_replay",
            ),
            ("phase_qnode_claim_boundary", "module_hardening_audit"),
            "functional_non_isolated",
            "partial",
            "partial",
            (
                "array adjoints are missing",
                "registry metadata mirror, LLVM/JIT lowering, and isolated benchmark evidence are missing",
            ),
        ),
        _inventory_row(
            "whole_program_frontend",
            "Static whole-program bytecode/source compiler frontend preflight",
            "python_reference",
            "src/scpn_quantum_control/whole_program_frontend.py",
            (
                "compile_whole_program_frontend",
                "WholeProgramCompilerFrontendReport",
                "WholeProgramSemanticsReport",
            ),
            ("src/scpn_quantum_control/whole_program_frontend.py",),
            ("scpn_quantum_engine/src/lib.rs",),
            ("docs/differentiable_api.md",),
            ("tests/test_whole_program_frontend.py", "tests/test_whole_program_ad_contracts.py"),
            ("docs/differentiable_api.md", "docs/differentiable_programming.md"),
            ("program_ad_bytecode_source_frontend",),
            ("module_hardening_audit", "support_surface_alignment"),
            "not_applicable",
            "not_applicable",
            "partial",
            (
                "static Python source/bytecode metadata has no executable Rust parity requirement",
                "compiler lowering remains blocked until a real Program AD backend exists",
            ),
        ),
        _inventory_row(
            "rust_compiler_ad_primitives",
            "Rust compiler-AD primitive kernels exposed through PyO3",
            "rust_backed",
            "src/scpn_quantum_control/compiler/mlir.py",
            (
                "compile_registered_primitive_to_executable",
                "compile_vector_dot_ad_to_native_llvm_jit",
            ),
            ("src/scpn_quantum_control/compiler/mlir.py", "src/scpn_quantum_engine.pyi"),
            ("scpn_quantum_engine/src/compiler_ad.rs", "scpn_quantum_engine/src/lib.rs"),
            ("docs/differentiable_api.md",),
            ("tests/test_phase_qnode_compiler_lowering.py", "tests/test_rust_new_functions.py"),
            ("docs/differentiable_api.md", "docs/rust_engine.md"),
            ("enzyme-native-square-gradient-20260616",),
            ("external_framework_comparison", "phase_qnode_claim_boundary"),
            "functional_non_isolated",
            "partial",
            "partial",
            (
                "isolated compiler-AD benchmark ID is missing",
                "broad LLVM/JIT lowering remains claim-blocked",
            ),
        ),
        _inventory_row(
            "differentiable_sota_scorecard",
            "Differentiable state-of-art category scorecard",
            "metadata_only",
            "src/scpn_quantum_control/differentiable_sota_scorecard.py",
            ("run_differentiable_sota_scorecard",),
            ("src/scpn_quantum_control/differentiable_sota_scorecard.py",),
            ("scpn_quantum_engine/src/lib.rs",),
            ("data/differentiable_phase_qnode/differentiable_sota_scorecard_20260620.md",),
            ("tests/test_differentiable_sota_scorecard.py",),
            ("docs/differentiable_api.md", "docs/differentiable_programming.md"),
            ("diff-sota-scorecard-20260620",),
            ("differentiable_sota_scorecard",),
            "not_applicable",
            "not_applicable",
            "not_applicable",
            ("governance evidence only; no executable Rust surface is required",),
        ),
        _inventory_row(
            "pennylane_plugin_matrix",
            "PennyLane QNode export/import and plugin/provider matrix",
            "provider_blocked",
            "src/scpn_quantum_control/phase/pennylane_bridge.py",
            ("run_pennylane_plugin_matrix", "import_phase_qnode_from_pennylane"),
            (
                "src/scpn_quantum_control/phase/pennylane_bridge.py",
                "src/scpn_quantum_control/phase/pennylane_import.py",
            ),
            ("scpn_quantum_engine/src/lib.rs",),
            ("docs/differentiable_api.md",),
            ("tests/test_phase_pennylane_bridge.py", "tests/test_phase_pennylane_import.py"),
            ("docs/differentiable_api.md", "docs/quantum_gradients.md"),
            ("pennylane_plugin_matrix",),
            ("external_framework_comparison", "phase_qnode_claim_boundary"),
            "blocked",
            "not_applicable",
            "partial",
            (
                "provider-plugin execution artefacts are missing",
                "hardware-plugin and provider-gradient parity artefacts are missing",
            ),
        ),
        _inventory_row(
            "qiskit_runtime_provider_gradients",
            "Qiskit Runtime/provider-gradient workflow evidence chain",
            "provider_blocked",
            "src/scpn_quantum_control/phase/qiskit_bridge.py",
            ("run_qiskit_maturity_audit",),
            (
                "src/scpn_quantum_control/phase/qiskit_bridge.py",
                "src/scpn_quantum_control/phase/provider_hardware_gradient_audit.py",
            ),
            ("scpn_quantum_engine/src/lib.rs",),
            ("docs/quantum_gradients.md",),
            (
                "tests/test_phase_qiskit_bridge.py",
                "tests/test_phase_provider_hardware_gradient_audit.py",
            ),
            ("docs/differentiable_api.md", "docs/quantum_gradients.md"),
            ("qiskit_runtime_qpu_provider_evidence_bundle",),
            ("phase_qnode_claim_boundary",),
            "blocked",
            "not_applicable",
            "partial",
            (
                "live-ticket Runtime/QPU evidence is missing",
                "provider-gradient methods are not attached to a live-ticket run",
            ),
        ),
        _inventory_row(
            "hardware_gradient_campaigns",
            "Ticketed provider and QPU hardware-gradient campaigns",
            "hardware_blocked",
            "src/scpn_quantum_control/phase/hardware_gradient_campaign.py",
            ("HardwareGradientCampaign", "run_hardware_gradient_campaign_plan"),
            (
                "src/scpn_quantum_control/phase/hardware_gradient_campaign.py",
                "src/scpn_quantum_control/phase/hardware_gradient_policy.py",
            ),
            ("scpn_quantum_engine/src/lib.rs",),
            ("docs/quantum_gradients.md",),
            (
                "tests/test_phase_hardware_gradient_campaign.py",
                "tests/test_phase_hardware_gradient_policy.py",
            ),
            ("docs/quantum_gradients.md", "docs/qpu_system_capability_dossier.md"),
            ("hardware_gradient_live_ticket",),
            ("phase_qnode_claim_boundary",),
            "blocked",
            "not_applicable",
            "partial",
            (
                "live-ticket hardware execution is missing",
                "raw counts, calibration, and reference simulator attachments are missing",
            ),
        ),
        _inventory_row(
            "catalyst_compiler_comparison",
            "Catalyst-style compiler workflow comparison row",
            "deprecate_before_promotion",
            "src/scpn_quantum_control/benchmarks/differentiable_external_comparison.py",
            ("run_differentiable_external_comparison_suite",),
            ("src/scpn_quantum_control/benchmarks/differentiable_external_comparison.py",),
            ("scpn_quantum_engine/src/compiler_ad.rs",),
            ("docs/differentiable_programming.md",),
            ("tests/test_differentiable_external_comparisons.py",),
            ("docs/differentiable_programming.md", "docs/differentiable_api.md"),
            ("catalyst_runner_missing",),
            ("differentiable_sota_scorecard",),
            "blocked",
            "partial",
            "partial",
            (
                "configured Catalyst qjit/MLIR/QIR runner evidence is missing",
                "compiled quantum-classical workflow parity is unimplemented",
            ),
        ),
        _inventory_row(
            "enzyme_mlir_compiler_ad",
            "Enzyme/MLIR compiler-AD maturity and breadth evidence",
            "compiler_native_not_rust",
            "src/scpn_quantum_control/compiler/mlir.py",
            ("run_enzyme_mlir_maturity_audit", "build_enzyme_mlir_compiler_ad_breadth_artifact"),
            ("src/scpn_quantum_control/compiler/mlir.py",),
            ("scpn_quantum_engine/src/compiler_ad.rs",),
            ("data/differentiable_phase_qnode/enzyme_mlir_maturity_audit_20260616.md",),
            ("tests/test_phase_qnode_compiler_lowering.py",),
            ("docs/differentiable_programming.md", "docs/differentiable_api.md"),
            ("enzyme_mlir_maturity_audit_20260616",),
            ("external_framework_comparison", "phase_qnode_claim_boundary"),
            "functional_non_isolated",
            "partial",
            "partial",
            (
                "11-case compiler-AD breadth evidence is incomplete",
                "isolated Enzyme/MLIR benchmark attachment is missing",
            ),
        ),
    )
    return _attach_ledger_surfaces(rows, claim_rows)


def _inventory_row(
    surface_id: str,
    title: str,
    classification: DifferentiableRustPythonInventoryClassification,
    owner_module: str,
    public_api: tuple[str, ...],
    python_surface: tuple[str, ...],
    rust_surface: tuple[str, ...],
    polyglot_surface: tuple[str, ...],
    test_surface: tuple[str, ...],
    docs_surface: tuple[str, ...],
    benchmark_surface: tuple[str, ...],
    claim_ids: tuple[str, ...],
    benchmark_status: DifferentiableRustPythonInventoryBenchmarkStatus,
    rust_parity_status: DifferentiableRustPythonInventoryRustParityStatus,
    polyglot_status: DifferentiableRustPythonInventoryPolyglotStatus,
    blockers: tuple[str, ...],
) -> DifferentiableRustPythonInventoryRow:
    return DifferentiableRustPythonInventoryRow(
        surface_id=surface_id,
        title=title,
        classification=classification,
        owner_module=owner_module,
        public_api=public_api,
        python_surface=python_surface,
        rust_surface=rust_surface,
        polyglot_surface=polyglot_surface,
        test_surface=test_surface,
        docs_surface=docs_surface,
        benchmark_surface=benchmark_surface,
        claim_ids=claim_ids,
        mypy_target=owner_module,
        docstring_status="complete_for_inventory_surface",
        benchmark_status=benchmark_status,
        rust_parity_status=rust_parity_status,
        polyglot_status=polyglot_status,
        blockers=blockers,
        next_hardening_rounds=(
            "Round 0 surface integrity and maintainability",
            "Round 5 Rustification readiness",
        ),
        claim_boundary=DIFFERENTIABLE_RUST_PYTHON_INVENTORY_CLAIM_BOUNDARY,
    )


def _attach_ledger_surfaces(
    rows: tuple[DifferentiableRustPythonInventoryRow, ...],
    claim_rows: Mapping[str, ClaimLedgerRow],
) -> tuple[DifferentiableRustPythonInventoryRow, ...]:
    attached: list[DifferentiableRustPythonInventoryRow] = []
    for row in rows:
        referenced = tuple(
            claim_rows[claim_id] for claim_id in row.claim_ids if claim_id in claim_rows
        )
        attached.append(
            DifferentiableRustPythonInventoryRow(
                surface_id=row.surface_id,
                title=row.title,
                classification=row.classification,
                owner_module=row.owner_module,
                public_api=row.public_api,
                python_surface=_unique_paths(
                    (
                        *row.python_surface,
                        *(path for item in referenced for path in item.implementation_surface),
                    )
                ),
                rust_surface=row.rust_surface,
                polyglot_surface=row.polyglot_surface,
                test_surface=_unique_paths(
                    (
                        *row.test_surface,
                        *(path for item in referenced for path in item.test_surface),
                    )
                ),
                docs_surface=_unique_paths(
                    (
                        *row.docs_surface,
                        *(path for item in referenced for path in item.docs_surface),
                        "data/differentiable_phase_qnode/differentiable_rust_python_inventory_20260620.md",
                    )
                ),
                benchmark_surface=_unique_paths(
                    (
                        *row.benchmark_surface,
                        *(
                            artifact
                            for item in referenced
                            for artifact in item.benchmark_artifact_ids
                        ),
                    )
                ),
                claim_ids=row.claim_ids,
                mypy_target=row.mypy_target,
                docstring_status=row.docstring_status,
                benchmark_status=row.benchmark_status,
                rust_parity_status=row.rust_parity_status,
                polyglot_status=row.polyglot_status,
                blockers=row.blockers,
                next_hardening_rounds=row.next_hardening_rounds,
                claim_boundary=row.claim_boundary,
            )
        )
    return tuple(attached)


def _validate_inventory_row(
    row: DifferentiableRustPythonInventoryRow,
    *,
    claim_rows: Mapping[str, ClaimLedgerRow],
    errors: list[str],
) -> None:
    for claim_id in row.claim_ids:
        if claim_id not in claim_rows:
            errors.append(f"{row.surface_id}: unknown claim-ledger row: {claim_id}")
    if (
        row.classification == "rust_backed"
        and row.rust_parity_status == "complete"
        and row.blockers
    ):
        errors.append(f"{row.surface_id}: ready Rust-backed rows must not carry blockers")
    if row.rustification_ready and row.benchmark_status == "not_run":
        errors.append(f"{row.surface_id}: ready rows require benchmark classification evidence")
    if row.classification in {"provider_blocked", "hardware_blocked"} and not row.blockers:
        errors.append(f"{row.surface_id}: blocked provider/hardware rows must list blockers")


def _row_paths(row: DifferentiableRustPythonInventoryRow) -> Iterable[str]:
    yield row.owner_module
    yield row.mypy_target
    yield from row.python_surface
    yield from row.rust_surface
    yield from row.polyglot_surface
    yield from row.test_surface
    yield from row.docs_surface


def _duplicates(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return tuple(sorted(duplicates))


def _unique_paths(paths: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(path for path in paths if path))


def _markdown_cell(value: str) -> str:
    return value.replace("\n", " ").replace("|", "\\|")


__all__ = [
    "DIFFERENTIABLE_RUST_PYTHON_INVENTORY_ARTIFACT_ID",
    "DIFFERENTIABLE_RUST_PYTHON_INVENTORY_CLAIM_BOUNDARY",
    "DIFFERENTIABLE_RUST_PYTHON_INVENTORY_SCHEMA",
    "READY_BENCHMARK_STATUSES",
    "REQUIRED_INVENTORY_CLASSIFICATIONS",
    "DifferentiableRustPythonInventory",
    "DifferentiableRustPythonInventoryBenchmarkStatus",
    "DifferentiableRustPythonInventoryClassification",
    "DifferentiableRustPythonInventoryPolyglotStatus",
    "DifferentiableRustPythonInventoryRow",
    "DifferentiableRustPythonInventoryRustParityStatus",
    "DifferentiableRustPythonInventoryValidation",
    "render_differentiable_rust_python_inventory_markdown",
    "run_differentiable_rust_python_inventory",
    "validate_differentiable_rust_python_inventory",
]
