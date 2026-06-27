# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable architecture map.
"""Architecture and Rustification map for differentiable-programming governance."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .differentiable_claim_ledger import REPO_ROOT
from .differentiable_rust_python_inventory import (
    DifferentiableRustPythonInventory,
    DifferentiableRustPythonInventoryRow,
    run_differentiable_rust_python_inventory,
    validate_differentiable_rust_python_inventory,
)
from .differentiable_sota_scorecard import (
    DifferentiableSOTACategory,
    DifferentiableSOTAScorecard,
    run_differentiable_sota_scorecard,
    validate_differentiable_sota_scorecard,
)

DifferentiableArchitectureLayerId = Literal[
    "public_api_facade",
    "qnode_framework_bridges",
    "program_ad_core",
    "compiler_ad_native_execution",
    "provider_hardware_boundary",
    "benchmark_and_claim_governance",
]

DIFFERENTIABLE_ARCHITECTURE_MAP_SCHEMA = "scpn_qc_differentiable_architecture_map_v1"
DIFFERENTIABLE_ARCHITECTURE_MAP_ARTIFACT_ID = "diff-architecture-rustification-map-20260627"
DIFFERENTIABLE_ARCHITECTURE_MAP_CLAIM_BOUNDARY = (
    "Differentiable architecture and Rustification routing map only; no broad "
    "Rustification promotion, provider execution, hardware execution, GPU "
    "execution, LLVM/JIT execution, or isolated benchmark claim is implied."
)
REQUIRED_ARCHITECTURE_LAYER_IDS: tuple[DifferentiableArchitectureLayerId, ...] = (
    "public_api_facade",
    "qnode_framework_bridges",
    "program_ad_core",
    "compiler_ad_native_execution",
    "provider_hardware_boundary",
    "benchmark_and_claim_governance",
)


@dataclass(frozen=True)
class DifferentiableArchitectureMapLayer:
    """One architecture layer tied to inventory, scorecard, and evidence paths."""

    layer_id: DifferentiableArchitectureLayerId | str
    title: str
    role: str
    owner_modules: tuple[str, ...]
    inventory_surface_ids: tuple[str, ...]
    sota_categories: tuple[DifferentiableSOTACategory, ...]
    python_surfaces: tuple[str, ...]
    rust_surfaces: tuple[str, ...]
    polyglot_surfaces: tuple[str, ...]
    test_surfaces: tuple[str, ...]
    docs_surfaces: tuple[str, ...]
    benchmark_surfaces: tuple[str, ...]
    blockers: tuple[str, ...]
    next_hardening_rounds: tuple[str, ...]
    claim_boundary: str

    def __post_init__(self) -> None:
        """Validate layer fields before emitting architecture evidence."""
        for field_name in (
            "layer_id",
            "title",
            "role",
            "claim_boundary",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")
        for field_name in (
            "owner_modules",
            "inventory_surface_ids",
            "sota_categories",
            "python_surfaces",
            "rust_surfaces",
            "polyglot_surfaces",
            "test_surfaces",
            "docs_surfaces",
            "benchmark_surfaces",
            "next_hardening_rounds",
        ):
            value = getattr(self, field_name)
            if not value or any(not str(item).strip() for item in value):
                raise ValueError(f"{field_name} must contain non-empty entries")

    @property
    def rustification_ready(self) -> bool:
        """Return whether this layer is free of declared Rustification blockers."""
        return not self.blockers

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready architecture layer."""
        return {
            "layer_id": self.layer_id,
            "title": self.title,
            "role": self.role,
            "owner_modules": list(self.owner_modules),
            "inventory_surface_ids": list(self.inventory_surface_ids),
            "sota_categories": list(self.sota_categories),
            "python_surfaces": list(self.python_surfaces),
            "rust_surfaces": list(self.rust_surfaces),
            "polyglot_surfaces": list(self.polyglot_surfaces),
            "test_surfaces": list(self.test_surfaces),
            "docs_surfaces": list(self.docs_surfaces),
            "benchmark_surfaces": list(self.benchmark_surfaces),
            "blockers": list(self.blockers),
            "next_hardening_rounds": list(self.next_hardening_rounds),
            "rustification_ready": self.rustification_ready,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class DifferentiableArchitectureMap:
    """Deterministic architecture map for differentiable Rustification routing."""

    schema: str
    artifact_id: str
    layers: tuple[DifferentiableArchitectureMapLayer, ...]
    rustification_ready: bool
    ready_layer_count: int
    total_layer_count: int
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready architecture map."""
        return {
            "schema": self.schema,
            "artifact_id": self.artifact_id,
            "rustification_ready": self.rustification_ready,
            "ready_layer_count": self.ready_layer_count,
            "total_layer_count": self.total_layer_count,
            "claim_boundary": self.claim_boundary,
            "layers": [layer.to_dict() for layer in self.layers],
        }


@dataclass(frozen=True)
class DifferentiableArchitectureMapValidation:
    """Validation result for a differentiable architecture map."""

    passed: bool
    errors: tuple[str, ...]
    checked_layer_ids: tuple[str, ...]
    checked_inventory_surface_ids: tuple[str, ...]
    checked_sota_categories: tuple[DifferentiableSOTACategory, ...]
    checked_paths: tuple[str, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready architecture-map validation evidence."""
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "checked_layer_ids": list(self.checked_layer_ids),
            "checked_inventory_surface_ids": list(self.checked_inventory_surface_ids),
            "checked_sota_categories": list(self.checked_sota_categories),
            "checked_paths": list(self.checked_paths),
            "claim_boundary": self.claim_boundary,
        }


def run_differentiable_architecture_map(
    *,
    inventory: DifferentiableRustPythonInventory | None = None,
    scorecard: DifferentiableSOTAScorecard | None = None,
) -> DifferentiableArchitectureMap:
    """Build the architecture and Rustification map from committed evidence."""
    loaded_inventory = (
        run_differentiable_rust_python_inventory() if inventory is None else inventory
    )
    loaded_scorecard = run_differentiable_sota_scorecard() if scorecard is None else scorecard
    inventory_rows = {row.surface_id: row for row in loaded_inventory.rows}
    layers = _default_architecture_layers(inventory_rows)
    ready_count = sum(1 for layer in layers if layer.rustification_ready)
    return DifferentiableArchitectureMap(
        schema=DIFFERENTIABLE_ARCHITECTURE_MAP_SCHEMA,
        artifact_id=DIFFERENTIABLE_ARCHITECTURE_MAP_ARTIFACT_ID,
        layers=layers,
        rustification_ready=(
            ready_count == len(layers)
            and loaded_inventory.rustification_ready
            and loaded_scorecard.promotion_ready
        ),
        ready_layer_count=ready_count,
        total_layer_count=len(layers),
        claim_boundary=DIFFERENTIABLE_ARCHITECTURE_MAP_CLAIM_BOUNDARY,
    )


def validate_differentiable_architecture_map(
    architecture_map: DifferentiableArchitectureMap,
    *,
    inventory: DifferentiableRustPythonInventory | None = None,
    scorecard: DifferentiableSOTAScorecard | None = None,
    repo_root: Path = REPO_ROOT,
) -> DifferentiableArchitectureMapValidation:
    """Validate architecture layers, references, paths, and readiness invariants."""
    loaded_inventory = (
        run_differentiable_rust_python_inventory() if inventory is None else inventory
    )
    loaded_scorecard = run_differentiable_sota_scorecard() if scorecard is None else scorecard
    inventory_validation = validate_differentiable_rust_python_inventory(
        loaded_inventory,
        repo_root=repo_root,
    )
    scorecard_validation = validate_differentiable_sota_scorecard(
        loaded_scorecard,
        repo_root=repo_root,
    )
    errors: list[str] = [
        f"inventory validation failed: {error}" for error in inventory_validation.errors
    ]
    errors.extend(f"scorecard validation failed: {error}" for error in scorecard_validation.errors)
    inventory_surface_ids = {row.surface_id for row in loaded_inventory.rows}
    sota_categories = {row.category for row in loaded_scorecard.rows}
    checked_paths: set[str] = set()
    checked_inventory_ids: set[str] = set()
    checked_sota_categories: set[DifferentiableSOTACategory] = set()
    layer_ids = tuple(str(layer.layer_id) for layer in architecture_map.layers)

    if architecture_map.schema != DIFFERENTIABLE_ARCHITECTURE_MAP_SCHEMA:
        errors.append(f"unexpected architecture-map schema: {architecture_map.schema}")
    if architecture_map.total_layer_count != len(architecture_map.layers):
        errors.append("total_layer_count does not match layer count")
    ready_count = sum(1 for layer in architecture_map.layers if layer.rustification_ready)
    if architecture_map.ready_layer_count != ready_count:
        errors.append("ready_layer_count does not match ready layers")
    expected_ready = (
        ready_count == len(architecture_map.layers)
        and loaded_inventory.rustification_ready
        and loaded_scorecard.promotion_ready
    )
    if architecture_map.rustification_ready != expected_ready:
        errors.append("rustification_ready does not match inventory and scorecard readiness")
    if layer_ids != tuple(REQUIRED_ARCHITECTURE_LAYER_IDS):
        errors.append("architecture layer IDs must match REQUIRED_ARCHITECTURE_LAYER_IDS exactly")
    for layer_id in _duplicates(layer_ids):
        errors.append(f"duplicate architecture layer_id: {layer_id}")

    for layer in architecture_map.layers:
        for surface_id in layer.inventory_surface_ids:
            checked_inventory_ids.add(surface_id)
            if surface_id not in inventory_surface_ids:
                errors.append(f"{layer.layer_id}: unknown inventory surface: {surface_id}")
        for category in layer.sota_categories:
            checked_sota_categories.add(category)
            if category not in sota_categories:
                errors.append(f"{layer.layer_id}: unknown SOTA category: {category}")
        if architecture_map.rustification_ready and layer.blockers:
            errors.append(f"{layer.layer_id}: ready architecture layers must not carry blockers")
        for path in _layer_paths(layer):
            checked_paths.add(path)
            if not (repo_root / path).exists():
                errors.append(f"{layer.layer_id}: evidence path does not exist: {path}")

    return DifferentiableArchitectureMapValidation(
        passed=not errors,
        errors=tuple(errors),
        checked_layer_ids=layer_ids,
        checked_inventory_surface_ids=tuple(sorted(checked_inventory_ids)),
        checked_sota_categories=tuple(sorted(checked_sota_categories)),
        checked_paths=tuple(sorted(checked_paths)),
        claim_boundary=(
            "Architecture-map validation only; validates layer routing, existing "
            "evidence paths, inventory references, and scorecard references "
            "without promoting Rust, LLVM/JIT, provider, hardware, GPU, or "
            "isolated benchmark claims."
        ),
    )


def render_differentiable_architecture_map_markdown(
    architecture_map: DifferentiableArchitectureMap,
) -> str:
    """Render a reviewer-facing Markdown summary of the architecture map."""
    lines = [
        "<!--",
        "SPDX-License-Identifier: AGPL-3.0-or-later",
        "Commercial license available",
        "© Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "© Code 2020–2026 Miroslav Šotek. All rights reserved.",
        "ORCID: 0009-0009-3560-0851",
        "Contact: www.anulum.li | protoscience@anulum.li",
        "SCPN Quantum Control — Differentiable Architecture and Rustification Map",
        "-->",
        "",
        "# Differentiable Architecture and Rustification Map",
        "",
        f"- Schema: `{architecture_map.schema}`",
        f"- Artifact ID: `{architecture_map.artifact_id}`",
        f"- Rustification ready: `{architecture_map.rustification_ready}`",
        f"- Ready layers: `{architecture_map.ready_layer_count}/{architecture_map.total_layer_count}`",
        f"- Claim boundary: {architecture_map.claim_boundary}",
        "",
        "| Layer | Inventory rows | SOTA categories | Blockers | Next rounds |",
        "|---|---|---|---|---|",
    ]
    for layer in architecture_map.layers:
        lines.append(
            "| `{layer}` | {inventory} | {categories} | {blockers} | {rounds} |".format(
                layer=layer.layer_id,
                inventory=_markdown_cell("<br>".join(layer.inventory_surface_ids)),
                categories=_markdown_cell("<br>".join(layer.sota_categories)),
                blockers=_markdown_cell("<br>".join(layer.blockers) or "none"),
                rounds=_markdown_cell("<br>".join(layer.next_hardening_rounds)),
            )
        )
    lines.append("")
    lines.append(
        "This map is routing evidence for the Rustification queue. It does not "
        "promote Rust, LLVM/JIT, provider, hardware, GPU, performance, or "
        "isolated benchmark claims."
    )
    lines.append("")
    return "\n".join(lines)


def _default_architecture_layers(
    inventory_rows: Mapping[str, DifferentiableRustPythonInventoryRow],
) -> tuple[DifferentiableArchitectureMapLayer, ...]:
    return (
        _layer(
            "public_api_facade",
            "Public API facade and dashboard evidence",
            "Owns the stable Python package entry points and routes callers to bounded evidence.",
            ("unified_differentiable_api",),
            ("docs_api_maintainability", "adoption_licensing"),
            inventory_rows,
        ),
        _layer(
            "qnode_framework_bridges",
            "QNode framework bridge layer",
            "Routes local QNode evidence into JAX, PyTorch, PennyLane, and Qiskit boundaries.",
            ("pennylane_plugin_matrix", "qiskit_runtime_provider_gradients"),
            (
                "jax_native_transforms",
                "pytorch_autograd_compile",
                "pennylane_qnode_device_plugin",
                "qiskit_runtime_provider_gradients",
            ),
            inventory_rows,
        ),
        _layer(
            "program_ad_core",
            "Program AD core",
            "Owns whole-program frontend evidence and bounded Rust Program AD replay.",
            ("rust_program_ad_ir", "whole_program_frontend"),
            ("rust_native_program_ad",),
            inventory_rows,
        ),
        _layer(
            "compiler_ad_native_execution",
            "Compiler AD native execution",
            "Owns compiler-native MLIR, LLVM, Enzyme, and Rust kernel evidence boundaries.",
            (
                "rust_compiler_ad_primitives",
                "enzyme_mlir_compiler_ad",
                "catalyst_compiler_comparison",
            ),
            ("catalyst_compiler_workflows", "enzyme_compiler_ad"),
            inventory_rows,
        ),
        _layer(
            "provider_hardware_boundary",
            "Provider and hardware boundary",
            "Owns live-ticket, provider, QPU, raw-count, calibration, and hardware blockers.",
            ("qiskit_runtime_provider_gradients", "hardware_gradient_campaigns"),
            ("provider_hardware_gradients", "qiskit_runtime_provider_gradients"),
            inventory_rows,
        ),
        _layer(
            "benchmark_and_claim_governance",
            "Benchmark and claim governance",
            "Owns scorecard, claim wording, benchmark promotion, docs, and licensing gates.",
            ("differentiable_sota_scorecard",),
            ("benchmark_promotion", "docs_api_maintainability", "adoption_licensing"),
            inventory_rows,
        ),
    )


def _layer(
    layer_id: DifferentiableArchitectureLayerId,
    title: str,
    role: str,
    surface_ids: tuple[str, ...],
    categories: tuple[DifferentiableSOTACategory, ...],
    inventory_rows: Mapping[str, DifferentiableRustPythonInventoryRow],
) -> DifferentiableArchitectureMapLayer:
    rows = tuple(inventory_rows[surface_id] for surface_id in surface_ids)
    return DifferentiableArchitectureMapLayer(
        layer_id=layer_id,
        title=title,
        role=role,
        owner_modules=_unique_paths(row.owner_module for row in rows),
        inventory_surface_ids=surface_ids,
        sota_categories=categories,
        python_surfaces=_unique_paths(path for row in rows for path in row.python_surface),
        rust_surfaces=_unique_paths(path for row in rows for path in row.rust_surface),
        polyglot_surfaces=_unique_paths(path for row in rows for path in row.polyglot_surface),
        test_surfaces=_unique_paths(path for row in rows for path in row.test_surface),
        docs_surfaces=_unique_paths(
            (
                *(path for row in rows for path in row.docs_surface),
                "data/differentiable_phase_qnode/differentiable_architecture_map_20260627.md",
            )
        ),
        benchmark_surfaces=_unique_paths(path for row in rows for path in row.benchmark_surface),
        blockers=_unique_paths(blocker for row in rows for blocker in row.blockers),
        next_hardening_rounds=_unique_paths(
            round_name for row in rows for round_name in row.next_hardening_rounds
        ),
        claim_boundary=DIFFERENTIABLE_ARCHITECTURE_MAP_CLAIM_BOUNDARY,
    )


def _layer_paths(layer: DifferentiableArchitectureMapLayer) -> Iterable[str]:
    yield from layer.owner_modules
    yield from layer.python_surfaces
    yield from layer.rust_surfaces
    yield from layer.polyglot_surfaces
    yield from layer.test_surfaces
    yield from layer.docs_surfaces


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
    "DIFFERENTIABLE_ARCHITECTURE_MAP_ARTIFACT_ID",
    "DIFFERENTIABLE_ARCHITECTURE_MAP_CLAIM_BOUNDARY",
    "DIFFERENTIABLE_ARCHITECTURE_MAP_SCHEMA",
    "REQUIRED_ARCHITECTURE_LAYER_IDS",
    "DifferentiableArchitectureLayerId",
    "DifferentiableArchitectureMap",
    "DifferentiableArchitectureMapLayer",
    "DifferentiableArchitectureMapValidation",
    "render_differentiable_architecture_map_markdown",
    "run_differentiable_architecture_map",
    "validate_differentiable_architecture_map",
]
