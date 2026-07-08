# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable architecture map tests
"""Tests for differentiable architecture and Rustification map governance."""

from __future__ import annotations

from pathlib import Path

import scpn_quantum_control as scpn
from scpn_quantum_control import (
    DifferentiableArchitectureMapLayer,
    differentiable_api,
    differentiable_module_hardening_registry,
    render_differentiable_architecture_map_markdown,
    run_differentiable_architecture_map,
    validate_differentiable_architecture_map,
)


def test_architecture_map_records_required_layers_and_boundaries() -> None:
    """The map must connect architecture layers to inventory and baseline evidence."""
    architecture_map = run_differentiable_architecture_map()

    assert architecture_map.schema == "scpn_qc_differentiable_architecture_map_v1"
    assert architecture_map.rustification_ready is False
    assert architecture_map.total_layer_count == len(architecture_map.layers)
    assert architecture_map.ready_layer_count == 0
    assert {layer.layer_id for layer in architecture_map.layers} == {
        "public_api_facade",
        "qnode_framework_bridges",
        "program_ad_core",
        "compiler_ad_native_execution",
        "provider_hardware_boundary",
        "benchmark_and_claim_governance",
    }
    assert "no broad Rustification promotion" in architecture_map.claim_boundary


def test_architecture_map_layers_are_path_backed_and_claim_bounded() -> None:
    """Each layer must cite real modules, tests, docs, benchmarks, and blockers."""
    architecture_map = run_differentiable_architecture_map()
    layers = {layer.layer_id: layer for layer in architecture_map.layers}

    program_ad = layers["program_ad_core"]
    assert program_ad.inventory_surface_ids == (
        "rust_program_ad_ir",
        "whole_program_frontend",
    )
    assert program_ad.baseline_categories == ("rust_native_program_ad",)
    assert "src/scpn_quantum_control/differentiable.py" in program_ad.python_surfaces
    assert "scpn_quantum_engine/program_ad_replay/src/program_ad_ir.rs" in program_ad.rust_surfaces
    assert "tests/test_program_ad_rust_bridge.py" in program_ad.test_surfaces
    assert any("array adjoints" in blocker for blocker in program_ad.blockers)

    governance = layers["benchmark_and_claim_governance"]
    assert governance.inventory_surface_ids == ("differentiable_baseline_scorecard",)
    assert governance.baseline_categories == (
        "benchmark_promotion",
        "docs_api_maintainability",
        "adoption_licensing",
    )
    assert "data/differentiable_phase_qnode/differentiable_baseline_scorecard_20260620.md" in (
        governance.docs_surfaces
    )

    validation = validate_differentiable_architecture_map(architecture_map)
    assert validation.passed, validation.errors
    assert "data/differentiable_phase_qnode/differentiable_architecture_map_20260627.md" in (
        validation.checked_paths
    )


def test_architecture_map_validation_rejects_unknown_references_and_paths(
    tmp_path: Path,
) -> None:
    """Validation must fail closed on stale architecture routing evidence."""
    invalid_layer = DifferentiableArchitectureMapLayer(
        layer_id="invalid_layer",
        title="Invalid layer",
        role="Invalid routing evidence.",
        owner_modules=("src/scpn_quantum_control/missing.py",),
        inventory_surface_ids=("missing_surface",),
        baseline_categories=("rust_native_program_ad",),
        python_surfaces=("src/scpn_quantum_control/missing.py",),
        rust_surfaces=("scpn_quantum_engine/src/missing.rs",),
        polyglot_surfaces=("docs/missing.md",),
        test_surfaces=("tests/test_missing.py",),
        docs_surfaces=("docs/missing.md",),
        benchmark_surfaces=("missing-benchmark",),
        blockers=("unexpected ready blocker",),
        next_hardening_rounds=("Round 5 Rustification readiness",),
        claim_boundary="test-only invalid layer",
    )
    architecture_map = type(run_differentiable_architecture_map())(
        schema="scpn_qc_differentiable_architecture_map_v1",
        artifact_id="test-architecture-map",
        layers=(invalid_layer,),
        rustification_ready=True,
        ready_layer_count=1,
        total_layer_count=1,
        claim_boundary="test-only invalid architecture map",
    )

    validation = validate_differentiable_architecture_map(
        architecture_map,
        repo_root=tmp_path,
    )

    assert not validation.passed
    assert any("unknown inventory surface" in error for error in validation.errors)
    assert any("evidence path does not exist" in error for error in validation.errors)
    assert any(
        "ready architecture layers must not carry blockers" in error for error in validation.errors
    )


def test_architecture_map_markdown_unified_api_and_exports() -> None:
    """The map must render, dispatch, and export through public package surfaces."""
    architecture_map = run_differentiable_architecture_map()
    markdown = render_differentiable_architecture_map_markdown(architecture_map)
    result = differentiable_api("architecture_rustification_map")

    assert "# Differentiable Architecture and Rustification Map" in markdown
    assert "program_ad_core" in markdown
    assert result.operation == "architecture_rustification_map"
    assert result.supported is False
    assert result.payload["total_layer_count"] == architecture_map.total_layer_count
    assert "run_differentiable_architecture_map" in scpn.__all__
    assert scpn.run_differentiable_architecture_map is run_differentiable_architecture_map
    registry_paths = {record.module_path for record in differentiable_module_hardening_registry()}
    assert "src/scpn_quantum_control/differentiable_architecture_map.py" in registry_paths
