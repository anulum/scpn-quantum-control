# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable architecture map tests
"""Tests for differentiable architecture and Rustification map governance."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, TypeVar, cast

import pytest

import scpn_quantum_control as scpn
from scpn_quantum_control import (
    DifferentiableArchitectureMapLayer,
    differentiable_api,
    differentiable_module_hardening_registry,
    render_differentiable_architecture_map_markdown,
    run_differentiable_architecture_map,
    validate_differentiable_architecture_map,
)
from scpn_quantum_control.differentiable_architecture_map import (
    DIFFERENTIABLE_ARCHITECTURE_MAP_ARTIFACT_ID,
    DIFFERENTIABLE_ARCHITECTURE_MAP_CLAIM_BOUNDARY,
    DIFFERENTIABLE_ARCHITECTURE_MAP_SCHEMA,
)
from scpn_quantum_control.differentiable_baseline_scorecard import (
    run_differentiable_baseline_scorecard,
)
from scpn_quantum_control.differentiable_claim_ledger import REPO_ROOT
from scpn_quantum_control.differentiable_rust_python_inventory import (
    run_differentiable_rust_python_inventory,
)

_T = TypeVar("_T")


def _replace_unchecked(instance: _T, /, **changes: object) -> _T:
    """Construct deliberately malformed frozen records for validator tests."""
    return cast(_T, replace(cast(Any, instance), **changes))


def test_architecture_map_records_required_layers_and_boundaries() -> None:
    """The map must connect architecture layers to inventory and baseline evidence."""
    architecture_map = run_differentiable_architecture_map()

    assert architecture_map.schema == DIFFERENTIABLE_ARCHITECTURE_MAP_SCHEMA
    assert architecture_map.artifact_id == DIFFERENTIABLE_ARCHITECTURE_MAP_ARTIFACT_ID
    assert architecture_map.claim_boundary == DIFFERENTIABLE_ARCHITECTURE_MAP_CLAIM_BOUNDARY
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


def test_architecture_layer_constructor_rejects_invalid_identity_and_blockers() -> None:
    """Layer records must reject unknown identities and malformed blocker text."""
    layer = run_differentiable_architecture_map().layers[0]

    with pytest.raises(ValueError, match="unknown architecture layer_id"):
        replace(layer, layer_id="unknown_layer")
    for field_name in ("layer_id", "title", "role", "claim_boundary"):
        with pytest.raises(ValueError, match=f"{field_name} must be non-empty"):
            _replace_unchecked(layer, **{field_name: " "})
    with pytest.raises(ValueError, match="title must be non-empty"):
        _replace_unchecked(layer, title=1)
    with pytest.raises(ValueError, match="blockers must contain only non-empty entries"):
        replace(layer, blockers=(" ",))
    with pytest.raises(ValueError, match="blockers must contain only non-empty entries"):
        _replace_unchecked(layer, blockers=["declared blocker"])
    with pytest.raises(ValueError, match="blockers must contain only non-empty entries"):
        _replace_unchecked(layer, blockers=(1,))


def test_architecture_layer_constructor_rejects_bad_sequence_entries() -> None:
    """Every required layer sequence must be populated, non-blank, and unique."""
    layer = run_differentiable_architecture_map().layers[0]
    sequence_fields = (
        "owner_modules",
        "inventory_surface_ids",
        "baseline_categories",
        "python_surfaces",
        "rust_surfaces",
        "polyglot_surfaces",
        "test_surfaces",
        "docs_surfaces",
        "benchmark_surfaces",
        "next_hardening_rounds",
    )

    for field_name in sequence_fields:
        with pytest.raises(ValueError, match=f"{field_name} must contain non-empty entries"):
            _replace_unchecked(layer, **{field_name: ()})
        with pytest.raises(ValueError, match=f"{field_name} must contain non-empty entries"):
            _replace_unchecked(layer, **{field_name: (" ",)})
        values = cast(tuple[str, ...], getattr(layer, field_name))
        with pytest.raises(ValueError, match=f"{field_name} must not contain duplicate entries"):
            _replace_unchecked(layer, **{field_name: (values[0], values[0])})

    with pytest.raises(ValueError, match="owner_modules must contain non-empty entries"):
        _replace_unchecked(layer, owner_modules=["src/scpn_quantum_control/differentiable.py"])
    with pytest.raises(ValueError, match="owner_modules must contain non-empty entries"):
        _replace_unchecked(layer, owner_modules=(1,))

    blocker = layer.blockers[0]
    with pytest.raises(ValueError, match="blockers must not contain duplicate entries"):
        replace(layer, blockers=(blocker, blocker))


def test_architecture_map_constructor_rejects_structurally_empty_records() -> None:
    """Aggregate records must carry identity, layers, and non-negative counts."""
    architecture_map = run_differentiable_architecture_map()

    for field_name in ("schema", "artifact_id", "claim_boundary"):
        with pytest.raises(ValueError, match=f"{field_name} must be non-empty"):
            _replace_unchecked(architecture_map, **{field_name: " "})
    with pytest.raises(ValueError, match="schema must be non-empty"):
        _replace_unchecked(architecture_map, schema=1)
    with pytest.raises(ValueError, match="layers must be a non-empty tuple"):
        replace(architecture_map, layers=())
    with pytest.raises(ValueError, match="layers must be a non-empty tuple"):
        _replace_unchecked(architecture_map, layers=list(architecture_map.layers))
    with pytest.raises(ValueError, match="layers must be a non-empty tuple"):
        _replace_unchecked(architecture_map, layers=(object(),))
    with pytest.raises(ValueError, match="rustification_ready must be a bool"):
        _replace_unchecked(architecture_map, rustification_ready=1)
    with pytest.raises(ValueError, match="counts must be non-negative"):
        _replace_unchecked(architecture_map, ready_layer_count=True)
    with pytest.raises(ValueError, match="counts must be non-negative"):
        _replace_unchecked(architecture_map, total_layer_count="6")
    with pytest.raises(ValueError, match="counts must be non-negative"):
        replace(architecture_map, ready_layer_count=-1)
    with pytest.raises(ValueError, match="counts must be non-negative"):
        replace(architecture_map, total_layer_count=-1)


def test_architecture_map_validation_rejects_identity_counts_and_readiness() -> None:
    """Validation must reject aggregate identity, count, and readiness drift."""
    architecture_map = run_differentiable_architecture_map()
    invalid = replace(
        architecture_map,
        schema="unexpected-schema",
        artifact_id="unexpected-artifact",
        rustification_ready=True,
        ready_layer_count=99,
        total_layer_count=99,
        claim_boundary="test-only boundary",
    )

    validation = validate_differentiable_architecture_map(invalid)

    assert not validation.passed
    assert any("unexpected architecture-map schema" in error for error in validation.errors)
    assert any("unexpected architecture-map artifact_id" in error for error in validation.errors)
    assert any("architecture-map claim_boundary" in error for error in validation.errors)
    assert "total_layer_count does not match layer count" in validation.errors
    assert "ready_layer_count does not match ready layers" in validation.errors
    assert (
        "rustification_ready does not match inventory and scorecard readiness" in validation.errors
    )
    assert any(
        "ready architecture layers must not carry blockers" in error for error in validation.errors
    )


def test_architecture_map_validation_rejects_unknown_references_and_unsafe_paths() -> None:
    """Validation must fail closed on stale references, routing, and escaped paths."""
    architecture_map = run_differentiable_architecture_map()
    invalid_layer = _replace_unchecked(
        architecture_map.layers[2],
        title="Drifted layer title",
        owner_modules=(
            "src/scpn_quantum_control/missing.py",
            "/etc/passwd",
            "../outside.py",
        ),
        inventory_surface_ids=("missing_surface",),
        baseline_categories=("missing_category",),
        claim_boundary="test-only invalid layer",
    )
    invalid = replace(
        architecture_map,
        layers=(
            *architecture_map.layers[:2],
            invalid_layer,
            *architecture_map.layers[3:],
        ),
    )

    validation = validate_differentiable_architecture_map(invalid)

    assert not validation.passed
    assert any(
        "unknown inventory surface: missing_surface" in error for error in validation.errors
    )
    assert any(
        "unknown baseline category: missing_category" in error for error in validation.errors
    )
    assert any("evidence path does not exist" in error for error in validation.errors)
    assert (
        sum("evidence path must be repository-relative" in error for error in validation.errors)
        == 2
    )
    assert any(
        "unmapped inventory surface: rust_program_ad_ir" in error for error in validation.errors
    )
    assert any(
        "unmapped baseline category: rust_native_program_ad" in error
        for error in validation.errors
    )
    assert any("claim_boundary does not match" in error for error in validation.errors)
    assert any(
        "title does not match inventory-derived routing" in error for error in validation.errors
    )
    assert any(
        "owner_modules does not match inventory-derived routing" in error
        for error in validation.errors
    )


def test_architecture_map_validation_rejects_duplicate_layer_ids() -> None:
    """The six canonical layer identities must remain ordered and unique."""
    architecture_map = run_differentiable_architecture_map()
    duplicate_layers = (
        architecture_map.layers[0],
        architecture_map.layers[0],
        *architecture_map.layers[2:],
    )
    invalid = replace(architecture_map, layers=duplicate_layers)

    validation = validate_differentiable_architecture_map(invalid)

    assert not validation.passed
    assert "architecture layer IDs must match REQUIRED_ARCHITECTURE_LAYER_IDS exactly" in (
        validation.errors
    )
    assert "duplicate architecture layer_id: public_api_facade" in validation.errors


def test_architecture_map_injected_dependencies_validate_and_fail_closed() -> None:
    """Injected inventory and scorecard inputs must follow the same strict contracts."""
    inventory = run_differentiable_rust_python_inventory()
    scorecard = run_differentiable_baseline_scorecard()
    architecture_map = run_differentiable_architecture_map(
        inventory=inventory,
        scorecard=scorecard,
    )

    validation = validate_differentiable_architecture_map(
        architecture_map,
        inventory=inventory,
        scorecard=scorecard,
    )
    assert validation.passed, validation.errors
    assert validation.to_dict()["passed"] is True

    invalid_upstreams = validate_differentiable_architecture_map(
        architecture_map,
        inventory=replace(inventory, schema="invalid-inventory-schema"),
        scorecard=replace(scorecard, schema="invalid-scorecard-schema"),
    )
    assert any("inventory validation failed" in error for error in invalid_upstreams.errors)
    assert any("scorecard validation failed" in error for error in invalid_upstreams.errors)

    incomplete_inventory = replace(
        inventory,
        rows=tuple(
            row for row in inventory.rows if row.surface_id != "unified_differentiable_api"
        ),
    )
    incomplete_validation = validate_differentiable_architecture_map(
        architecture_map,
        inventory=incomplete_inventory,
        scorecard=scorecard,
    )
    assert any(
        "unknown inventory surface: unified_differentiable_api" in error
        for error in incomplete_validation.errors
    )
    assert any(
        "cannot derive canonical architecture routing" in error
        for error in incomplete_validation.errors
    )
    with pytest.raises(ValueError, match="missing required inventory surface"):
        run_differentiable_architecture_map(
            inventory=incomplete_inventory,
            scorecard=scorecard,
        )


def test_architecture_map_committed_artifacts_match_public_serializers() -> None:
    """Committed JSON and Markdown must be byte-identical to public renderers."""
    architecture_map = run_differentiable_architecture_map()
    json_path = (
        REPO_ROOT / "data/differentiable_phase_qnode/differentiable_architecture_map_20260627.json"
    )
    markdown_path = json_path.with_suffix(".md")

    expected_json = json.dumps(architecture_map.to_dict(), indent=2, sort_keys=True) + "\n"
    expected_markdown = render_differentiable_architecture_map_markdown(architecture_map) + "\n"

    assert json_path.read_text(encoding="utf-8") == expected_json
    assert markdown_path.read_text(encoding="utf-8") == expected_markdown


def test_architecture_map_markdown_escapes_table_content() -> None:
    """Reviewer Markdown must keep blocker text inside its table cell."""
    architecture_map = run_differentiable_architecture_map()
    escaped_layer = replace(
        architecture_map.layers[0],
        blockers=("first line | bounded\nsecond line",),
    )
    candidate = replace(
        architecture_map,
        layers=(escaped_layer, *architecture_map.layers[1:]),
    )

    markdown = render_differentiable_architecture_map_markdown(candidate)

    assert "first line \\| bounded second line" in markdown


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
    assert scpn.DifferentiableArchitectureMapLayer is DifferentiableArchitectureMapLayer
    registry_paths = {record.module_path for record in differentiable_module_hardening_registry()}
    assert "src/scpn_quantum_control/differentiable_architecture_map.py" in registry_paths
