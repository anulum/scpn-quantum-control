# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable Rust/Python inventory tests
"""Tests for differentiable Rust/Python surface inventory governance."""

from __future__ import annotations

from pathlib import Path
from typing import get_args

import scpn_quantum_control as scpn
from scpn_quantum_control.differentiable_api import differentiable_api
from scpn_quantum_control.differentiable_module_hardening_audit import (
    differentiable_module_hardening_registry,
)
from scpn_quantum_control.differentiable_rust_python_inventory import (
    REQUIRED_INVENTORY_CLASSIFICATIONS,
    DifferentiableRustPythonInventoryClassification,
    DifferentiableRustPythonInventoryRow,
    render_differentiable_rust_python_inventory_markdown,
    run_differentiable_rust_python_inventory,
    validate_differentiable_rust_python_inventory,
)


def test_inventory_records_required_rustification_classifications() -> None:
    """The rustification inventory must classify every differentiable surface."""
    inventory = run_differentiable_rust_python_inventory()

    assert inventory.schema == "scpn_qc_differentiable_rust_python_inventory_v1"
    assert inventory.rustification_ready is False
    assert inventory.total_surface_count == len(inventory.rows)
    assert inventory.ready_surface_count < inventory.total_surface_count
    assert set(inventory.classification_counts) >= set(REQUIRED_INVENTORY_CLASSIFICATIONS)
    assert {row.surface_id for row in inventory.rows} >= {
        "unified_differentiable_api",
        "rust_program_ad_ir",
        "rust_compiler_ad_primitives",
        "pennylane_plugin_matrix",
        "qiskit_runtime_provider_gradients",
        "differentiable_sota_scorecard",
    }
    assert "no broad rustification promotion" in inventory.claim_boundary


def test_inventory_rows_are_claim_bounded_and_path_backed() -> None:
    """Rows must include owner, tests, docs, benchmark, Rust, and polyglot state."""
    inventory = run_differentiable_rust_python_inventory()
    rows = {row.surface_id: row for row in inventory.rows}

    rust_ir = rows["rust_program_ad_ir"]
    assert rust_ir.classification == "rust_backed"
    assert rust_ir.rust_parity_status == "partial"
    assert "scpn_quantum_engine/src/program_ad_ir.rs" in rust_ir.rust_surface
    assert "tests/test_phase_qnode_rust_parity.py" in rust_ir.test_surface
    assert any("branch replay" in blocker for blocker in rust_ir.blockers)

    qiskit = rows["qiskit_runtime_provider_gradients"]
    assert qiskit.classification == "provider_blocked"
    assert qiskit.rust_parity_status == "not_applicable"
    assert any("live-ticket" in blocker for blocker in qiskit.blockers)
    assert qiskit.benchmark_status == "blocked"

    validation = validate_differentiable_rust_python_inventory(inventory)
    assert validation.passed, validation.errors
    assert "src/scpn_quantum_control/differentiable_api.py" in validation.checked_paths
    assert "scpn_quantum_engine/src/program_ad_ir.rs" in validation.checked_paths


def test_inventory_validation_rejects_missing_paths_and_ready_blockers(
    tmp_path: Path,
) -> None:
    """Validation must fail on stale paths and impossible promotion states."""
    row = DifferentiableRustPythonInventoryRow(
        surface_id="bad_ready_surface",
        title="Bad ready surface",
        classification="rust_backed",
        owner_module="src/scpn_quantum_control/missing.py",
        public_api=("missing_api",),
        python_surface=("src/scpn_quantum_control/missing.py",),
        rust_surface=("scpn_quantum_engine/src/missing.rs",),
        polyglot_surface=("src/scpn_quantum_control/missing.py",),
        test_surface=("tests/test_missing.py",),
        docs_surface=("docs/missing.md",),
        benchmark_surface=("missing-benchmark",),
        claim_ids=("missing_claim",),
        mypy_target="src/scpn_quantum_control/missing.py",
        docstring_status="complete",
        benchmark_status="isolated_required",
        rust_parity_status="complete",
        polyglot_status="complete",
        blockers=("should not be present",),
        next_hardening_rounds=("Round 5 Rustification readiness",),
        claim_boundary="test-only invalid row",
    )
    inventory = type(run_differentiable_rust_python_inventory())(
        schema="scpn_qc_differentiable_rust_python_inventory_v1",
        artifact_id="test-inventory",
        rows=(row,),
        rustification_ready=True,
        ready_surface_count=1,
        total_surface_count=1,
        classification_counts={"rust_backed": 1},
        claim_boundary="test-only invalid inventory",
    )

    validation = validate_differentiable_rust_python_inventory(inventory, repo_root=tmp_path)

    assert not validation.passed
    assert any("evidence path does not exist" in error for error in validation.errors)
    assert any(
        "ready Rust-backed rows must not carry blockers" in error for error in validation.errors
    )
    assert any("unknown claim-ledger row" in error for error in validation.errors)


def test_inventory_markdown_and_unified_api_dispatch() -> None:
    """The inventory must render and dispatch through the unified API."""
    inventory = run_differentiable_rust_python_inventory()
    markdown = render_differentiable_rust_python_inventory_markdown(inventory)
    result = differentiable_api("rust_python_inventory")

    assert "# Differentiable Rust/Python Surface Inventory" in markdown
    assert "rust_program_ad_ir" in markdown
    assert "provider_blocked" in markdown
    assert result.operation == "rust_python_inventory"
    assert result.supported is False
    assert result.payload["rustification_ready"] is False
    assert result.payload["total_surface_count"] == inventory.total_surface_count
    assert "no broad rustification promotion" in result.claim_boundary


def test_inventory_is_exported_and_registered_for_hardening() -> None:
    """Top-level exports and hardening registry must include the inventory."""
    registry_paths = {record.module_path for record in differentiable_module_hardening_registry()}

    assert scpn.run_differentiable_rust_python_inventory is (
        run_differentiable_rust_python_inventory
    )
    assert "run_differentiable_rust_python_inventory" in scpn.__all__
    assert "rust_backed" in get_args(DifferentiableRustPythonInventoryClassification)
    assert "src/scpn_quantum_control/differentiable_rust_python_inventory.py" in registry_paths
