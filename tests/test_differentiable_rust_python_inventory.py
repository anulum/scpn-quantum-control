# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable Rust/Python inventory tests
"""Tests for differentiable Rust/Python surface inventory governance."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast, get_args

import pytest

import scpn_quantum_control as scpn
from scpn_quantum_control.differentiable_api import differentiable_api
from scpn_quantum_control.differentiable_claim_ledger import (
    load_differentiable_claim_ledger,
)
from scpn_quantum_control.differentiable_module_hardening_audit import (
    differentiable_module_hardening_registry,
)
from scpn_quantum_control.differentiable_rust_python_inventory import (
    REQUIRED_INVENTORY_CLASSIFICATIONS,
    DifferentiableRustPythonInventory,
    DifferentiableRustPythonInventoryBenchmarkStatus,
    DifferentiableRustPythonInventoryClassification,
    DifferentiableRustPythonInventoryPolyglotStatus,
    DifferentiableRustPythonInventoryRow,
    DifferentiableRustPythonInventoryRustParityStatus,
    render_differentiable_rust_python_inventory_markdown,
    run_differentiable_rust_python_inventory,
    validate_differentiable_rust_python_inventory,
)
from tools import differentiable_rust_python_inventory_quality_gates as inventory_quality_gates


def test_inventory_quality_gate_spec_is_exact_and_focused() -> None:
    """The owner gate must mirror strict static and exact branch checks."""
    static_gates = dict(inventory_quality_gates.build_static_quality_gates("python"))
    cohort = inventory_quality_gates.DIFFERENTIABLE_RUST_PYTHON_INVENTORY_QUALITY_RATCHET

    assert (
        static_gates["mypy-strict-differentiable-rust-python-inventory-quality"][-len(cohort) :]
        == cohort
    )
    assert (
        static_gates["ruff D differentiable-rust-python-inventory quality ratchet"][-len(cohort) :]
        == cohort
    )

    coverage_gates = inventory_quality_gates.build_coverage_gates("python")
    assert "--branch" in coverage_gates[0][1]
    assert (
        coverage_gates[0][1][-1:]
        == inventory_quality_gates.DIFFERENTIABLE_RUST_PYTHON_INVENTORY_COVERAGE_COHORT
    )
    assert "--fail-under=100" in coverage_gates[1][1]
    assert "--include=*/differentiable_rust_python_inventory.py" in coverage_gates[1][1]


def _row_record(
    *,
    surface_id: str = "test_surface",
    title: str = "Test surface",
    classification: DifferentiableRustPythonInventoryClassification = "python_reference",
    owner_module: str = "src/scpn_quantum_control/differentiable_rust_python_inventory.py",
    public_api: tuple[str, ...] = ("run_differentiable_rust_python_inventory",),
    python_surface: tuple[str, ...] = (
        "src/scpn_quantum_control/differentiable_rust_python_inventory.py",
    ),
    rust_surface: tuple[str, ...] = ("scpn_quantum_engine/src/lib.rs",),
    polyglot_surface: tuple[str, ...] = ("docs/differentiable_api.md",),
    test_surface: tuple[str, ...] = ("tests/test_differentiable_rust_python_inventory.py",),
    docs_surface: tuple[str, ...] = ("docs/differentiable_programming.md",),
    benchmark_surface: tuple[str, ...] = ("test-benchmark",),
    claim_ids: tuple[str, ...] = ("differentiable_rust_python_inventory",),
    mypy_target: str = "src/scpn_quantum_control/differentiable_rust_python_inventory.py",
    docstring_status: str = "complete",
    benchmark_status: DifferentiableRustPythonInventoryBenchmarkStatus = "not_applicable",
    rust_parity_status: DifferentiableRustPythonInventoryRustParityStatus = "not_applicable",
    polyglot_status: DifferentiableRustPythonInventoryPolyglotStatus = "not_applicable",
    blockers: tuple[str, ...] = (),
    next_hardening_rounds: tuple[str, ...] = ("Round 5 Rustification readiness",),
    claim_boundary: str = "test-only inventory boundary",
) -> DifferentiableRustPythonInventoryRow:
    """Construct one row through its typed public record contract."""
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
        mypy_target=mypy_target,
        docstring_status=docstring_status,
        benchmark_status=benchmark_status,
        rust_parity_status=rust_parity_status,
        polyglot_status=polyglot_status,
        blockers=blockers,
        next_hardening_rounds=next_hardening_rounds,
        claim_boundary=claim_boundary,
    )


def _inventory_with_row(
    inventory: DifferentiableRustPythonInventory,
    row: DifferentiableRustPythonInventoryRow,
) -> DifferentiableRustPythonInventory:
    """Return an internally consistent one-row inventory."""
    classification_counts: dict[str, int] = {
        classification: int(classification == row.classification)
        for classification in REQUIRED_INVENTORY_CLASSIFICATIONS
    }
    return replace(
        inventory,
        rows=(row,),
        rustification_ready=row.rustification_ready,
        ready_surface_count=int(row.rustification_ready),
        total_surface_count=1,
        classification_counts=classification_counts,
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
        "differentiable_baseline_scorecard",
    }
    assert "no broad rustification promotion" in inventory.claim_boundary


def test_inventory_rows_are_claim_bounded_and_path_backed() -> None:
    """Rows must include owner, tests, docs, benchmark, Rust, and polyglot state."""
    inventory = run_differentiable_rust_python_inventory()
    rows = {row.surface_id: row for row in inventory.rows}

    rust_ir = rows["rust_program_ad_ir"]
    assert rust_ir.classification == "rust_backed"
    assert rust_ir.rust_parity_status == "partial"
    assert "scpn_quantum_engine/program_ad_replay/src/program_ad_ir.rs" in rust_ir.rust_surface
    assert (
        "scpn_quantum_engine/program_ad_replay/src/program_ad_signal_reduction.rs"
        in rust_ir.rust_surface
    )
    assert (
        "scpn_quantum_engine/program_ad_replay/src/program_ad_cumulative_reduction.rs"
        in rust_ir.rust_surface
    )
    assert (
        "scpn_quantum_engine/program_ad_replay/src/program_ad_stencil_reduction.rs"
        in rust_ir.rust_surface
    )
    assert "scpn_quantum_engine/fuzz/fuzz_targets/program_ad_ir.rs" in rust_ir.rust_surface
    assert "tests/test_phase_qnode_rust_parity.py" in rust_ir.test_surface
    assert "tests/test_program_ad_rust_signal_bridge.py" in rust_ir.test_surface
    assert "tests/test_program_ad_rust_cumulative_bridge.py" in rust_ir.test_surface
    assert "scpn_quantum_engine/tests/program_ad_signal.rs" in rust_ir.test_surface
    assert "scpn_quantum_engine/tests/program_ad_cumulative.rs" in rust_ir.test_surface
    assert "scpn_quantum_engine/tests/program_ad_stencil.rs" in rust_ir.test_surface
    assert "scpn_quantum_engine/tests/program_ad_panic_boundary.rs" in rust_ir.test_surface
    assert "scpn_quantum_engine/fuzz/fuzz_targets/program_ad_ir.rs" in rust_ir.test_surface
    assert any("array adjoints" in blocker for blocker in rust_ir.blockers)

    qiskit = rows["qiskit_runtime_provider_gradients"]
    assert qiskit.classification == "provider_blocked"
    assert qiskit.rust_parity_status == "not_applicable"
    assert any("live-ticket" in blocker for blocker in qiskit.blockers)
    assert qiskit.benchmark_status == "blocked"

    validation = validate_differentiable_rust_python_inventory(inventory)
    assert validation.passed, validation.errors
    assert "src/scpn_quantum_control/differentiable_api.py" in validation.checked_paths
    assert "scpn_quantum_engine/program_ad_replay/src/program_ad_ir.rs" in validation.checked_paths


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


def test_committed_inventory_artifacts_match_public_serializers() -> None:
    """Committed JSON and Markdown must match the public inventory routes."""
    inventory = run_differentiable_rust_python_inventory()
    data_root = Path(__file__).resolve().parents[1] / "data/differentiable_phase_qnode"
    json_path = data_root / "differentiable_rust_python_inventory_20260620.json"
    markdown_path = json_path.with_suffix(".md")

    expected_json = json.dumps(inventory.to_dict(), indent=2, sort_keys=True) + "\n"
    expected_markdown = render_differentiable_rust_python_inventory_markdown(inventory) + "\n"

    assert json_path.read_text(encoding="utf-8") == expected_json
    assert markdown_path.read_text(encoding="utf-8") == expected_markdown


def test_inventory_is_exported_and_registered_for_hardening() -> None:
    """Top-level exports and hardening registry must include the inventory."""
    registry_paths = {record.module_path for record in differentiable_module_hardening_registry()}

    assert scpn.run_differentiable_rust_python_inventory is (
        run_differentiable_rust_python_inventory
    )
    assert "run_differentiable_rust_python_inventory" in scpn.__all__
    assert "rust_backed" in get_args(DifferentiableRustPythonInventoryClassification)
    assert "src/scpn_quantum_control/differentiable_rust_python_inventory.py" in registry_paths


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        pytest.param(
            lambda: _row_record(
                classification=cast(
                    DifferentiableRustPythonInventoryClassification,
                    "unknown",
                )
            ),
            "unknown inventory classification: unknown",
            id="classification",
        ),
        pytest.param(
            lambda: _row_record(
                benchmark_status=cast(
                    DifferentiableRustPythonInventoryBenchmarkStatus,
                    "unknown",
                )
            ),
            "unknown inventory benchmark status: unknown",
            id="benchmark-status",
        ),
        pytest.param(
            lambda: _row_record(
                rust_parity_status=cast(
                    DifferentiableRustPythonInventoryRustParityStatus,
                    "unknown",
                )
            ),
            "unknown inventory Rust parity status: unknown",
            id="rust-parity-status",
        ),
        pytest.param(
            lambda: _row_record(
                polyglot_status=cast(
                    DifferentiableRustPythonInventoryPolyglotStatus,
                    "unknown",
                )
            ),
            "unknown inventory polyglot status: unknown",
            id="polyglot-status",
        ),
        pytest.param(
            lambda: _row_record(surface_id=" "),
            "surface_id must be non-empty",
            id="surface-id",
        ),
        pytest.param(
            lambda: _row_record(title=""),
            "title must be non-empty",
            id="title",
        ),
        pytest.param(
            lambda: _row_record(owner_module="\t"),
            "owner_module must be non-empty",
            id="owner-module",
        ),
        pytest.param(
            lambda: _row_record(mypy_target=""),
            "mypy_target must be non-empty",
            id="mypy-target",
        ),
        pytest.param(
            lambda: _row_record(docstring_status=" "),
            "docstring_status must be non-empty",
            id="docstring-status",
        ),
        pytest.param(
            lambda: _row_record(claim_boundary=""),
            "claim_boundary must be non-empty",
            id="claim-boundary",
        ),
        pytest.param(
            lambda: _row_record(public_api=()),
            "public_api must contain non-empty entries",
            id="public-api",
        ),
        pytest.param(
            lambda: _row_record(python_surface=("",)),
            "python_surface must contain non-empty entries",
            id="python-surface",
        ),
        pytest.param(
            lambda: _row_record(rust_surface=()),
            "rust_surface must contain non-empty entries",
            id="rust-surface",
        ),
        pytest.param(
            lambda: _row_record(polyglot_surface=(" ",)),
            "polyglot_surface must contain non-empty entries",
            id="polyglot-surface",
        ),
        pytest.param(
            lambda: _row_record(test_surface=()),
            "test_surface must contain non-empty entries",
            id="test-surface",
        ),
        pytest.param(
            lambda: _row_record(docs_surface=("",)),
            "docs_surface must contain non-empty entries",
            id="docs-surface",
        ),
        pytest.param(
            lambda: _row_record(benchmark_surface=()),
            "benchmark_surface must contain non-empty entries",
            id="benchmark-surface",
        ),
        pytest.param(
            lambda: _row_record(claim_ids=(" ",)),
            "claim_ids must contain non-empty entries",
            id="claim-ids",
        ),
        pytest.param(
            lambda: _row_record(next_hardening_rounds=()),
            "next_hardening_rounds must contain non-empty entries",
            id="hardening-rounds",
        ),
        pytest.param(
            lambda: _row_record(blockers=("",)),
            "blockers must contain only non-empty entries",
            id="blank-blocker",
        ),
    ],
)
def test_inventory_row_rejects_invalid_public_fields(
    factory: Callable[[], DifferentiableRustPythonInventoryRow],
    message: str,
) -> None:
    """The public row record must reject each malformed field family."""
    with pytest.raises(ValueError, match=message):
        factory()


def test_inventory_validation_serializes_structural_errors() -> None:
    """Validation must expose stale schema, counts, and classification totals."""
    ledger = load_differentiable_claim_ledger()
    inventory = run_differentiable_rust_python_inventory(ledger=ledger)
    classification_counts = dict(inventory.classification_counts)
    classification_counts["python_reference"] += 1
    invalid_inventory = replace(
        inventory,
        schema="stale-rust-python-inventory",
        total_surface_count=inventory.total_surface_count + 1,
        classification_counts=classification_counts,
    )

    validation = validate_differentiable_rust_python_inventory(
        invalid_inventory,
        ledger=ledger,
    )
    payload = validation.to_dict()

    assert not validation.passed
    assert "unexpected inventory schema: stale-rust-python-inventory" in validation.errors
    assert "total_surface_count does not match row count" in validation.errors
    assert "classification_counts does not match rows for python_reference" in validation.errors
    assert payload["passed"] is False
    assert payload["errors"] == list(validation.errors)
    assert payload["checked_surface_ids"] == list(validation.checked_surface_ids)
    assert payload["checked_claim_ids"] == list(validation.checked_claim_ids)
    assert payload["checked_paths"] == list(validation.checked_paths)


def test_inventory_validation_rejects_duplicate_surface_ids() -> None:
    """Validation must report duplicate IDs without collapsing row evidence."""
    ledger = load_differentiable_claim_ledger()
    inventory = run_differentiable_rust_python_inventory(ledger=ledger)
    rows = inventory.rows
    duplicate = replace(rows[1], surface_id=rows[0].surface_id)
    invalid_inventory = replace(
        inventory,
        rows=(rows[0], duplicate, *rows[2:]),
    )

    validation = validate_differentiable_rust_python_inventory(
        invalid_inventory,
        ledger=ledger,
    )

    assert not validation.passed
    assert "duplicate inventory surface_id: unified_differentiable_api" in validation.errors


def test_inventory_validation_requires_benchmark_classification_for_ready_candidate() -> None:
    """A parity-complete Rust candidate must not retain ``not_run`` evidence."""
    ledger = load_differentiable_claim_ledger()
    inventory = run_differentiable_rust_python_inventory(ledger=ledger)
    row = replace(
        inventory.rows[0],
        classification="rust_backed",
        benchmark_status="not_run",
        rust_parity_status="complete",
        polyglot_status="complete",
        blockers=(),
    )
    invalid_inventory = _inventory_with_row(inventory, row)

    validation = validate_differentiable_rust_python_inventory(
        invalid_inventory,
        ledger=ledger,
    )

    assert not validation.passed
    assert (
        "unified_differentiable_api: ready rows require benchmark classification evidence"
        in validation.errors
    )


def test_inventory_validation_requires_blockers_for_blocked_provider_row() -> None:
    """Blocked provider rows must explain why execution cannot be promoted."""
    ledger = load_differentiable_claim_ledger()
    inventory = run_differentiable_rust_python_inventory(ledger=ledger)
    qiskit = next(
        row for row in inventory.rows if row.surface_id == "qiskit_runtime_provider_gradients"
    )
    invalid_inventory = _inventory_with_row(inventory, replace(qiskit, blockers=()))

    validation = validate_differentiable_rust_python_inventory(
        invalid_inventory,
        ledger=ledger,
    )

    assert not validation.passed
    assert (
        "qiskit_runtime_provider_gradients: blocked provider/hardware rows must list blockers"
        in validation.errors
    )


def test_inventory_markdown_escapes_blocker_cells() -> None:
    """The renderer must neutralise newlines and Markdown table separators."""
    inventory = run_differentiable_rust_python_inventory()
    row = replace(inventory.rows[0], blockers=("line\nbreak|pipe",))

    markdown = render_differentiable_rust_python_inventory_markdown(
        _inventory_with_row(inventory, row)
    )

    assert "line break\\|pipe" in markdown
