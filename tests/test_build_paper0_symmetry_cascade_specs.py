# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 symmetry cascade builder tests
"""Tests for Paper 0 symmetry-cascade spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import load_symmetry_cascade_validation_spec
from scripts.build_paper0_symmetry_cascade_specs import (
    SOURCE_LEDGER_IDS,
    build_from_ledger,
    build_symmetry_cascade_specs,
    render_report,
    write_outputs,
)


def test_symmetry_cascade_builder_preserves_full_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R01582", "P0R01596"]
    assert bundle.summary["source_record_count"] == 15
    assert bundle.summary["consumed_source_record_count"] == 15
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 4
    assert bundle.summary["math_ids"] == []
    assert bundle.summary["image_ids"] == ["IMG0031"]
    assert bundle.summary["table_ids"] == []
    assert bundle.summary["next_source_boundary"] == "P0R01597"
    assert [spec.key for spec in bundle.specs] == [
        "symmetry_cascade.cascade_opening",
        "symmetry_cascade.three_breaks_architecture",
        "symmetry_cascade.psi_field_potential_stability",
        "symmetry_cascade.world_interface_summary",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_symmetry_cascade_builder_keeps_claims_and_boundaries() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "Source-Field begins as a state of perfect simple symmetry"
        in specs["symmetry_cascade.cascade_opening"].source_formulae
    )
    assert (
        "Break 1 selects physical laws via Prime Directive of Genesis"
        in specs["symmetry_cascade.three_breaks_architecture"].source_formulae
    )
    assert (
        "Break 2 condenses a universal field into localised selves through a Mexican-hat landscape"
        in specs["symmetry_cascade.three_breaks_architecture"].source_formulae
    )
    assert (
        "stable vacuum prevents runaway collapse and provides a foundation for reality"
        in specs["symmetry_cascade.psi_field_potential_stability"].source_formulae
    )
    assert (
        "Psi-field does not interact through a new undiscovered direct force"
        in specs["symmetry_cascade.world_interface_summary"].source_formulae
    )
    assert (
        "geometric interface subtly influences spacetime geometry and gravity equations"
        in specs["symmetry_cascade.world_interface_summary"].source_formulae
    )
    assert (
        "informational interface biases quantum probabilities and stabilises delicate quantum states"
        in specs["symmetry_cascade.world_interface_summary"].source_formulae
    )


def test_symmetry_cascade_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R01591":
            continue
        records.append(
            {
                "ledger_id": record_id,
                "source_record_id": f"{record_id}:stub",
                "source_block_index": int(record_id[3:]),
                "canonical_category": "mechanism",
                "block_type": "Para",
                "math_ids": [],
                "image_ids": [],
                "table_id": None,
                "section_path": "Paper 0 > Symmetry Cascade",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_symmetry_cascade_specs(records)


def test_symmetry_cascade_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_symmetry_cascade_validation_spec(
        "symmetry_cascade.world_interface_summary",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Symmetry Cascade Specs" in report
    assert loaded["key"] == "symmetry_cascade.world_interface_summary"
    assert "new undiscovered direct force" in loaded["source_formulae"][0]
    assert "Symmetry Cascade" in render_report(bundle)
