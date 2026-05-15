# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom I Psi-field spec builder tests
"""Tests for Paper 0 Axiom I Psi-field spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import load_axiom_i_psi_field_validation_spec
from scripts.build_paper0_axiom_i_psi_field_specs import (
    SOURCE_LEDGER_IDS,
    build_axiom_i_psi_field_specs,
    build_from_ledger,
    render_report,
    write_outputs,
)


def test_axiom_i_psi_field_builder_preserves_full_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R00670", "P0R00702"]
    assert bundle.summary["source_record_count"] == 33
    assert bundle.summary["consumed_source_record_count"] == 33
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["blank_separator_count"] == 2
    assert bundle.summary["next_source_boundary"] == "P0R00703"
    assert [spec.key for spec in bundle.specs] == [
        "axiom_i_psi_field.axiom_i_source_boundary",
        "axiom_i_psi_field.psi_field_formalisation",
        "axiom_i_psi_field.predictive_coding_generative_model",
        "axiom_i_psi_field.hint_ontological_ground",
        "axiom_i_psi_field.multilayer_consciousness_definition",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_axiom_i_psi_field_builder_keeps_equation_and_mechanism_labels() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "P0R00680:psi_field_universal_complex_scalar"
        in specs["axiom_i_psi_field.psi_field_formalisation"].source_equation_ids
    )
    assert (
        "Psi-field as universal complex scalar field"
        in specs["axiom_i_psi_field.psi_field_formalisation"].source_formulae
    )
    assert (
        "cosmic generative model"
        in specs["axiom_i_psi_field.predictive_coding_generative_model"].source_formulae
    )
    assert (
        "H_int = -lambda * Psi_s * sigma"
        in specs["axiom_i_psi_field.hint_ontological_ground"].source_formulae
    )
    assert (
        "Psi_s as ontological ground"
        in specs["axiom_i_psi_field.hint_ontological_ground"].source_formulae
    )
    assert (
        "consciousness is not a single property"
        in specs["axiom_i_psi_field.multilayer_consciousness_definition"].source_formulae
    )


def test_axiom_i_psi_field_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R00692":
            continue
        records.append(
            {
                "ledger_id": record_id,
                "source_record_id": f"{record_id}:stub",
                "source_block_index": int(record_id[3:]),
                "section_path": "Paper 0 > Axiom I",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_axiom_i_psi_field_specs(records)


def test_axiom_i_psi_field_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_axiom_i_psi_field_validation_spec(
        "axiom_i_psi_field.predictive_coding_generative_model",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Axiom I Psi-Field Specs" in report
    assert "cosmic generative model" in report
    assert loaded["key"] == "axiom_i_psi_field.predictive_coding_generative_model"
    assert "cosmic generative model" in loaded["source_formulae"]
    assert "Axiom I Psi-Field" in render_report(bundle)
