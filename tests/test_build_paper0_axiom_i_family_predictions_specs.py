# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom I family predictions builder tests
"""Tests for Paper 0 Axiom I family-satisfaction and prediction spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_axiom_i_family_predictions_validation_spec,
)
from scripts.build_paper0_axiom_i_family_predictions_specs import (
    SOURCE_LEDGER_IDS,
    build_axiom_i_family_predictions_specs,
    build_from_ledger,
    render_report,
    write_outputs,
)


def test_family_predictions_builder_preserves_contiguous_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R00747", "P0R00756"]
    assert bundle.summary["source_record_count"] == 10
    assert bundle.summary["consumed_source_record_count"] == 10
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["conditional_prediction_count"] == 3
    assert bundle.summary["rejected_model_class_count"] == 5
    assert bundle.summary["blank_separator_count"] == 1
    assert bundle.summary["next_source_boundary"] == "P0R00757"
    assert [spec.key for spec in bundle.specs] == [
        "axiom_i_family_predictions.criteria_satisfaction",
        "axiom_i_family_predictions.rejected_model_classes",
        "axiom_i_family_predictions.conditional_predictions",
        "axiom_i_family_predictions.decision_rule",
        "axiom_i_family_predictions.su_n_qualia_confinement_boundary",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_family_predictions_builder_keeps_prediction_and_rejection_labels() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "amplitude-phase decomposition Psi = rho exp(i theta)"
        in specs["axiom_i_family_predictions.criteria_satisfaction"].source_formulae
    )
    assert (
        "real scalar lacks internal phase"
        in specs["axiom_i_family_predictions.rejected_model_classes"].source_formulae
    )
    assert (
        "non-Abelian base rejected at the minimal tier for now"
        in specs["axiom_i_family_predictions.rejected_model_classes"].source_formulae
    )
    assert (
        "conserved Psi-charge Noether current"
        in specs["axiom_i_family_predictions.conditional_predictions"].source_formulae
    )
    assert (
        "massive infoton m_A = g v"
        in specs["axiom_i_family_predictions.conditional_predictions"].source_formulae
    )
    assert (
        "evidence against predictions triggers model-class escalation or replacement"
        in specs["axiom_i_family_predictions.decision_rule"].source_formulae
    )


def test_family_predictions_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R00753":
            continue
        records.append(
            {
                "ledger_id": record_id,
                "source_record_id": f"{record_id}:stub",
                "source_block_index": int(record_id[3:]),
                "section_path": "Paper 0 > Axiom I > Conditional Predictions",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_axiom_i_family_predictions_specs(records)


def test_family_predictions_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_axiom_i_family_predictions_validation_spec(
        "axiom_i_family_predictions.conditional_predictions",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Axiom I Family Predictions Specs" in report
    assert "Noether" in report
    assert loaded["key"] == "axiom_i_family_predictions.conditional_predictions"
    assert "massive infoton m_A = g v" in loaded["source_formulae"]
    assert "Family Predictions" in render_report(bundle)
