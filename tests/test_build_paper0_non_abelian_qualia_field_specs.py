# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Non-Abelian qualia field builder tests
"""Tests for Paper 0 Non-Abelian qualia-field spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import load_non_abelian_qualia_field_validation_spec
from scripts.build_paper0_non_abelian_qualia_field_specs import (
    SOURCE_LEDGER_IDS,
    build_from_ledger,
    build_non_abelian_qualia_field_specs,
    render_report,
    write_outputs,
)


def test_non_abelian_qualia_field_builder_preserves_full_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R01103", "P0R01134"]
    assert bundle.summary["source_record_count"] == 32
    assert bundle.summary["consumed_source_record_count"] == 32
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["structural_record_count"] == 2
    assert bundle.summary["context_record_count"] == 13
    assert bundle.summary["claim_record_count"] == 13
    assert bundle.summary["validation_target_record_count"] == 4
    assert bundle.summary["anomaly_condition_record_count"] == 10
    assert bundle.summary["confinement_record_count"] == 9
    assert bundle.summary["topological_entanglement_record_count"] == 7
    assert bundle.summary["next_source_boundary"] == "P0R01135"
    assert [spec.key for spec in bundle.specs] == [
        "non_abelian_qualia_field.boundary_and_rationale",
        "non_abelian_qualia_field.self_interacting_gauge_bosons",
        "non_abelian_qualia_field.anomaly_cancellation_condition",
        "non_abelian_qualia_field.confinement_binding_boundary",
        "non_abelian_qualia_field.topological_entanglement_resolution",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_non_abelian_qualia_field_builder_keeps_equations_and_boundaries() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "Beyond U(1): The Hypothesis of a Non-Abelian Qualia Field"
        in specs["non_abelian_qualia_field.boundary_and_rationale"].source_formulae
    )
    assert (
        "F_munua = partial_mu A_nua - partial_nu A_mua + g f_abc A_mub A_nuc"
        in specs["non_abelian_qualia_field.self_interacting_gauge_bosons"].source_formulae
    )
    assert (
        "sum([d_abc * q_i_a * q_i_b * q_i_c for i in fermions]) == 0"
        in specs["non_abelian_qualia_field.anomaly_cancellation_condition"].source_formulae
    )
    assert (
        "is_renormalizable = sum([d_abc * q_i_a * q_i_b * q_i_c for i in transducers]) == 0"
        in specs["non_abelian_qualia_field.anomaly_cancellation_condition"].source_formulae
    )
    assert (
        "V(r) approx sigma r linearly increasing potential is a source claim"
        in specs["non_abelian_qualia_field.confinement_binding_boundary"].source_formulae
    )
    assert (
        "SU(N) confinement is defined as topological entanglement, not singlet neutrality"
        in specs["non_abelian_qualia_field.topological_entanglement_resolution"].source_formulae
    )
    assert (
        "Qualia-Balls are predicted as pure Info-Gluon bound-state configurations"
        in specs["non_abelian_qualia_field.topological_entanglement_resolution"].source_formulae
    )


def test_non_abelian_qualia_field_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R01125":
            continue
        records.append(
            {
                "ledger_id": record_id,
                "source_record_id": f"{record_id}:stub",
                "source_block_index": int(record_id[3:]),
                "section_path": "Paper 0 > Non-Abelian Qualia Field",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_non_abelian_qualia_field_specs(records)


def test_non_abelian_qualia_field_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_non_abelian_qualia_field_validation_spec(
        "non_abelian_qualia_field.anomaly_cancellation_condition",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Non-Abelian Qualia Field Specs" in report
    assert loaded["key"] == "non_abelian_qualia_field.anomaly_cancellation_condition"
    assert "d_abc is the fully symmetric tensor" in loaded["source_formulae"]
    assert "Non-Abelian Qualia Field" in render_report(bundle)
