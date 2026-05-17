# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 derived interaction opening builder tests
"""Tests for Paper 0 derived interaction opening spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_derived_interaction_opening_validation_spec,
)
from scripts.build_paper0_derived_interaction_opening_specs import (
    SOURCE_LEDGER_IDS,
    build_derived_interaction_opening_specs,
    build_from_ledger,
    render_report,
    write_outputs,
)


def test_derived_interaction_opening_builder_preserves_full_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R01384", "P0R01421"]
    assert bundle.summary["source_record_count"] == 38
    assert bundle.summary["consumed_source_record_count"] == 38
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["math_ids"] == []
    assert bundle.summary["image_ids"] == ["IMG0022", "IMG0023"]
    assert bundle.summary["table_ids"] == []
    assert bundle.summary["next_source_boundary"] == "P0R01422"
    assert [spec.key for spec in bundle.specs] == [
        "derived_interaction_opening.gauge_theory_grounding",
        "derived_interaction_opening.predictive_coding_mapping",
        "derived_interaction_opening.h_int_gauge_identification",
        "derived_interaction_opening.intrinsic_properties_quantum_numbers",
        "derived_interaction_opening.gauge_principle_nonabelian_boundary",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_derived_interaction_opening_builder_keeps_equations_and_boundaries() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "global U(1) phase symmetry implies conserved Psi-charge via Noether theorem"
        in specs["derived_interaction_opening.gauge_theory_grounding"].source_formulae
    )
    assert (
        "ig A_mu (Psi* partial_mu Psi - Psi partial_mu Psi*) drives infoton signal generation"
        in specs["derived_interaction_opening.predictive_coding_mapping"].source_formulae
    )
    assert (
        "L_interaction = i g A_mu J_mu"
        in specs["derived_interaction_opening.h_int_gauge_identification"].source_formulae
    )
    assert (
        "Psi-field couples via local U(1) to spin-1 infoton A_mu"
        in specs[
            "derived_interaction_opening.intrinsic_properties_quantum_numbers"
        ].source_formulae
    )
    assert (
        "Non-Abelian SU(N) internal qualia structure is a hypothesis"
        in specs["derived_interaction_opening.gauge_principle_nonabelian_boundary"].source_formulae
    )


def test_derived_interaction_opening_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R01418":
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
                "section_path": "Paper 0 > Derived Interaction Opening",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_derived_interaction_opening_specs(records)


def test_derived_interaction_opening_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_derived_interaction_opening_validation_spec(
        "derived_interaction_opening.h_int_gauge_identification",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Derived Interaction Opening Specs" in report
    assert loaded["key"] == "derived_interaction_opening.h_int_gauge_identification"
    assert "H_int" in loaded["source_formulae"][0]
    assert "Derived Interaction Opening" in render_report(bundle)
