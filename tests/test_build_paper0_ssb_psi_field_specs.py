# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 SSB Psi-field builder tests
"""Tests for Paper 0 SSB Psi-field spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import load_ssb_psi_field_validation_spec
from scripts.build_paper0_ssb_psi_field_specs import (
    SOURCE_LEDGER_IDS,
    build_from_ledger,
    build_ssb_psi_field_specs,
    render_report,
    write_outputs,
)


def test_ssb_psi_field_builder_preserves_full_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R01272", "P0R01332"]
    assert bundle.summary["source_record_count"] == 61
    assert bundle.summary["consumed_source_record_count"] == 61
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 8
    assert bundle.summary["math_ids"] == ["EQ0013", "EQ0014"]
    assert bundle.summary["image_ids"] == ["IMG0021"]
    assert bundle.summary["table_ids"] == []
    assert bundle.summary["next_source_boundary"] == "P0R01333"
    assert [spec.key for spec in bundle.specs] == [
        "ssb_psi_field.section_overview_and_three_implications",
        "ssb_psi_field.popular_context_short_range_particle_self",
        "ssb_psi_field.predictive_coding_core_belief",
        "ssb_psi_field.psi_s_coupling_integration",
        "ssb_psi_field.mexican_hat_vacuum_selection",
        "ssb_psi_field.eft_sextic_stability_and_mass",
        "ssb_psi_field.global_goldstone_boundary",
        "ssb_psi_field.local_higgs_architecture_implications",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_ssb_psi_field_builder_keeps_equations_and_boundaries() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "V(|Psi|) = -mu^2 |Psi|^2 + lambda |Psi|^4"
        in specs["ssb_psi_field.section_overview_and_three_implications"].source_formulae
    )
    assert (
        "Psi-Higgs is a predicted new particle, not observed evidence"
        in specs["ssb_psi_field.popular_context_short_range_particle_self"].source_formulae
    )
    assert (
        "Psi = 0 represents a flat maximum-entropy prior"
        in specs["ssb_psi_field.predictive_coding_core_belief"].source_formulae
    )
    assert (
        "H_int = -lambda * Psi_s * sigma"
        in specs["ssb_psi_field.psi_s_coupling_integration"].source_formulae
    )
    assert (
        "true vacua form a circle with |Psi| = sqrt(mu^2/lambda)"
        in specs["ssb_psi_field.mexican_hat_vacuum_selection"].source_formulae
    )
    assert (
        "m_h^2 = -2 mu^2 + 12 lambda v^2 + 30 gamma v^4/Lambda^2"
        in specs["ssb_psi_field.eft_sextic_stability_and_mass"].source_formulae
    )
    assert (
        "global U(1) breaking gives one massive scalar and one massless Goldstone boson"
        in specs["ssb_psi_field.global_goldstone_boundary"].source_formulae
    )
    assert (
        "m_A = sqrt(2) g v"
        in specs["ssb_psi_field.local_higgs_architecture_implications"].source_formulae
    )


def test_ssb_psi_field_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R01306":
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
                "section_path": "Paper 0 > SSB Psi Field",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_ssb_psi_field_specs(records)


def test_ssb_psi_field_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_ssb_psi_field_validation_spec(
        "ssb_psi_field.eft_sextic_stability_and_mass",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 SSB Psi-Field Specs" in report
    assert loaded["key"] == "ssb_psi_field.eft_sextic_stability_and_mass"
    assert "bounded from below" in loaded["source_formulae"][0]
    assert "SSB Psi-Field" in render_report(bundle)
