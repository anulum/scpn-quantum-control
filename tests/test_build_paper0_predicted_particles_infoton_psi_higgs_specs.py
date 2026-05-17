# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 predicted particles builder tests
"""Tests for Paper 0 infoton and Psi-Higgs prediction spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_predicted_particles_infoton_psi_higgs_validation_spec,
)
from scripts.build_paper0_predicted_particles_infoton_psi_higgs_specs import (
    SOURCE_LEDGER_IDS,
    build_from_ledger,
    build_predicted_particles_infoton_psi_higgs_specs,
    render_report,
    write_outputs,
)


def test_predicted_particles_builder_preserves_full_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R01597", "P0R01622"]
    assert bundle.summary["source_record_count"] == 26
    assert bundle.summary["consumed_source_record_count"] == 26
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 4
    assert bundle.summary["math_ids"] == []
    assert bundle.summary["image_ids"] == []
    assert bundle.summary["table_ids"] == []
    assert bundle.summary["next_source_boundary"] == "P0R01623"
    assert [spec.key for spec in bundle.specs] == [
        "predicted_particles_infoton_psi_higgs.particle_prediction_opening",
        "predicted_particles_infoton_psi_higgs.search_strategy_summary",
        "predicted_particles_infoton_psi_higgs.active_inference_mapping",
        "predicted_particles_infoton_psi_higgs.h_int_falsifiability_bridge",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_predicted_particles_builder_keeps_equations_searches_and_boundaries() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "infoton gauge field A_mu receives mass m_A = g v"
        in specs[
            "predicted_particles_infoton_psi_higgs.particle_prediction_opening"
        ].source_formulae
    )
    assert (
        "Psi-Higgs radial excitation mass is m_h = sqrt(2 lambda) v"
        in specs[
            "predicted_particles_infoton_psi_higgs.particle_prediction_opening"
        ].source_formulae
    )
    assert (
        "LHC search strategy includes exotic Higgs decays h_SM -> h_Psi h_Psi"
        in specs["predicted_particles_infoton_psi_higgs.search_strategy_summary"].source_formulae
    )
    assert (
        "bosonic clouds around spinning black holes could yield monochromatic gravitational waves"
        in specs["predicted_particles_infoton_psi_higgs.search_strategy_summary"].source_formulae
    )
    assert (
        "massive infoton carries prediction-error signal"
        in specs["predicted_particles_infoton_psi_higgs.active_inference_mapping"].source_formulae
    )
    assert (
        "H_int = -lambda * Psi_s * sigma"
        in specs[
            "predicted_particles_infoton_psi_higgs.h_int_falsifiability_bridge"
        ].source_formulae
    )
    assert (
        "interaction range is set by infoton mass m_A = g v"
        in specs[
            "predicted_particles_infoton_psi_higgs.h_int_falsifiability_bridge"
        ].source_formulae
    )


def test_predicted_particles_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R01608":
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
                "section_path": "Paper 0 > Predicted Particles",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_predicted_particles_infoton_psi_higgs_specs(records)


def test_predicted_particles_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_predicted_particles_infoton_psi_higgs_validation_spec(
        "predicted_particles_infoton_psi_higgs.h_int_falsifiability_bridge",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Predicted Particles Infoton Psi-Higgs Specs" in report
    assert loaded["key"] == "predicted_particles_infoton_psi_higgs.h_int_falsifiability_bridge"
    assert "H_int" in loaded["source_formulae"][0]
    assert "Predicted Particles" in render_report(bundle)
