# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Psi-Higgs scalar builder tests
"""Tests for Paper 0 Psi-Higgs new-scalar-particle spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_psi_higgs_new_scalar_particle_validation_spec,
)
from scripts.build_paper0_psi_higgs_new_scalar_particle_specs import (
    SOURCE_LEDGER_IDS,
    build_from_ledger,
    build_psi_higgs_new_scalar_particle_specs,
    render_report,
    write_outputs,
)


def test_psi_higgs_new_scalar_particle_builder_preserves_full_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R01638", "P0R01646"]
    assert bundle.summary["source_record_count"] == 9
    assert bundle.summary["consumed_source_record_count"] == 9
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 3
    assert bundle.summary["math_ids"] == []
    assert bundle.summary["image_ids"] == []
    assert bundle.summary["table_ids"] == []
    assert bundle.summary["next_source_boundary"] == "P0R01647"
    assert [spec.key for spec in bundle.specs] == [
        "psi_higgs_new_scalar_particle.scalar_remnant_identity",
        "psi_higgs_new_scalar_particle.potential_mass_term",
        "psi_higgs_new_scalar_particle.mass_and_detection_boundary",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_psi_higgs_new_scalar_particle_builder_keeps_mass_terms_and_boundaries() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "h(x) magnitude fluctuations correspond to a new massive physical scalar particle"
        in specs["psi_higgs_new_scalar_particle.scalar_remnant_identity"].source_formulae
    )
    assert (
        "V(v+h) ~= V(v) + 1/2 (2 mu^2) h^2 + ..."
        in specs["psi_higgs_new_scalar_particle.potential_mass_term"].source_formulae
    )
    assert (
        "L_mass,h = -lambda v^2 h^2"
        in specs["psi_higgs_new_scalar_particle.potential_mass_term"].source_formulae
    )
    assert (
        "physical particle mass is m_h = sqrt(2 lambda) v"
        in specs["psi_higgs_new_scalar_particle.mass_and_detection_boundary"].source_formulae
    )
    assert (
        "discovery would provide evidence only if observed; this fixture records the source prediction"
        in specs["psi_higgs_new_scalar_particle.mass_and_detection_boundary"].source_formulae
    )


def test_psi_higgs_new_scalar_particle_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R01643":
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
                "section_path": "Paper 0 > Psi-Higgs New Scalar Particle",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_psi_higgs_new_scalar_particle_specs(records)


def test_psi_higgs_new_scalar_particle_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_psi_higgs_new_scalar_particle_validation_spec(
        "psi_higgs_new_scalar_particle.mass_and_detection_boundary",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Psi-Higgs New Scalar Particle Specs" in report
    assert loaded["key"] == "psi_higgs_new_scalar_particle.mass_and_detection_boundary"
    assert "m_h = sqrt(2 lambda) v" in loaded["source_formulae"]
    assert "Psi-Higgs New Scalar Particle" in render_report(bundle)
