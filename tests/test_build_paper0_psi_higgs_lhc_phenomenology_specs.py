# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Psi-Higgs LHC phenomenology builder tests
"""Tests for Paper 0 Psi-Higgs LHC phenomenology spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_psi_higgs_lhc_phenomenology_validation_spec,
)
from scripts.build_paper0_psi_higgs_lhc_phenomenology_specs import (
    SOURCE_LEDGER_IDS,
    build_from_ledger,
    build_psi_higgs_lhc_phenomenology_specs,
    render_report,
    write_outputs,
)


def test_psi_higgs_lhc_phenomenology_builder_preserves_full_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R01655", "P0R01668"]
    assert bundle.summary["source_record_count"] == 14
    assert bundle.summary["consumed_source_record_count"] == 14
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 3
    assert bundle.summary["math_ids"] == []
    assert bundle.summary["image_ids"] == []
    assert bundle.summary["table_ids"] == []
    assert bundle.summary["next_source_boundary"] == "P0R01669"
    assert [spec.key for spec in bundle.specs] == [
        "psi_higgs_lhc_phenomenology.phenomenology_bridge",
        "psi_higgs_lhc_phenomenology.scalar_mixing_mechanism",
        "psi_higgs_lhc_phenomenology.scalar_potential_and_cross_term",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_psi_higgs_lhc_phenomenology_builder_keeps_portal_formalism_and_boundaries() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "quantum excitation of the Psi-field is termed the Psi-Higgs boson h_Psi"
        in specs["psi_higgs_lhc_phenomenology.phenomenology_bridge"].source_formulae
    )
    assert (
        "V(|Psi|) = -m^2 |Psi|^2 + lambda |Psi|^4 facilitates SSB"
        in specs["psi_higgs_lhc_phenomenology.scalar_mixing_mechanism"].source_formulae
    )
    assert (
        "Psi-field scalar can mix with the Standard Model Higgs h_SM"
        in specs["psi_higgs_lhc_phenomenology.scalar_mixing_mechanism"].source_formulae
    )
    assert (
        "V(H, Psi) = V_SM(H) + V_Psi(Psi) + V_mix(H, Psi)"
        in specs["psi_higgs_lhc_phenomenology.scalar_potential_and_cross_term"].source_formulae
    )
    assert (
        "V_mix = lambda_mix (H^dagger H) |Psi|^2"
        in specs["psi_higgs_lhc_phenomenology.scalar_potential_and_cross_term"].source_formulae
    )
    assert (
        "mixing term generates lambda_mix v_h v_psi h_bare h_Psi,bare in the mass matrix"
        in specs["psi_higgs_lhc_phenomenology.scalar_potential_and_cross_term"].source_formulae
    )


def test_psi_higgs_lhc_phenomenology_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R01662":
            continue
        records.append(
            {
                "ledger_id": record_id,
                "source_record_id": f"{record_id}:stub",
                "source_block_index": int(record_id[3:]),
                "canonical_category": "claim",
                "block_type": "Para",
                "math_ids": [],
                "image_ids": [],
                "table_id": None,
                "section_path": "Paper 0 > Psi-Higgs LHC Phenomenology",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_psi_higgs_lhc_phenomenology_specs(records)


def test_psi_higgs_lhc_phenomenology_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_psi_higgs_lhc_phenomenology_validation_spec(
        "psi_higgs_lhc_phenomenology.scalar_potential_and_cross_term",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Psi-Higgs LHC Phenomenology Specs" in report
    assert loaded["key"] == "psi_higgs_lhc_phenomenology.scalar_potential_and_cross_term"
    assert "V_mix = lambda_mix" in loaded["source_formulae"][3]
    assert "Psi-Higgs LHC Phenomenology" in render_report(bundle)
