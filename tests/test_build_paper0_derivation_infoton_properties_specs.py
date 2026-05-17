# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 infoton-properties derivation builder tests
"""Tests for Paper 0 infoton-properties derivation spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_derivation_infoton_properties_validation_spec,
)
from scripts.build_paper0_derivation_infoton_properties_specs import (
    SOURCE_LEDGER_IDS,
    build_derivation_infoton_properties_specs,
    build_from_ledger,
    render_report,
    write_outputs,
)


def test_derivation_infoton_properties_builder_preserves_full_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R01623", "P0R01637"]
    assert bundle.summary["source_record_count"] == 15
    assert bundle.summary["consumed_source_record_count"] == 15
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 4
    assert bundle.summary["math_ids"] == []
    assert bundle.summary["image_ids"] == []
    assert bundle.summary["table_ids"] == []
    assert bundle.summary["next_source_boundary"] == "P0R01638"
    assert [spec.key for spec in bundle.specs] == [
        "derivation_infoton_properties.lagrangian_and_potential",
        "derivation_infoton_properties.vev_and_goldstone_absorption",
        "derivation_infoton_properties.mass_identification",
        "derivation_infoton_properties.range_consequence",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_derivation_infoton_properties_builder_keeps_equations_and_boundaries() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "L = (D_mu Psi)^*(D^mu Psi) - V(|Psi|) - 1/4 F_mu_nu F^mu_nu"
        in specs["derivation_infoton_properties.lagrangian_and_potential"].source_formulae
    )
    assert (
        "D_mu = partial_mu - i g A_mu"
        in specs["derivation_infoton_properties.lagrangian_and_potential"].source_formulae
    )
    assert (
        "V(|Psi|) = -mu^2 |Psi|^2 + lambda |Psi|^4"
        in specs["derivation_infoton_properties.lagrangian_and_potential"].source_formulae
    )
    assert (
        "Psi(x) = 1/sqrt(2) (v + h(x)) exp(i xi(x)/v)"
        in specs["derivation_infoton_properties.vev_and_goldstone_absorption"].source_formulae
    )
    assert (
        "Goldstone boson xi(x) is absorbed by A_mu and removed from the physical spectrum"
        in specs["derivation_infoton_properties.vev_and_goldstone_absorption"].source_formulae
    )
    assert (
        "m_A = g v" in specs["derivation_infoton_properties.mass_identification"].source_formulae
    )
    assert (
        "range of a force mediated by mass m_A is approximately lambda_range ~= hbar / (m_A c)"
        in specs["derivation_infoton_properties.range_consequence"].source_formulae
    )


def test_derivation_infoton_properties_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R01631":
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
                "section_path": "Paper 0 > Derivation of the Infoton's Properties",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_derivation_infoton_properties_specs(records)


def test_derivation_infoton_properties_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_derivation_infoton_properties_validation_spec(
        "derivation_infoton_properties.mass_identification",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Derivation Infoton Properties Specs" in report
    assert loaded["key"] == "derivation_infoton_properties.mass_identification"
    assert "m_A = g v" in loaded["source_formulae"]
    assert "Derivation Infoton Properties" in render_report(bundle)
