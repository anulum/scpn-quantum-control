# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 mass eigenstates mixing-angle builder tests
"""Tests for Paper 0 mass-eigenstates mixing-angle spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_mass_eigenstates_mixing_angle_validation_spec,
)
from scripts.build_paper0_mass_eigenstates_mixing_angle_specs import (
    SOURCE_LEDGER_IDS,
    build_from_ledger,
    build_mass_eigenstates_mixing_angle_specs,
    render_report,
    write_outputs,
)


def test_mass_eigenstates_mixing_angle_builder_preserves_full_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R01669", "P0R01683"]
    assert bundle.summary["source_record_count"] == 15
    assert bundle.summary["consumed_source_record_count"] == 15
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 3
    assert bundle.summary["math_ids"] == []
    assert bundle.summary["image_ids"] == []
    assert bundle.summary["table_ids"] == []
    assert bundle.summary["next_source_boundary"] == "P0R01684"
    assert [spec.key for spec in bundle.specs] == [
        "mass_eigenstates_mixing_angle.mass_eigenstate_rotation",
        "mass_eigenstates_mixing_angle.lhc_invisible_decay_bound",
        "mass_eigenstates_mixing_angle.perturbative_target_boundary",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_mass_eigenstates_mixing_angle_builder_keeps_rotation_bounds_and_boundaries() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "[h_SM, h_Psi]^T = [[cos theta, sin theta], [-sin theta, cos theta]] [h_bare, h_Psi,bare]^T"
        in specs["mass_eigenstates_mixing_angle.mass_eigenstate_rotation"].source_formulae
    )
    assert (
        "tan(2 theta) = 2 lambda_mix v_h v_psi / (m_h_bare^2 - m_Psi_bare^2)"
        in specs["mass_eigenstates_mixing_angle.mass_eigenstate_rotation"].source_formulae
    )
    assert (
        "ATLAS and CMS constrain BR_inv < 0.11 at 95 percent confidence level"
        in specs["mass_eigenstates_mixing_angle.lhc_invisible_decay_bound"].source_formulae
    )
    assert (
        "sin^2 theta lesssim 0.1 implies sin theta lesssim 0.31"
        in specs["mass_eigenstates_mixing_angle.lhc_invisible_decay_bound"].source_formulae
    )
    assert (
        "lambda_mix must be perturbatively small, lambda_mix << 1"
        in specs["mass_eigenstates_mixing_angle.perturbative_target_boundary"].source_formulae
    )
    assert (
        "subsequent search strategies are anchored in a falsification-or-discovery parameter target"
        in specs["mass_eigenstates_mixing_angle.perturbative_target_boundary"].source_formulae
    )


def test_mass_eigenstates_mixing_angle_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R01680":
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
                "section_path": "Paper 0 > Mass Eigenstates Mixing Angle",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_mass_eigenstates_mixing_angle_specs(records)


def test_mass_eigenstates_mixing_angle_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_mass_eigenstates_mixing_angle_validation_spec(
        "mass_eigenstates_mixing_angle.lhc_invisible_decay_bound",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Mass Eigenstates Mixing Angle Specs" in report
    assert loaded["key"] == "mass_eigenstates_mixing_angle.lhc_invisible_decay_bound"
    assert "BR_inv" in loaded["source_formulae"][4]
    assert "Mass Eigenstates Mixing Angle" in render_report(bundle)
