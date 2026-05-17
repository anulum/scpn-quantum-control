# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 phenomenological Lagrangian builder tests
"""Tests for Paper 0 phenomenological Lagrangian spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_phenomenological_lagrangian_validation_spec,
)
from scripts.build_paper0_phenomenological_lagrangian_specs import (
    SOURCE_LEDGER_IDS,
    build_from_ledger,
    build_phenomenological_lagrangian_specs,
    render_report,
    write_outputs,
)


def test_phenomenological_lagrangian_builder_preserves_full_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R01333", "P0R01383"]
    assert bundle.summary["source_record_count"] == 51
    assert bundle.summary["consumed_source_record_count"] == 51
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["math_ids"] == [
        "EQ0015",
        "EQ0016",
        "EQ0017",
        "EQ0018",
        "EQ0019",
        "EQ0020",
    ]
    assert bundle.summary["image_ids"] == []
    assert bundle.summary["table_ids"] == []
    assert bundle.summary["next_source_boundary"] == "P0R01384"
    assert [spec.key for spec in bundle.specs] == [
        "phenomenological_lagrangian.section_opening_dual_coupling",
        "phenomenological_lagrangian.predictive_coding_free_energy",
        "phenomenological_lagrangian.black_box_interaction",
        "phenomenological_lagrangian.master_interaction_terms",
        "phenomenological_lagrangian.architecture_stationary_action",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_phenomenological_lagrangian_builder_keeps_equations_and_boundaries() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "L_total decomposes into L_Psi, L_Physical, and L_Int"
        in specs["phenomenological_lagrangian.section_opening_dual_coupling"].source_formulae
    )
    assert (
        "stationary action is mapped to the Free Energy Principle"
        in specs["phenomenological_lagrangian.predictive_coding_free_energy"].source_formulae
    )
    assert (
        "H_int = -lambda * Psi_s * sigma"
        in specs["phenomenological_lagrangian.black_box_interaction"].source_formulae
    )
    assert (
        "L_Int = L_Geometric + L_Informational"
        in specs["phenomenological_lagrangian.master_interaction_terms"].source_formulae
    )
    assert (
        "Z = integral D Psi D Phi_Physical exp(i S_Master / hbar)"
        in specs["phenomenological_lagrangian.architecture_stationary_action"].source_formulae
    )
    assert (
        "delta S_Master / delta theta_iL = 0 implies d theta_iL / dt = omega_iL + sum K_ij sin(Delta theta) + ..."
        in specs["phenomenological_lagrangian.architecture_stationary_action"].source_formulae
    )


def test_phenomenological_lagrangian_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R01370":
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
                "section_path": "Paper 0 > Phenomenological Lagrangian",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_phenomenological_lagrangian_specs(records)


def test_phenomenological_lagrangian_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_phenomenological_lagrangian_validation_spec(
        "phenomenological_lagrangian.master_interaction_terms",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Phenomenological Lagrangian Specs" in report
    assert loaded["key"] == "phenomenological_lagrangian.master_interaction_terms"
    assert "L_Total" in loaded["source_formulae"][0]
    assert "Phenomenological Lagrangian" in render_report(bundle)
