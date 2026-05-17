# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 geometric coupling consistency builder tests
"""Tests for Paper 0 geometric-coupling consistency spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_geometric_coupling_consistency_validation_spec,
)
from scripts.build_paper0_geometric_coupling_consistency_specs import (
    SOURCE_LEDGER_IDS,
    build_from_ledger,
    build_geometric_coupling_consistency_specs,
    render_report,
    write_outputs,
)


def test_geometric_coupling_consistency_builder_preserves_full_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R01135", "P0R01188"]
    assert bundle.summary["source_record_count"] == 54
    assert bundle.summary["consumed_source_record_count"] == 54
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 6
    assert bundle.summary["math_ids"] == ["EQ0010", "EQ0011", "EQ0012"]
    assert bundle.summary["image_ids"] == ["IMG0020"]
    assert bundle.summary["table_ids"] == ["TBL002"]
    assert bundle.summary["next_source_boundary"] == "P0R01189"
    assert [spec.key for spec in bundle.specs] == [
        "geometric_coupling_consistency.coupling_problem_boundary",
        "geometric_coupling_consistency.minimal_curved_spacetime_coupling",
        "geometric_coupling_consistency.non_minimal_consistency_condition",
        "geometric_coupling_consistency.derived_geometric_lagrangian",
        "geometric_coupling_consistency.complete_covariant_action",
        "geometric_coupling_consistency.interpretation_prediction_comments",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_geometric_coupling_consistency_builder_keeps_equations_and_boundaries() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "direct non-minimal coupling to the Ricci scalar R needs a separate principle"
        in specs["geometric_coupling_consistency.coupling_problem_boundary"].source_formulae
    )
    assert (
        "L_Psi_curved = g^{mu nu}(nabla_mu Psi)^*(nabla_nu Psi) - V(|Psi|)"
        in specs[
            "geometric_coupling_consistency.minimal_curved_spacetime_coupling"
        ].source_formulae
    )
    assert (
        "L_non_minimal = - xi R Psi^* Psi"
        in specs[
            "geometric_coupling_consistency.non_minimal_consistency_condition"
        ].source_formulae
    )
    assert (
        "massless scalar conformal invariance selects xi = 1/6"
        in specs[
            "geometric_coupling_consistency.non_minimal_consistency_condition"
        ].source_formulae
    )
    assert (
        "L_Geometric_prime = - g_PsiG R Psi^* Psi"
        in specs["geometric_coupling_consistency.derived_geometric_lagrangian"].source_formulae
    )
    assert (
        "L_Int_prime = L_Informational_prime + L_Geometric_prime"
        in specs["geometric_coupling_consistency.complete_covariant_action"].source_formulae
    )
    assert (
        "j_Psi^mu = i g (Psi^* nabla_mu Psi - Psi nabla_mu Psi^*)"
        in specs[
            "geometric_coupling_consistency.interpretation_prediction_comments"
        ].source_formulae
    )


def test_geometric_coupling_consistency_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R01162":
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
                "section_path": "Paper 0 > Geometric Coupling Consistency",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_geometric_coupling_consistency_specs(records)


def test_geometric_coupling_consistency_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_geometric_coupling_consistency_validation_spec(
        "geometric_coupling_consistency.non_minimal_consistency_condition",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Geometric Coupling Consistency Specs" in report
    assert loaded["key"] == "geometric_coupling_consistency.non_minimal_consistency_condition"
    assert "xi is a dimensionless coupling constant" in loaded["source_formulae"]
    assert "Geometric Coupling Consistency" in render_report(bundle)
