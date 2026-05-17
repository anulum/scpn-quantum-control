# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 derived Lagrangian detail builder tests
"""Tests for Paper 0 derived Lagrangian detail spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import load_derived_lagrangian_detail_validation_spec
from scripts.build_paper0_derived_lagrangian_detail_specs import (
    SOURCE_LEDGER_IDS,
    build_derived_lagrangian_detail_specs,
    build_from_ledger,
    render_report,
    write_outputs,
)


def test_derived_lagrangian_detail_builder_preserves_full_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R01422", "P0R01509"]
    assert bundle.summary["source_record_count"] == 88
    assert bundle.summary["consumed_source_record_count"] == 88
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 7
    assert bundle.summary["math_ids"] == ["EQ0021", "EQ0022"]
    assert bundle.summary["image_ids"] == ["IMG0024", "IMG0025", "IMG0026", "IMG0027", "IMG0028"]
    assert bundle.summary["table_ids"] == []
    assert bundle.summary["next_source_boundary"] == "P0R01510"
    assert [spec.key for spec in bundle.specs] == [
        "derived_lagrangian_detail.derived_lint_split",
        "derived_lagrangian_detail.informational_lagrangian_fim_kinetics",
        "derived_lagrangian_detail.operational_pullback_protocol",
        "derived_lagrangian_detail.observable_l4_l5_prediction",
        "derived_lagrangian_detail.neural_fim_covariance_strategy",
        "derived_lagrangian_detail.domain_constraints_local_physics",
        "derived_lagrangian_detail.geometric_constants_predictions",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_derived_lagrangian_detail_builder_keeps_equations_and_boundaries() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "L_Int' = L_Informational' + L_Geometric'"
        in specs["derived_lagrangian_detail.derived_lint_split"].source_formulae
    )
    assert (
        "g_tilde_F^{mu nu} = Lambda_I^-2 (pi* g_F)^{mu nu}"
        in specs["derived_lagrangian_detail.informational_lagrangian_fim_kinetics"].source_formulae
    )
    assert (
        "L_gauge = -1/4 g_tilde_F^{mu alpha} g_tilde_F^{nu beta} F_mu_nu F_alpha_beta"
        in specs["derived_lagrangian_detail.operational_pullback_protocol"].source_formulae
    )
    assert (
        "NV-centre sensors are prediction targets, not reported evidence"
        in specs["derived_lagrangian_detail.observable_l4_l5_prediction"].source_formulae
    )
    assert (
        "FIM must use full covariance matrix Sigma(theta)"
        in specs["derived_lagrangian_detail.neural_fim_covariance_strategy"].source_formulae
    )
    assert (
        "pullback cannot reference non-local or acausal data"
        in specs["derived_lagrangian_detail.domain_constraints_local_physics"].source_formulae
    )
    assert (
        "L_Geometric' = -xi R Psi* Psi"
        in specs["derived_lagrangian_detail.geometric_constants_predictions"].source_formulae
    )


def test_derived_lagrangian_detail_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R01448":
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
                "section_path": "Paper 0 > Derived Lagrangian Detail",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_derived_lagrangian_detail_specs(records)


def test_derived_lagrangian_detail_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_derived_lagrangian_detail_validation_spec(
        "derived_lagrangian_detail.operational_pullback_protocol",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Derived Lagrangian Detail Specs" in report
    assert loaded["key"] == "derived_lagrangian_detail.operational_pullback_protocol"
    assert "I_ij" in loaded["source_formulae"][2]
    assert "Derived Lagrangian Detail" in render_report(bundle)
