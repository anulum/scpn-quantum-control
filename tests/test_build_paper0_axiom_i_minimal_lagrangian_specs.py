# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom I minimal Lagrangian builder tests
"""Tests for Paper 0 Axiom I minimal Psi-field Lagrangian spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_axiom_i_minimal_lagrangian_validation_spec,
)
from scripts.build_paper0_axiom_i_minimal_lagrangian_specs import (
    SOURCE_LEDGER_IDS,
    build_axiom_i_minimal_lagrangian_specs,
    build_from_ledger,
    render_report,
    write_outputs,
)


def test_minimal_lagrangian_builder_preserves_contiguous_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R00733", "P0R00746"]
    assert bundle.summary["source_record_count"] == 14
    assert bundle.summary["consumed_source_record_count"] == 14
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["minimal_criterion_count"] == 3
    assert bundle.summary["equation_record_count"] == 4
    assert bundle.summary["next_source_boundary"] == "P0R00747"
    assert [spec.key for spec in bundle.specs] == [
        "axiom_i_minimal_lagrangian.purpose_and_criteria",
        "axiom_i_minimal_lagrangian.traceable_model_class_boundary",
        "axiom_i_minimal_lagrangian.minimal_family_field_content",
        "axiom_i_minimal_lagrangian.l_min_operator_terms",
        "axiom_i_minimal_lagrangian.potential_and_ssb_boundary",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_minimal_lagrangian_builder_keeps_equation_and_operator_labels() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "P0R00742:L_min_lagrangian_density"
        in specs["axiom_i_minimal_lagrangian.l_min_operator_terms"].source_equation_ids
    )
    assert (
        "L_min = |D_mu Psi|^2 - V(|Psi|) - 1/4 g_F F F - xi R |Psi|^2"
        in specs["axiom_i_minimal_lagrangian.l_min_operator_terms"].source_formulae
    )
    assert (
        "D_mu = partial_mu - i g A_mu"
        in specs["axiom_i_minimal_lagrangian.l_min_operator_terms"].source_formulae
    )
    assert (
        "F_mu_nu = partial_mu A_nu - partial_nu A_mu"
        in specs["axiom_i_minimal_lagrangian.l_min_operator_terms"].source_formulae
    )
    assert (
        "V(|Psi|) = -mu^2 |Psi|^2 + lambda |Psi|^4"
        in specs["axiom_i_minimal_lagrangian.potential_and_ssb_boundary"].source_formulae
    )
    assert (
        "boundedness below and spontaneous symmetry breaking"
        in specs["axiom_i_minimal_lagrangian.potential_and_ssb_boundary"].source_formulae
    )


def test_minimal_lagrangian_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R00742":
            continue
        records.append(
            {
                "ledger_id": record_id,
                "source_record_id": f"{record_id}:stub",
                "source_block_index": int(record_id[3:]),
                "section_path": "Paper 0 > Axiom I > Minimal Lagrangian",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_axiom_i_minimal_lagrangian_specs(records)


def test_minimal_lagrangian_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_axiom_i_minimal_lagrangian_validation_spec(
        "axiom_i_minimal_lagrangian.l_min_operator_terms",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Axiom I Minimal Lagrangian Specs" in report
    assert "L_min" in report
    assert loaded["key"] == "axiom_i_minimal_lagrangian.l_min_operator_terms"
    assert "D_mu = partial_mu - i g A_mu" in loaded["source_formulae"]
    assert "Minimal Lagrangian" in render_report(bundle)
