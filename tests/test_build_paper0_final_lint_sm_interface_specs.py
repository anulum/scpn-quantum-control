# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 final LInt SM interface builder tests
"""Tests for Paper 0 final LInt and SM-interface spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import load_final_lint_sm_interface_validation_spec
from scripts.build_paper0_final_lint_sm_interface_specs import (
    SOURCE_LEDGER_IDS,
    build_final_lint_sm_interface_specs,
    build_from_ledger,
    render_report,
    write_outputs,
)


def test_final_lint_sm_interface_builder_preserves_full_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R01510", "P0R01581"]
    assert bundle.summary["source_record_count"] == 72
    assert bundle.summary["consumed_source_record_count"] == 72
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 6
    assert bundle.summary["math_ids"] == ["EQ0023", "EQ0024", "EQ0025"]
    assert bundle.summary["image_ids"] == ["IMG0029", "IMG0030"]
    assert bundle.summary["table_ids"] == []
    assert bundle.summary["next_source_boundary"] == "P0R01582"
    assert [spec.key for spec in bundle.specs] == [
        "final_lint_sm_interface.final_lint_dual_clause",
        "final_lint_sm_interface.free_energy_and_h_int_mapping",
        "final_lint_sm_interface.foundational_physics_equations",
        "final_lint_sm_interface.standard_model_indirect_coupling",
        "final_lint_sm_interface.predictive_interface_mapping",
        "final_lint_sm_interface.downstream_sm_manifestations",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_final_lint_sm_interface_builder_keeps_equations_and_boundaries() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "L_Geometric = -xi R Psi*Psi"
        in specs["final_lint_sm_interface.final_lint_dual_clause"].source_formulae
    )
    assert (
        "H_int = -lambda * Psi_s * sigma"
        in specs["final_lint_sm_interface.free_energy_and_h_int_mapping"].source_formulae
    )
    assert (
        "L_Int = L_Geometric + L_Informational"
        in specs["final_lint_sm_interface.foundational_physics_equations"].source_formulae
    )
    assert (
        "Psi-field does not introduce new direct forces or particles that directly couple to SM fields"
        in specs["final_lint_sm_interface.standard_model_indirect_coupling"].source_formulae
    )
    assert (
        "prediction is realised as biasing quantum probabilities without violating quantum mechanics"
        in specs["final_lint_sm_interface.predictive_interface_mapping"].source_formulae
    )
    assert (
        "chiral weak-force and ALP-mediated links are dotted exploratory hypotheses"
        in specs["final_lint_sm_interface.downstream_sm_manifestations"].source_formulae
    )


def test_final_lint_sm_interface_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R01578":
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
                "section_path": "Paper 0 > Final LInt SM Interface",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_final_lint_sm_interface_specs(records)


def test_final_lint_sm_interface_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_final_lint_sm_interface_validation_spec(
        "final_lint_sm_interface.standard_model_indirect_coupling",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Final LInt SM Interface Specs" in report
    assert loaded["key"] == "final_lint_sm_interface.standard_model_indirect_coupling"
    assert "Indirect Coupling" in loaded["source_formulae"][0]
    assert "Final LInt SM Interface" in render_report(bundle)
