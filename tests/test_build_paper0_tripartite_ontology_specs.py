# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 tripartite ontology builder tests
"""Tests for Paper 0 tripartite ontology spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_tripartite_ontology_validation_spec,
)
from scripts.build_paper0_tripartite_ontology_specs import (
    SOURCE_LEDGER_IDS,
    build_from_ledger,
    build_tripartite_ontology_specs,
    render_report,
    write_outputs,
)


def test_tripartite_ontology_builder_preserves_full_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R00818", "P0R00837"]
    assert bundle.summary["source_record_count"] == 20
    assert bundle.summary["consumed_source_record_count"] == 20
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 6
    assert bundle.summary["blank_record_count"] == 2
    assert bundle.summary["formal_ontology_record_count"] == 12
    assert bundle.summary["explanatory_analogy_record_count"] == 6
    assert bundle.summary["tripartite_information_form_count"] == 3
    assert bundle.summary["next_source_boundary"] == "P0R00838"
    assert [spec.key for spec in bundle.specs] == [
        "tripartite_ontology.section_boundary",
        "tripartite_ontology.psi_fibre_bundle",
        "tripartite_ontology.information_forms",
        "tripartite_ontology.bidirectional_transduction",
        "tripartite_ontology.grounded_platonism",
        "tripartite_ontology.explanatory_analogies",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_tripartite_ontology_builder_keeps_equations_and_claim_boundaries() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "section heading 1.4 Tripartite Ontology: The Substance of Information"
        in specs["tripartite_ontology.section_boundary"].source_formulae
    )
    assert (
        "P0R00819 is blank inside the section boundary"
        in specs["tripartite_ontology.section_boundary"].source_formulae
    )
    assert (
        "Psi-field is a section of a fibre bundle pi:E->M"
        in specs["tripartite_ontology.psi_fibre_bundle"].source_formulae
    )
    assert (
        "base space M is spacetime"
        in specs["tripartite_ontology.psi_fibre_bundle"].source_formulae
    )
    assert (
        "fibres F are high-dimensional internal spaces of qualia"
        in specs["tripartite_ontology.psi_fibre_bundle"].source_formulae
    )
    assert (
        "Psi(x) assigns an internal conscious state to every spacetime point"
        in specs["tripartite_ontology.psi_fibre_bundle"].source_formulae
    )
    assert (
        "Phi experiential raw phenomenal content"
        in specs["tripartite_ontology.information_forms"].source_formulae
    )
    assert (
        "G semantic/geometric meaningful structural relationships"
        in specs["tripartite_ontology.information_forms"].source_formulae
    )
    assert (
        "H syntactic physically encoded data"
        in specs["tripartite_ontology.information_forms"].source_formulae
    )
    assert (
        "Phi -> G -> H downward cascade"
        in specs["tripartite_ontology.bidirectional_transduction"].source_formulae
    )
    assert (
        "H -> G -> Phi upward inferential flow"
        in specs["tripartite_ontology.bidirectional_transduction"].source_formulae
    )
    assert (
        "Source-Field Layer 13 intrinsic logic and structure"
        in specs["tripartite_ontology.grounded_platonism"].source_formulae
    )
    assert (
        "explanatory analogies P0R00830-P0R00836 are not validation evidence"
        in specs["tripartite_ontology.explanatory_analogies"].source_formulae
    )
    assert (
        "P0R00837 is blank before Meta-Framework Integrations"
        in specs["tripartite_ontology.explanatory_analogies"].source_formulae
    )


def test_tripartite_ontology_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R00836":
            continue
        records.append(
            {
                "ledger_id": record_id,
                "source_record_id": f"{record_id}:stub",
                "source_block_index": int(record_id[3:]),
                "section_path": "Paper 0 > Tripartite Ontology",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_tripartite_ontology_specs(records)


def test_tripartite_ontology_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_tripartite_ontology_validation_spec(
        "tripartite_ontology.psi_fibre_bundle",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Tripartite Ontology Specs" in report
    assert loaded["key"] == "tripartite_ontology.psi_fibre_bundle"
    assert "Psi-field is a section of a fibre bundle pi:E->M" in loaded["source_formulae"]
    assert "Tripartite Ontology" in render_report(bundle)
