# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 category universal grammar builder tests
"""Tests for Paper 0 category/universal-grammar spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_category_universal_grammar_validation_spec,
)
from scripts.build_paper0_category_universal_grammar_specs import (
    SOURCE_LEDGER_IDS,
    build_category_universal_grammar_specs,
    build_from_ledger,
    render_report,
    write_outputs,
)


def test_category_universal_grammar_builder_preserves_full_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R00905", "P0R00986"]
    assert bundle.summary["source_record_count"] == 82
    assert bundle.summary["consumed_source_record_count"] == 82
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 8
    assert bundle.summary["blank_record_count"] == 3
    assert bundle.summary["image_record_count"] == 5
    assert bundle.summary["figure_caption_record_count"] == 11
    assert bundle.summary["formal_category_record_count"] == 27
    assert bundle.summary["meta_framework_record_count"] == 15
    assert bundle.summary["next_source_boundary"] == "P0R00987"
    assert [spec.key for spec in bundle.specs] == [
        "category_universal_grammar.section_boundary",
        "category_universal_grammar.category_objects_morphisms",
        "category_universal_grammar.functor_bridge",
        "category_universal_grammar.topos_kan_foundation",
        "category_universal_grammar.explanatory_analogies",
        "category_universal_grammar.meta_framework_integrations",
        "category_universal_grammar.formal_diagram_records",
        "category_universal_grammar.kan_extension_inference",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_category_universal_grammar_builder_keeps_equations_and_boundaries() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "1.5 The Universal Grammar: A Category-Theoretic Foundation"
        in specs["category_universal_grammar.section_boundary"].source_formulae
    )
    assert (
        "objects are 15 layers L1...L15"
        in specs["category_universal_grammar.category_objects_morphisms"].source_formulae
    )
    assert (
        "morphisms are projection maps f: Li -> Lj"
        in specs["category_universal_grammar.category_objects_morphisms"].source_formulae
    )
    assert (
        "composition (f o g)(x) = f(g(x))"
        in specs["category_universal_grammar.category_objects_morphisms"].source_formulae
    )
    assert (
        "F: Consciousness -> Physics"
        in specs["category_universal_grammar.functor_bridge"].source_formulae
    )
    assert (
        "G: Physics -> Consciousness"
        in specs["category_universal_grammar.functor_bridge"].source_formulae
    )
    assert "eta: F => G" in specs["category_universal_grammar.functor_bridge"].source_formulae
    assert (
        "Omega = {true, false, uncertain}"
        in specs["category_universal_grammar.topos_kan_foundation"].source_formulae
    )
    assert (
        "B^A represents all possible projections from layer A to layer B"
        in specs["category_universal_grammar.topos_kan_foundation"].source_formulae
    )
    assert (
        "P0R00915 is blank after explanatory analogy records"
        in specs["category_universal_grammar.explanatory_analogies"].source_formulae
    )
    assert (
        "category itself is formal architecture of hierarchical generative model"
        in specs["category_universal_grammar.meta_framework_integrations"].source_formulae
    )
    assert (
        "H_int = -lambda * Psis * sigma"
        in specs["category_universal_grammar.meta_framework_integrations"].source_formulae
    )
    assert (
        "image placeholders and figure captions are not validation evidence"
        in specs["category_universal_grammar.formal_diagram_records"].source_formulae
    )
    assert (
        "Lan_F(G) = best approximation from below"
        in specs["category_universal_grammar.kan_extension_inference"].source_formulae
    )
    assert (
        "Psi_inferred = Lan_physical(Psi_true)"
        in specs["category_universal_grammar.kan_extension_inference"].source_formulae
    )
    assert (
        "next boundary is P0R00987 Part II Physical Sector"
        in specs["category_universal_grammar.kan_extension_inference"].source_formulae
    )


def test_category_universal_grammar_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R00984":
            continue
        records.append(
            {
                "ledger_id": record_id,
                "source_record_id": f"{record_id}:stub",
                "source_block_index": int(record_id[3:]),
                "section_path": "Paper 0 > Category Universal Grammar",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_category_universal_grammar_specs(records)


def test_category_universal_grammar_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_category_universal_grammar_validation_spec(
        "category_universal_grammar.kan_extension_inference",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Category Universal Grammar Specs" in report
    assert loaded["key"] == "category_universal_grammar.kan_extension_inference"
    assert "Psi_inferred = Lan_physical(Psi_true)" in loaded["source_formulae"]
    assert "Category Universal Grammar" in render_report(bundle)
