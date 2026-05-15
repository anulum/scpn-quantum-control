# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom II opening builder tests
"""Tests for Paper 0 Axiom II opening and interaction-axiom spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_axiom_ii_opening_validation_spec,
)
from scripts.build_paper0_axiom_ii_opening_specs import (
    SOURCE_LEDGER_IDS,
    build_axiom_ii_opening_specs,
    build_from_ledger,
    render_report,
    write_outputs,
)


def test_axiom_ii_opening_builder_preserves_contiguous_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R00761", "P0R00769"]
    assert bundle.summary["source_record_count"] == 9
    assert bundle.summary["consumed_source_record_count"] == 9
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 4
    assert bundle.summary["heading_record_count"] == 4
    assert bundle.summary["axiom_statement_count"] == 1
    assert bundle.summary["falsifiability_boundary_count"] == 1
    assert bundle.summary["next_source_boundary"] == "P0R00770"
    assert [spec.key for spec in bundle.specs] == [
        "axiom_ii_opening.section_headings",
        "axiom_ii_opening.source_material",
        "axiom_ii_opening.ontology_to_dynamics",
        "axiom_ii_opening.interaction_axiom",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_axiom_ii_opening_builder_keeps_source_formulae_and_boundaries() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "Axiom II: The Language of Information Geometry"
        in specs["axiom_ii_opening.section_headings"].source_formulae
    )
    assert (
        "The Fisher Information Metric (FIM) as the Solution"
        in specs["axiom_ii_opening.section_headings"].source_formulae
    )
    assert (
        "all fundamental interactions are informational transactions"
        in specs["axiom_ii_opening.source_material"].source_formulae
    )
    assert "ontology to dynamics" in specs["axiom_ii_opening.ontology_to_dynamics"].source_formulae
    assert (
        "single universal grammar for all fundamental forces"
        in specs["axiom_ii_opening.ontology_to_dynamics"].source_formulae
    )
    assert (
        "All fundamental interactions are informational and geometric"
        in specs["axiom_ii_opening.interaction_axiom"].source_formulae
    )
    assert (
        "falsifiable physical hypothesis"
        in specs["axiom_ii_opening.interaction_axiom"].source_formulae
    )
    assert (
        "system informational structure precedes coupling analysis"
        in specs["axiom_ii_opening.interaction_axiom"].source_formulae
    )


def test_axiom_ii_opening_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R00768":
            continue
        records.append(
            {
                "ledger_id": record_id,
                "source_record_id": f"{record_id}:stub",
                "source_block_index": int(record_id[3:]),
                "section_path": "Paper 0 > Axiom II > Opening",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_axiom_ii_opening_specs(records)


def test_axiom_ii_opening_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_axiom_ii_opening_validation_spec(
        "axiom_ii_opening.interaction_axiom",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Axiom II Opening Specs" in report
    assert loaded["key"] == "axiom_ii_opening.interaction_axiom"
    assert "falsifiable physical hypothesis" in loaded["source_formulae"]
    assert "Axiom II Opening" in render_report(bundle)
