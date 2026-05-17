# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 LHC search roadmap builder tests
"""Tests for Paper 0 LHC search-strategy roadmap spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_lhc_search_strategy_roadmap_validation_spec,
)
from scripts.build_paper0_lhc_search_strategy_roadmap_specs import (
    SOURCE_LEDGER_IDS,
    build_from_ledger,
    build_lhc_search_strategy_roadmap_specs,
    render_report,
    write_outputs,
)


def test_lhc_search_strategy_roadmap_builder_preserves_full_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R01684", "P0R01692"]
    assert bundle.summary["source_record_count"] == 9
    assert bundle.summary["consumed_source_record_count"] == 9
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 3
    assert bundle.summary["math_ids"] == []
    assert bundle.summary["image_ids"] == []
    assert bundle.summary["table_ids"] == ["TBL003"]
    assert bundle.summary["next_source_boundary"] == "P0R01693"
    assert [spec.key for spec in bundle.specs] == [
        "lhc_search_strategy_roadmap.search_signature_overview",
        "lhc_search_strategy_roadmap.table_roadmap",
        "lhc_search_strategy_roadmap.ssb_cascade_transition",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_lhc_search_strategy_roadmap_builder_keeps_channels_table_and_transition() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "exotic Higgs decays include h_SM -> h_Psi h_Psi when m_hPsi < 125 GeV"
        in specs["lhc_search_strategy_roadmap.search_signature_overview"].source_formulae
    )
    assert (
        "resonant production includes pp -> h_Psi -> ZZ and pp -> h_Psi -> WW"
        in specs["lhc_search_strategy_roadmap.search_signature_overview"].source_formulae
    )
    assert (
        "TBL003 is the source table for the proposed experimental search parameters"
        in specs["lhc_search_strategy_roadmap.table_roadmap"].source_formulae
    )
    assert (
        "2.4 The SSB Cascade: Origin of Mass & The Solitonic Self"
        in specs["lhc_search_strategy_roadmap.ssb_cascade_transition"].source_formulae
    )


def test_lhc_search_strategy_roadmap_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R01689":
            continue
        records.append(
            {
                "ledger_id": record_id,
                "source_record_id": f"{record_id}:stub",
                "source_block_index": int(record_id[3:]),
                "canonical_category": "context",
                "block_type": "Para",
                "math_ids": [],
                "image_ids": [],
                "table_id": None,
                "section_path": "Paper 0 > LHC Search Strategy Roadmap",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_lhc_search_strategy_roadmap_specs(records)


def test_lhc_search_strategy_roadmap_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_lhc_search_strategy_roadmap_validation_spec(
        "lhc_search_strategy_roadmap.table_roadmap",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert payload["summary"]["table_ids"] == ["TBL003"]
    assert "Paper 0 LHC Search Strategy Roadmap Specs" in report
    assert loaded["key"] == "lhc_search_strategy_roadmap.table_roadmap"
    assert "TBL003" in loaded["source_formulae"][1]
    assert "LHC Search Strategy Roadmap" in render_report(bundle)
