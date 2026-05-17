# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 experimental signatures builder tests
"""Tests for Paper 0 experimental-signatures search-strategy spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_experimental_signatures_search_strategies_validation_spec,
)
from scripts.build_paper0_experimental_signatures_search_strategies_specs import (
    SOURCE_LEDGER_IDS,
    build_experimental_signatures_search_strategies_specs,
    build_from_ledger,
    render_report,
    write_outputs,
)


def test_experimental_signatures_builder_preserves_full_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R01647", "P0R01654"]
    assert bundle.summary["source_record_count"] == 8
    assert bundle.summary["consumed_source_record_count"] == 8
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 4
    assert bundle.summary["math_ids"] == []
    assert bundle.summary["image_ids"] == []
    assert bundle.summary["table_ids"] == []
    assert bundle.summary["next_source_boundary"] == "P0R01655"
    assert [spec.key for spec in bundle.specs] == [
        "experimental_signatures_search_strategies.falsifiability_frame",
        "experimental_signatures_search_strategies.collider_channel",
        "experimental_signatures_search_strategies.cosmological_channel",
        "experimental_signatures_search_strategies.complementary_test_boundary",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_experimental_signatures_builder_keeps_search_channels_and_boundaries() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "two new particles are the massive vector infoton and massive scalar Psi-Higgs"
        in specs["experimental_signatures_search_strategies.falsifiability_frame"].source_formulae
    )
    assert (
        "h_SM -> h_Psi h_Psi"
        in specs["experimental_signatures_search_strategies.collider_channel"].source_formulae
    )
    assert (
        "CMS and ATLAS could search for excess events with invariant-mass signatures"
        in specs["experimental_signatures_search_strategies.collider_channel"].source_formulae
    )
    assert (
        "scalar boson clouds could form around spinning black holes by superradiance"
        in specs["experimental_signatures_search_strategies.cosmological_channel"].source_formulae
    )
    assert (
        "LISA, Einstein Telescope, and Cosmic Explorer are future or next-generation search instruments"
        in specs["experimental_signatures_search_strategies.cosmological_channel"].source_formulae
    )
    assert (
        "abstract claims are transformed into concrete falsifiable hypotheses"
        in specs[
            "experimental_signatures_search_strategies.complementary_test_boundary"
        ].source_formulae
    )


def test_experimental_signatures_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R01651":
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
                "section_path": "Paper 0 > Experimental Signatures and Search Strategies",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_experimental_signatures_search_strategies_specs(records)


def test_experimental_signatures_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_experimental_signatures_search_strategies_validation_spec(
        "experimental_signatures_search_strategies.collider_channel",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Experimental Signatures Search Strategies Specs" in report
    assert loaded["key"] == "experimental_signatures_search_strategies.collider_channel"
    assert "h_SM -> h_Psi h_Psi" in loaded["source_formulae"]
    assert "Experimental Signatures Search Strategies" in render_report(bundle)
