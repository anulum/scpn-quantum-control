# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 SSB hierarchy genesis builder tests
"""Tests for Paper 0 SSB hierarchy-genesis spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import load_ssb_hierarchy_genesis_validation_spec
from scripts.build_paper0_ssb_hierarchy_genesis_specs import (
    SOURCE_LEDGER_IDS,
    build_from_ledger,
    build_ssb_hierarchy_genesis_specs,
    render_report,
    write_outputs,
)


def test_ssb_hierarchy_genesis_builder_preserves_full_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R01693", "P0R01713"]
    assert bundle.summary["source_record_count"] == 21
    assert bundle.summary["consumed_source_record_count"] == 21
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 3
    assert bundle.summary["math_ids"] == []
    assert bundle.summary["image_ids"] == []
    assert bundle.summary["table_ids"] == []
    assert bundle.summary["next_source_boundary"] == "P0R01714"
    assert [spec.key for spec in bundle.specs] == [
        "ssb_hierarchy_genesis.architecture_cascade",
        "ssb_hierarchy_genesis.conformal_torsion_seeding",
        "ssb_hierarchy_genesis.three_strike_explanation",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_ssb_hierarchy_genesis_builder_keeps_cascade_torsion_and_analogy_boundaries() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "15-layer SCPN architecture emerges through Sequential Spontaneous Symmetry Breaking events"
        in specs["ssb_hierarchy_genesis.architecture_cascade"].source_formulae
    )
    assert (
        "Primordial Break L15/L14 selects physical laws and constants"
        in specs["ssb_hierarchy_genesis.architecture_cascade"].source_formulae
    )
    assert (
        "V_eff(|Psi|, t -> 0+) = -mu^2(T_SEC) |Psi|^2 + lambda |Psi|^4"
        in specs["ssb_hierarchy_genesis.conformal_torsion_seeding"].source_formulae
    )
    assert (
        "universe is source-framed as an iterative learning system rather than random reset"
        in specs["ssb_hierarchy_genesis.conformal_torsion_seeding"].source_formulae
    )
    assert (
        "First Strike chooses laws of physics through Layer 15 guiding intelligence"
        in specs["ssb_hierarchy_genesis.three_strike_explanation"].source_formulae
    )
    assert (
        "15-layer universe is the remnant statue after cosmic sculpting"
        in specs["ssb_hierarchy_genesis.three_strike_explanation"].source_formulae
    )


def test_ssb_hierarchy_genesis_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R01704":
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
                "section_path": "Paper 0 > SSB Hierarchy Genesis",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_ssb_hierarchy_genesis_specs(records)


def test_ssb_hierarchy_genesis_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_ssb_hierarchy_genesis_validation_spec(
        "ssb_hierarchy_genesis.conformal_torsion_seeding",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 SSB Hierarchy Genesis Specs" in report
    assert loaded["key"] == "ssb_hierarchy_genesis.conformal_torsion_seeding"
    assert "T_SEC" in loaded["source_formulae"][1]
    assert "SSB Hierarchy Genesis" in render_report(bundle)
