# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 master Lagrangian intro builder tests
"""Tests for Paper 0 master-interaction-Lagrangian introduction promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_master_lagrangian_intro_validation_spec,
)
from scripts.build_paper0_master_lagrangian_intro_specs import (
    SOURCE_LEDGER_IDS,
    build_from_ledger,
    build_master_lagrangian_intro_specs,
    render_report,
    write_outputs,
)


def test_master_lagrangian_intro_builder_preserves_full_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R00987", "P0R01017"]
    assert bundle.summary["source_record_count"] == 31
    assert bundle.summary["consumed_source_record_count"] == 31
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 6
    assert bundle.summary["blank_record_count"] == 2
    assert bundle.summary["introduction_record_count"] == 13
    assert bundle.summary["meta_framework_record_count"] == 16
    assert bundle.summary["gauge_inference_record_count"] == 6
    assert bundle.summary["psis_coupling_record_count"] == 9
    assert bundle.summary["next_source_boundary"] == "P0R01018"
    assert [spec.key for spec in bundle.specs] == [
        "master_lagrangian_intro.part_ii_boundary",
        "master_lagrangian_intro.first_principles_framing",
        "master_lagrangian_intro.two_stream_derivation",
        "master_lagrangian_intro.explanatory_analogies",
        "master_lagrangian_intro.gauge_inference_integration",
        "master_lagrangian_intro.psis_coupling_gauge_interpretation",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_master_lagrangian_intro_builder_keeps_formulas_and_boundaries() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "Part II: The Physical Sector (Field Theory & Quantization)"
        in specs["master_lagrangian_intro.part_ii_boundary"].source_formulae
    )
    assert (
        "2.1 Master Interaction Lagrangian: Derivation from First Principles"
        in specs["master_lagrangian_intro.part_ii_boundary"].source_formulae
    )
    assert (
        "L_Int' is presented as predictive constrained explanatory foundation"
        in specs["master_lagrangian_intro.first_principles_framing"].source_formulae
    )
    assert (
        "local U(1) gauge invariance of a free complex scalar Psi-field"
        in specs["master_lagrangian_intro.two_stream_derivation"].source_formulae
    )
    assert (
        "spin-1 gauge field infoton A_mu"
        in specs["master_lagrangian_intro.two_stream_derivation"].source_formulae
    )
    assert (
        "FIM kinetic term for the infoton"
        in specs["master_lagrangian_intro.two_stream_derivation"].source_formulae
    )
    assert (
        "-xi R Psi*Psi non-minimal coupling"
        in specs["master_lagrangian_intro.two_stream_derivation"].source_formulae
    )
    assert (
        "P0R00988 and P0R01001 are blank records"
        in specs["master_lagrangian_intro.explanatory_analogies"].source_formulae
    )
    assert (
        "gauge invariance as prerequisite for inference"
        in specs["master_lagrangian_intro.gauge_inference_integration"].source_formulae
    )
    assert (
        "infoton as prediction-error signal"
        in specs["master_lagrangian_intro.gauge_inference_integration"].source_formulae
    )
    assert (
        "H_int = -lambda * Psis * sigma"
        in specs["master_lagrangian_intro.psis_coupling_gauge_interpretation"].source_formulae
    )
    assert (
        "J_mu = i(Psi* partial_mu Psi - Psi partial_mu Psi*)"
        in specs["master_lagrangian_intro.psis_coupling_gauge_interpretation"].source_formulae
    )
    assert (
        "lambda is the fundamental gauge coupling constant g"
        in specs["master_lagrangian_intro.psis_coupling_gauge_interpretation"].source_formulae
    )
    assert (
        "next boundary is P0R01018 gauge-principle derivation"
        in specs["master_lagrangian_intro.psis_coupling_gauge_interpretation"].source_formulae
    )


def test_master_lagrangian_intro_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R01013":
            continue
        records.append(
            {
                "ledger_id": record_id,
                "source_record_id": f"{record_id}:stub",
                "source_block_index": int(record_id[3:]),
                "section_path": "Paper 0 > Master Interaction Lagrangian",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_master_lagrangian_intro_specs(records)


def test_master_lagrangian_intro_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_master_lagrangian_intro_validation_spec(
        "master_lagrangian_intro.psis_coupling_gauge_interpretation",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Master Lagrangian Intro Specs" in report
    assert loaded["key"] == "master_lagrangian_intro.psis_coupling_gauge_interpretation"
    assert "H_int = -lambda * Psis * sigma" in loaded["source_formulae"]
    assert "Master Lagrangian Intro" in render_report(bundle)
