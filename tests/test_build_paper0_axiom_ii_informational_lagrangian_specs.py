# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom II informational Lagrangian builder tests
"""Tests for Paper 0 Axiom II informational-Lagrangian spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_axiom_ii_informational_lagrangian_validation_spec,
)
from scripts.build_paper0_axiom_ii_informational_lagrangian_specs import (
    SOURCE_LEDGER_IDS,
    build_axiom_ii_informational_lagrangian_specs,
    build_from_ledger,
    render_report,
    write_outputs,
)


def test_informational_lagrangian_builder_preserves_contiguous_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R00782", "P0R00790"]
    assert bundle.summary["source_record_count"] == 9
    assert bundle.summary["consumed_source_record_count"] == 9
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 4
    assert bundle.summary["lagrangian_heading_count"] == 1
    assert bundle.summary["standard_gauge_baseline_count"] == 3
    assert bundle.summary["scpn_gauge_count"] == 3
    assert bundle.summary["pullback_protocol_count"] == 1
    assert bundle.summary["next_source_boundary"] == "P0R00791"
    assert [spec.key for spec in bundle.specs] == [
        "axiom_ii_informational_lagrangian.kinetic_term_modification",
        "axiom_ii_informational_lagrangian.standard_gauge_baseline",
        "axiom_ii_informational_lagrangian.scpn_gauge_lagrangian",
        "axiom_ii_informational_lagrangian.operational_pullback_protocol",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_informational_lagrangian_builder_keeps_equations_and_boundaries() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "kinetic term of the infoton gauge field A_mu"
        in specs["axiom_ii_informational_lagrangian.kinetic_term_modification"].source_formulae
    )
    assert (
        "replace spacetime metric with pulled-back information metric"
        in specs["axiom_ii_informational_lagrangian.kinetic_term_modification"].source_formulae
    )
    assert (
        "L_gauge = -1/4 g^{mu alpha} g^{nu beta} F_{mu nu} F_{alpha beta}"
        in specs["axiom_ii_informational_lagrangian.standard_gauge_baseline"].source_formulae
    )
    assert (
        "dynamics governed by spacetime geometry"
        in specs["axiom_ii_informational_lagrangian.standard_gauge_baseline"].source_formulae
    )
    assert (
        "L_gauge = -1/4 tilde_g_F^{mu alpha} tilde_g_F^{nu beta} F_{mu nu} F_{alpha beta}"
        in specs["axiom_ii_informational_lagrangian.scpn_gauge_lagrangian"].source_formulae
    )
    assert (
        "dynamics governed by informational geometry"
        in specs["axiom_ii_informational_lagrangian.scpn_gauge_lagrangian"].source_formulae
    )
    assert (
        "tilde_g_F is the Fisher Information Metric pulled back from the abstract statistical manifold"
        in specs["axiom_ii_informational_lagrangian.operational_pullback_protocol"].source_formulae
    )
    assert (
        "Operational Pullback Protocol is detailed in Chapter 6"
        in specs["axiom_ii_informational_lagrangian.operational_pullback_protocol"].source_formulae
    )
    assert (
        "Axiom II a falsifiable predictive hypothesis"
        in specs["axiom_ii_informational_lagrangian.operational_pullback_protocol"].source_formulae
    )


def test_informational_lagrangian_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R00788":
            continue
        records.append(
            {
                "ledger_id": record_id,
                "source_record_id": f"{record_id}:stub",
                "source_block_index": int(record_id[3:]),
                "section_path": "Paper 0 > Axiom II > Informational Lagrangian",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_axiom_ii_informational_lagrangian_specs(records)


def test_informational_lagrangian_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_axiom_ii_informational_lagrangian_validation_spec(
        "axiom_ii_informational_lagrangian.scpn_gauge_lagrangian",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Axiom II Informational Lagrangian Specs" in report
    assert loaded["key"] == "axiom_ii_informational_lagrangian.scpn_gauge_lagrangian"
    assert (
        "L_gauge = -1/4 tilde_g_F^{mu alpha} tilde_g_F^{nu beta} F_{mu nu} F_{alpha beta}"
        in loaded["source_formulae"]
    )
    assert "Informational Lagrangian" in render_report(bundle)
