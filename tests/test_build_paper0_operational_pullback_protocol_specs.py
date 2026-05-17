# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 operational pullback protocol builder tests
"""Tests for Paper 0 operational-pullback protocol spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_operational_pullback_protocol_validation_spec,
)
from scripts.build_paper0_operational_pullback_protocol_specs import (
    SOURCE_LEDGER_IDS,
    build_from_ledger,
    build_operational_pullback_protocol_specs,
    render_report,
    write_outputs,
)


def test_operational_pullback_protocol_builder_preserves_full_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R01242", "P0R01271"]
    assert bundle.summary["source_record_count"] == 30
    assert bundle.summary["consumed_source_record_count"] == 30
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 6
    assert bundle.summary["math_ids"] == []
    assert bundle.summary["image_ids"] == []
    assert bundle.summary["table_ids"] == []
    assert bundle.summary["next_source_boundary"] == "P0R01272"
    assert [spec.key for spec in bundle.specs] == [
        "operational_pullback_protocol.section_and_protocol_boundary",
        "operational_pullback_protocol.statistical_bundle_and_fim",
        "operational_pullback_protocol.spacetime_pullback_and_normalisation",
        "operational_pullback_protocol.observable_sections_and_l4_l5_case",
        "operational_pullback_protocol.full_covariance_fim_strategy",
        "operational_pullback_protocol.eft_lorentz_locality_constraints",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_operational_pullback_protocol_builder_keeps_formulae_and_boundaries() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "Operational Pullback Protocol (Revision 11.00)"
        in specs["operational_pullback_protocol.section_and_protocol_boundary"].source_formulae
    )
    assert (
        "I_ij(theta) = E_p(y|x,theta)[partial_i log p partial_j log p]"
        in specs["operational_pullback_protocol.statistical_bundle_and_fim"].source_formulae
    )
    assert (
        "g_F_mu_nu(x) = (partial_mu theta^i(x)) I_ij(theta(x)) (partial_nu theta^j(x))"
        in specs[
            "operational_pullback_protocol.spacetime_pullback_and_normalisation"
        ].source_formulae
    )
    assert (
        "g_tilde_F^mu_nu(x) = Lambda_I^-2 (g_F^-1)^mu_nu"
        in specs[
            "operational_pullback_protocol.spacetime_pullback_and_normalisation"
        ].source_formulae
    )
    assert (
        "coding efficiency is proportional to det(I(theta))"
        in specs[
            "operational_pullback_protocol.observable_sections_and_l4_l5_case"
        ].source_formulae
    )
    assert (
        "NV-centre detection remains a prediction target, not validation evidence"
        in specs[
            "operational_pullback_protocol.observable_sections_and_l4_l5_case"
        ].source_formulae
    )
    assert (
        "full covariance Fisher information uses mean-gradient and covariance-gradient terms"
        in specs["operational_pullback_protocol.full_covariance_fim_strategy"].source_formulae
    )
    assert (
        "EFT interpretation, Lorentz invariance, locality, and causality are constraint boundaries"
        in specs["operational_pullback_protocol.eft_lorentz_locality_constraints"].source_formulae
    )


def test_operational_pullback_protocol_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R01264":
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
                "section_path": "Paper 0 > Operational Pullback Protocol",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_operational_pullback_protocol_specs(records)


def test_operational_pullback_protocol_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_operational_pullback_protocol_validation_spec(
        "operational_pullback_protocol.full_covariance_fim_strategy",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Operational Pullback Protocol Specs" in report
    assert loaded["key"] == "operational_pullback_protocol.full_covariance_fim_strategy"
    assert "full covariance Fisher information" in loaded["source_formulae"]
    assert "Operational Pullback Protocol" in render_report(bundle)
