# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Lorentz EFT resolution builder tests
"""Tests for Paper 0 Lorentz-covariance/EFT-resolution spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import load_lorentz_eft_resolution_validation_spec
from scripts.build_paper0_lorentz_eft_resolution_specs import (
    SOURCE_LEDGER_IDS,
    build_from_ledger,
    build_lorentz_eft_resolution_specs,
    render_report,
    write_outputs,
)


def test_lorentz_eft_resolution_builder_preserves_full_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R01078", "P0R01102"]
    assert bundle.summary["source_record_count"] == 25
    assert bundle.summary["consumed_source_record_count"] == 25
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["blank_record_count"] == 1
    assert bundle.summary["lorentz_tension_record_count"] == 4
    assert bundle.summary["fundamental_action_record_count"] == 5
    assert bundle.summary["biological_medium_record_count"] == 7
    assert bundle.summary["consistency_record_count"] == 4
    assert bundle.summary["ghost_action_record_count"] == 5
    assert bundle.summary["next_source_boundary"] == "P0R01103"
    assert [spec.key for spec in bundle.specs] == [
        "lorentz_eft_resolution.boundary_and_tension",
        "lorentz_eft_resolution.fundamental_lorentz_invariant_action",
        "lorentz_eft_resolution.biological_medium_effective_metric",
        "lorentz_eft_resolution.consistency_implications",
        "lorentz_eft_resolution.ghost_action_boundary",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_lorentz_eft_resolution_builder_keeps_equations_and_boundaries() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "Formal Resolution of Lorentz Covariance: The FIM as an Emergent Effective Metric"
        in specs["lorentz_eft_resolution.boundary_and_tension"].source_formulae
    )
    assert (
        "naive replacement by pulled-back FIM explicitly breaks local Lorentz invariance"
        in specs["lorentz_eft_resolution.boundary_and_tension"].source_formulae
    )
    assert (
        "FIM acts as emergent dielectric-like tensor in EFT"
        in specs["lorentz_eft_resolution.boundary_and_tension"].source_formulae
    )
    assert (
        "F_mu_nu = partial_mu A_nu - partial_nu A_mu"
        in specs["lorentz_eft_resolution.fundamental_lorentz_invariant_action"].source_formulae
    )
    assert (
        "Lambda_I suppresses the higher-dimension informational operator"
        in specs["lorentz_eft_resolution.fundamental_lorentz_invariant_action"].source_formulae
    )
    assert (
        "L_gauge = -1/4 F_mu_nu F_alpha_beta(eta^mu_alpha eta^nu_beta - c/Lambda_I^4 gF^mu_alpha gF^nu_beta)"
        in specs["lorentz_eft_resolution.fundamental_lorentz_invariant_action"].source_formulae
    )
    assert (
        "g_eff^mu_alpha = eta^mu_alpha - c/(2 Lambda_I^2) gF^mu_alpha"
        in specs["lorentz_eft_resolution.biological_medium_effective_metric"].source_formulae
    )
    assert (
        "L_eff = -1/4 g_eff^mu_alpha g_eff^nu_beta F_mu_nu F_alpha_beta"
        in specs["lorentz_eft_resolution.biological_medium_effective_metric"].source_formulae
    )
    assert (
        "fundamental L_gauge is a true Lorentz scalar"
        in specs["lorentz_eft_resolution.consistency_implications"].source_formulae
    )
    assert (
        "Ward-Takahashi identities hold"
        in specs["lorentz_eft_resolution.consistency_implications"].source_formulae
    )
    assert (
        "L_ghost = cbar [gF^mu_nu partial_mu(partial_nu + i g [A_nu, dot])] c"
        in specs["lorentz_eft_resolution.ghost_action_boundary"].source_formulae
    )
    assert (
        "P0R01102 is a structural separator"
        in specs["lorentz_eft_resolution.ghost_action_boundary"].source_formulae
    )
    assert (
        "next boundary is P0R01103 Non-Abelian qualia field"
        in specs["lorentz_eft_resolution.ghost_action_boundary"].source_formulae
    )


def test_lorentz_eft_resolution_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R01085":
            continue
        records.append(
            {
                "ledger_id": record_id,
                "source_record_id": f"{record_id}:stub",
                "source_block_index": int(record_id[3:]),
                "section_path": "Paper 0 > Lorentz EFT Resolution",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_lorentz_eft_resolution_specs(records)


def test_lorentz_eft_resolution_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_lorentz_eft_resolution_validation_spec(
        "lorentz_eft_resolution.fundamental_lorentz_invariant_action",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Lorentz EFT Resolution Specs" in report
    assert loaded["key"] == "lorentz_eft_resolution.fundamental_lorentz_invariant_action"
    assert "F_mu_nu = partial_mu A_nu - partial_nu A_mu" in loaded["source_formulae"]
    assert "Lorentz EFT Resolution" in render_report(bundle)
