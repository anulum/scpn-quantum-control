# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Aqueous Substrate (Domain I Interface) builder tests
"""Tests for Paper 0 The Aqueous Substrate (Domain I Interface) source-accounting specs."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_paper0_the_aqueous_substrate_domain_i_interface_specs import (
    build_from_ledger,
    write_outputs,
)


def test_build_the_aqueous_substrate_domain_i_interface_specs_preserves_source_slice() -> None:
    bundle = build_from_ledger()
    assert bundle.summary["source_ledger_span"] == ["P0R05331", "P0R05346"]
    assert bundle.summary["source_record_count"] == 16
    assert bundle.summary["consumed_source_record_count"] == 16
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["unconsumed_source_ledger_ids"] == []
    assert bundle.summary["spec_count"] == 4
    assert bundle.summary["next_source_boundary"] == "P0R05347"
    assert bundle.summary["math_ids"] == ["EQ0103"]
    assert bundle.summary["image_ids"] == []
    assert bundle.summary["table_ids"] == []


def test_build_the_aqueous_substrate_domain_i_interface_specs_preserves_component_source_formulae() -> (
    None
):
    bundle = build_from_ledger()
    by_context = {spec.context_id: spec for spec in bundle.specs}
    assert set(by_context) == {
        "the_genesis_of_life_abiogenesis_as_a_guided_phase_transition",
        "coherence_domains_cds_predicted_by_qed_interfacial_water_forms_cds_where",
        "integration_in_l1_cds_shield_microtubule_qubits_and_support_frhlich_cond",
        "the_aqueous_substrate_domain_i_interface",
    }
    for spec in bundle.specs:
        assert spec.source_formulae
        assert (
            spec.claim_boundary
            == "source-bounded the aqueous substrate domain i interface source-accounting bridge; not validation evidence"
        )
        assert spec.hardware_status == "source_methodology_no_experiment"


def test_write_the_aqueous_substrate_domain_i_interface_outputs(tmp_path: Path) -> None:
    outputs = write_outputs(build_from_ledger(), output_dir=tmp_path, date_tag="2099-01-02")
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["summary"]["coverage_match"] is True
    assert (
        payload["summary"]["claim_boundary"]
        == "source-bounded the aqueous substrate domain i interface source-accounting bridge; not validation evidence"
    )
    assert "Paper 0 " + "The Aqueous Substrate (Domain I Interface)" + " Specs" in report
    assert "P0R05331 - P0R05346" in report
