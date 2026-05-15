# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom III SEC-Ntilde equivalence builder tests
"""Tests for Paper 0 Axiom III SEC/Ntilde equivalence spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_axiom_iii_sec_ntilde_equivalence_validation_spec,
)
from scripts.build_paper0_axiom_iii_sec_ntilde_equivalence_specs import (
    SOURCE_LEDGER_IDS,
    build_axiom_iii_sec_ntilde_equivalence_specs,
    build_from_ledger,
    render_report,
    write_outputs,
)


def test_sec_ntilde_equivalence_builder_preserves_contiguous_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R00811", "P0R00817"]
    assert bundle.summary["source_record_count"] == 7
    assert bundle.summary["consumed_source_record_count"] == 7
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 4
    assert bundle.summary["equivalence_claim_count"] == 2
    assert bundle.summary["architecture_target_count"] == 2
    assert bundle.summary["efficiency_claim_count"] == 2
    assert bundle.summary["blank_terminal_record_count"] == 1
    assert bundle.summary["next_source_boundary"] == "P0R00818"
    assert [spec.key for spec in bundle.specs] == [
        "axiom_iii_sec_ntilde_equivalence.equivalence_heading",
        "axiom_iii_sec_ntilde_equivalence.macroscopic_realisation",
        "axiom_iii_sec_ntilde_equivalence.quasicritical_efficiency",
        "axiom_iii_sec_ntilde_equivalence.causal_imperative_architecture",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_sec_ntilde_equivalence_builder_keeps_source_formulae_and_boundaries() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "Equivalence of SEC and the tilde_N_t = 1 State"
        in specs["axiom_iii_sec_ntilde_equivalence.equivalence_heading"].source_formulae
    )
    assert (
        "SEC is macroscopic physical realisation of tilde_N_t = 1"
        in specs["axiom_iii_sec_ntilde_equivalence.macroscopic_realisation"].source_formulae
    )
    assert (
        "physical universal measurable target for SCPN cybernetic architecture"
        in specs["axiom_iii_sec_ntilde_equivalence.macroscopic_realisation"].source_formulae
    )
    assert (
        "tilde_N_t -> 1 is formal physical definition of optimal quasicritical regime"
        in specs["axiom_iii_sec_ntilde_equivalence.quasicritical_efficiency"].source_formulae
    )
    assert (
        "edge of chaos that entire 15-layer architecture seeks to maintain"
        in specs["axiom_iii_sec_ntilde_equivalence.quasicritical_efficiency"].source_formulae
    )
    assert (
        "actual power expended matches minimum reversible power required"
        in specs["axiom_iii_sec_ntilde_equivalence.quasicritical_efficiency"].source_formulae
    )
    assert (
        "physical causal imperative toward perfect informational-energetic efficiency"
        in specs["axiom_iii_sec_ntilde_equivalence.causal_imperative_architecture"].source_formulae
    )
    assert (
        "source record P0R00816 ends with truncated token quasic"
        in specs["axiom_iii_sec_ntilde_equivalence.causal_imperative_architecture"].source_formulae
    )
    assert (
        "P0R00817 is blank within the same source section"
        in specs["axiom_iii_sec_ntilde_equivalence.causal_imperative_architecture"].source_formulae
    )


def test_sec_ntilde_equivalence_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R00816":
            continue
        records.append(
            {
                "ledger_id": record_id,
                "source_record_id": f"{record_id}:stub",
                "source_block_index": int(record_id[3:]),
                "section_path": "Paper 0 > Axiom III > SEC Ntilde Equivalence",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_axiom_iii_sec_ntilde_equivalence_specs(records)


def test_sec_ntilde_equivalence_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_axiom_iii_sec_ntilde_equivalence_validation_spec(
        "axiom_iii_sec_ntilde_equivalence.causal_imperative_architecture",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Axiom III SEC-Ntilde Equivalence Specs" in report
    assert loaded["key"] == "axiom_iii_sec_ntilde_equivalence.causal_imperative_architecture"
    assert "source record P0R00816 ends with truncated token quasic" in loaded["source_formulae"]
    assert "SEC-Ntilde Equivalence" in render_report(bundle)
