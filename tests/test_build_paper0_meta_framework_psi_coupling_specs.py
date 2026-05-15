# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 meta-framework Psi coupling builder tests
"""Tests for Paper 0 meta-framework/Psi-coupling spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_meta_framework_psi_coupling_validation_spec,
)
from scripts.build_paper0_meta_framework_psi_coupling_specs import (
    SOURCE_LEDGER_IDS,
    build_from_ledger,
    build_meta_framework_psi_coupling_specs,
    render_report,
    write_outputs,
)


def test_meta_framework_psi_coupling_builder_preserves_full_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R00838", "P0R00904"]
    assert bundle.summary["source_record_count"] == 67
    assert bundle.summary["consumed_source_record_count"] == 67
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 7
    assert bundle.summary["predictive_coding_record_count"] == 14
    assert bundle.summary["psi_coupling_record_count"] == 25
    assert bundle.summary["formal_restatement_record_count"] == 22
    assert bundle.summary["image_or_figure_record_count"] == 6
    assert bundle.summary["blank_record_count"] == 2
    assert bundle.summary["next_source_boundary"] == "P0R00905"
    assert [spec.key for spec in bundle.specs] == [
        "meta_framework_psi_coupling.meta_framework_boundary",
        "meta_framework_psi_coupling.predictive_coding_loop",
        "meta_framework_psi_coupling.psi_interaction_hamiltonian",
        "meta_framework_psi_coupling.coupling_projection",
        "meta_framework_psi_coupling.formal_ontology_restatement",
        "meta_framework_psi_coupling.figure_and_image_records",
        "meta_framework_psi_coupling.repeated_ontology_block",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_meta_framework_psi_coupling_builder_keeps_formulae_and_boundaries() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "Meta-Framework Integrations"
        in specs["meta_framework_psi_coupling.meta_framework_boundary"].source_formulae
    )
    assert (
        "fibre bundle is state space of beliefs"
        in specs["meta_framework_psi_coupling.predictive_coding_loop"].source_formulae
    )
    assert (
        "tripartite ontology is active inference loop"
        in specs["meta_framework_psi_coupling.predictive_coding_loop"].source_formulae
    )
    assert (
        "H_int = -lambda * Psis * sigma"
        in specs["meta_framework_psi_coupling.psi_interaction_hamiltonian"].source_formulae
    )
    assert (
        "Psis is a section of the universal fibre bundle"
        in specs["meta_framework_psi_coupling.psi_interaction_hamiltonian"].source_formulae
    )
    assert (
        "sigma is the physical system at a point in base space M"
        in specs["meta_framework_psi_coupling.psi_interaction_hamiltonian"].source_formulae
    )
    assert (
        "H_int realises projection from total space onto a specific fibre"
        in specs["meta_framework_psi_coupling.coupling_projection"].source_formulae
    )
    assert (
        "geometric content G is transduced into syntactic state H of sigma"
        in specs["meta_framework_psi_coupling.coupling_projection"].source_formulae
    )
    assert (
        "Psi(x) in E"
        in specs["meta_framework_psi_coupling.formal_ontology_restatement"].source_formulae
    )
    assert (
        "pi:E->M with M=Spacetime"
        in specs["meta_framework_psi_coupling.formal_ontology_restatement"].source_formulae
    )
    assert (
        "image placeholders are source records, not evidence"
        in specs["meta_framework_psi_coupling.figure_and_image_records"].source_formulae
    )
    assert (
        "P0R00875 and P0R00897 are blank records"
        in specs["meta_framework_psi_coupling.repeated_ontology_block"].source_formulae
    )
    assert (
        "next boundary is P0R00905 Section 1.5 Universal Grammar"
        in specs["meta_framework_psi_coupling.repeated_ontology_block"].source_formulae
    )


def test_meta_framework_psi_coupling_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R00847":
            continue
        records.append(
            {
                "ledger_id": record_id,
                "source_record_id": f"{record_id}:stub",
                "source_block_index": int(record_id[3:]),
                "section_path": "Paper 0 > Meta Framework Psi Coupling",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_meta_framework_psi_coupling_specs(records)


def test_meta_framework_psi_coupling_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_meta_framework_psi_coupling_validation_spec(
        "meta_framework_psi_coupling.psi_interaction_hamiltonian",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Meta-Framework Psi Coupling Specs" in report
    assert loaded["key"] == "meta_framework_psi_coupling.psi_interaction_hamiltonian"
    assert "H_int = -lambda * Psis * sigma" in loaded["source_formulae"]
    assert "Meta-Framework Psi Coupling" in render_report(bundle)
