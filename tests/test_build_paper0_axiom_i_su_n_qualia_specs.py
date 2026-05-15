# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom I SU(N) qualia builder tests
"""Tests for Paper 0 Axiom I SU(N) qualia-confinement spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_axiom_i_su_n_qualia_validation_spec,
)
from scripts.build_paper0_axiom_i_su_n_qualia_specs import (
    SOURCE_LEDGER_IDS,
    build_axiom_i_su_n_qualia_specs,
    build_from_ledger,
    render_report,
    write_outputs,
)


def test_su_n_qualia_builder_preserves_contiguous_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R00757", "P0R00760"]
    assert bundle.summary["source_record_count"] == 4
    assert bundle.summary["consumed_source_record_count"] == 4
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 3
    assert bundle.summary["gauge_boson_formula_count"] == 1
    assert bundle.summary["confinement_formula_count"] == 1
    assert bundle.summary["blank_separator_count"] == 1
    assert bundle.summary["next_source_boundary"] == "P0R00761"
    assert [spec.key for spec in bundle.specs] == [
        "axiom_i_su_n_qualia.group_extension",
        "axiom_i_su_n_qualia.confinement_hypothesis",
        "axiom_i_su_n_qualia.macroscopic_colored_state",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_su_n_qualia_builder_keeps_source_formulae_and_boundaries() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "SU(N) gauge group for N primary qualic dimensions"
        in specs["axiom_i_su_n_qualia.group_extension"].source_formulae
    )
    assert "N^2-1 info-gluons" in specs["axiom_i_su_n_qualia.group_extension"].source_formulae
    assert (
        "Qualia Confinement infrared slavery"
        in specs["axiom_i_su_n_qualia.confinement_hypothesis"].source_formulae
    )
    assert (
        "V(r) approx sigma r"
        in specs["axiom_i_su_n_qualia.confinement_hypothesis"].source_formulae
    )
    assert (
        "irreducible tensor product of confined qualia charges"
        in specs["axiom_i_su_n_qualia.macroscopic_colored_state"].source_formulae
    )
    assert (
        "Betti numbers beta_k of the Consciousness Manifold"
        in specs["axiom_i_su_n_qualia.macroscopic_colored_state"].source_formulae
    )


def test_su_n_qualia_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R00758":
            continue
        records.append(
            {
                "ledger_id": record_id,
                "source_record_id": f"{record_id}:stub",
                "source_block_index": int(record_id[3:]),
                "section_path": "Paper 0 > Axiom I > SU(N) Qualia Confinement",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_axiom_i_su_n_qualia_specs(records)


def test_su_n_qualia_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_axiom_i_su_n_qualia_validation_spec(
        "axiom_i_su_n_qualia.confinement_hypothesis",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Axiom I SU(N) Qualia Confinement Specs" in report
    assert loaded["key"] == "axiom_i_su_n_qualia.confinement_hypothesis"
    assert "V(r) approx sigma r" in loaded["source_formulae"]
    assert "SU(N) Qualia" in render_report(bundle)
