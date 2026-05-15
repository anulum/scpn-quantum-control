# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom I meta-coupling builder tests
"""Tests for Paper 0 Axiom I meta-framework and coupling spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_axiom_i_meta_coupling_validation_spec,
)
from scripts.build_paper0_axiom_i_meta_coupling_specs import (
    SOURCE_LEDGER_IDS,
    build_axiom_i_meta_coupling_specs,
    build_from_ledger,
    render_report,
    write_outputs,
)


def test_meta_coupling_builder_preserves_contiguous_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R00717", "P0R00732"]
    assert bundle.summary["source_record_count"] == 16
    assert bundle.summary["consumed_source_record_count"] == 16
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["interaction_component_count"] == 3
    assert bundle.summary["next_source_boundary"] == "P0R00733"
    assert [spec.key for spec in bundle.specs] == [
        "axiom_i_meta_coupling.predictive_coding_hardware",
        "axiom_i_meta_coupling.hint_component_justification",
        "axiom_i_meta_coupling.psis_complex_scalar_requirement",
        "axiom_i_meta_coupling.gauge_interaction_requirement",
        "axiom_i_meta_coupling.sigma_q_ball_requirement",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_meta_coupling_builder_keeps_coupling_and_falsification_labels() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "minimal algorithmic complexity"
        in specs["axiom_i_meta_coupling.predictive_coding_hardware"].source_formulae
    )
    assert (
        "phase theta carries beliefs or priors"
        in specs["axiom_i_meta_coupling.predictive_coding_hardware"].source_formulae
    )
    assert (
        "H_int = -lambda * Psi_s * sigma"
        in specs["axiom_i_meta_coupling.hint_component_justification"].source_formulae
    )
    assert (
        "Psi_s must be a complex scalar"
        in specs["axiom_i_meta_coupling.psis_complex_scalar_requirement"].source_formulae
    )
    assert (
        "gauge boson infoton mediates H_int"
        in specs["axiom_i_meta_coupling.gauge_interaction_requirement"].source_formulae
    )
    assert (
        "sigma must be a stable charge-supported soliton"
        in specs["axiom_i_meta_coupling.sigma_q_ball_requirement"].source_formulae
    )


def test_meta_coupling_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R00730":
            continue
        records.append(
            {
                "ledger_id": record_id,
                "source_record_id": f"{record_id}:stub",
                "source_block_index": int(record_id[3:]),
                "section_path": "Paper 0 > Axiom I > Psi-s Coupling",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_axiom_i_meta_coupling_specs(records)


def test_meta_coupling_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_axiom_i_meta_coupling_validation_spec(
        "axiom_i_meta_coupling.gauge_interaction_requirement",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Axiom I Meta-Coupling Specs" in report
    assert "H_int" in report
    assert loaded["key"] == "axiom_i_meta_coupling.gauge_interaction_requirement"
    assert "gauge boson infoton mediates H_int" in loaded["source_formulae"]
    assert "Meta-Coupling" in render_report(bundle)
