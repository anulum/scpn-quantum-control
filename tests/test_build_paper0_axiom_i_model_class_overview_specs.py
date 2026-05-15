# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom I model-class overview builder tests
"""Tests for Paper 0 Axiom I model-class overview spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_axiom_i_model_class_overview_validation_spec,
)
from scripts.build_paper0_axiom_i_model_class_overview_specs import (
    SOURCE_LEDGER_IDS,
    build_axiom_i_model_class_overview_specs,
    build_from_ledger,
    render_report,
    write_outputs,
)


def test_model_class_overview_builder_preserves_contiguous_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R00703", "P0R00716"]
    assert bundle.summary["source_record_count"] == 14
    assert bundle.summary["consumed_source_record_count"] == 14
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 4
    assert bundle.summary["selection_criterion_count"] == 3
    assert bundle.summary["blank_separator_count"] == 1
    assert bundle.summary["next_source_boundary"] == "P0R00717"
    assert [spec.key for spec in bundle.specs] == [
        "axiom_i_model_class_overview.three_criteria",
        "axiom_i_model_class_overview.complex_scalar_local_u1_ssb",
        "axiom_i_model_class_overview.rejected_alternatives_and_predictions",
        "axiom_i_model_class_overview.pedagogical_three_job_restatement",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_model_class_overview_builder_keeps_selection_and_rejection_labels() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "P0R00705:irreducible_spin0_degree"
        in specs["axiom_i_model_class_overview.three_criteria"].source_equation_ids
    )
    assert (
        "irreducible spin-0 degree of freedom"
        in specs["axiom_i_model_class_overview.three_criteria"].source_formulae
    )
    assert (
        "complex scalar field with local U(1) gauge symmetry"
        in specs["axiom_i_model_class_overview.complex_scalar_local_u1_ssb"].source_formulae
    )
    assert (
        "Mexican hat potential for SSB"
        in specs["axiom_i_model_class_overview.complex_scalar_local_u1_ssb"].source_formulae
    )
    assert (
        "real scalar field lacks phase"
        in specs[
            "axiom_i_model_class_overview.rejected_alternatives_and_predictions"
        ].source_formulae
    )
    assert (
        "concrete laboratory predictions"
        in specs["axiom_i_model_class_overview.pedagogical_three_job_restatement"].source_formulae
    )


def test_model_class_overview_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R00708":
            continue
        records.append(
            {
                "ledger_id": record_id,
                "source_record_id": f"{record_id}:stub",
                "source_block_index": int(record_id[3:]),
                "section_path": "Paper 0 > Axiom I > Model-Class Justification",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_axiom_i_model_class_overview_specs(records)


def test_model_class_overview_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_axiom_i_model_class_overview_validation_spec(
        "axiom_i_model_class_overview.complex_scalar_local_u1_ssb",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Axiom I Model-Class Overview Specs" in report
    assert "local U(1)" in report
    assert loaded["key"] == "axiom_i_model_class_overview.complex_scalar_local_u1_ssb"
    assert "Mexican hat potential for SSB" in loaded["source_formulae"]
    assert "Model-Class Overview" in render_report(bundle)
