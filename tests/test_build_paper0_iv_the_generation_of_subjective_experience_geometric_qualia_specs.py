# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 IV. The Generation of Subjective Experience (Geometric Qualia) builder tests
"""Tests for Paper 0 IV. The Generation of Subjective Experience (Geometric Qualia) source-accounting specs."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_paper0_iv_the_generation_of_subjective_experience_geometric_qualia_specs import (
    build_from_ledger,
    write_outputs,
)


def test_build_iv_the_generation_of_subjective_experience_geometric_qualia_specs_preserves_source_slice() -> (
    None
):
    bundle = build_from_ledger()
    assert bundle.summary["source_ledger_span"] == ["P0R06015", "P0R06022"]
    assert bundle.summary["source_record_count"] == 8
    assert bundle.summary["consumed_source_record_count"] == 8
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["unconsumed_source_ledger_ids"] == []
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["next_source_boundary"] == "P0R06023"
    assert bundle.summary["math_ids"] == []
    assert bundle.summary["image_ids"] == []
    assert bundle.summary["table_ids"] == []


def test_build_iv_the_generation_of_subjective_experience_geometric_qualia_specs_preserves_component_source_formulae() -> (
    None
):
    bundle = build_from_ledger()
    by_context = {spec.context_id: spec for spec in bundle.specs}
    assert set(by_context) == {
        "the_metric_tensor_gmu_encodes_the_valence_the_intrinsic_curvature_ricci",
        "the_ontology_of_experience",
        "formalisation_of_geometric_qualia",
        "the_connection_defines_the_flow_of_experience_stream_of_consciousness_vi",
        "iv_the_generation_of_subjective_experience_geometric_qualia",
    }
    for spec in bundle.specs:
        assert spec.source_formulae
        assert (
            spec.claim_boundary
            == "source-bounded iv the generation of subjective experience geometric qualia source-accounting bridge; not validation evidence"
        )
        assert spec.hardware_status == "source_methodology_no_experiment"


def test_write_iv_the_generation_of_subjective_experience_geometric_qualia_outputs(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(build_from_ledger(), output_dir=tmp_path, date_tag="2099-01-02")
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["summary"]["coverage_match"] is True
    assert (
        payload["summary"]["claim_boundary"]
        == "source-bounded iv the generation of subjective experience geometric qualia source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 " + "IV. The Generation of Subjective Experience (Geometric Qualia)" + " Specs"
        in report
    )
    assert "P0R06015 - P0R06022" in report
