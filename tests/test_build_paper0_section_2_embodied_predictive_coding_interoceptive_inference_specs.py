# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. Embodied Predictive Coding (Interoceptive Inference) builder tests
"""Tests for Paper 0 2. Embodied Predictive Coding (Interoceptive Inference) source-accounting specs."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_paper0_section_2_embodied_predictive_coding_interoceptive_inference_specs import (
    build_from_ledger,
    write_outputs,
)


def test_build_section_2_embodied_predictive_coding_interoceptive_inference_specs_preserves_source_slice() -> (
    None
):
    bundle = build_from_ledger()
    assert bundle.summary["source_ledger_span"] == ["P0R04967", "P0R04974"]
    assert bundle.summary["source_record_count"] == 8
    assert bundle.summary["consumed_source_record_count"] == 8
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["unconsumed_source_ledger_ids"] == []
    assert bundle.summary["spec_count"] == 2
    assert bundle.summary["next_source_boundary"] == "P0R04975"
    assert bundle.summary["math_ids"] == []
    assert bundle.summary["image_ids"] == ["IMG0118"]
    assert bundle.summary["table_ids"] == []


def test_build_section_2_embodied_predictive_coding_interoceptive_inference_specs_preserves_component_source_formulae() -> (
    None
):
    bundle = build_from_ledger()
    by_context = {spec.context_id: spec for spec in bundle.specs}
    assert set(by_context) == {
        "2_embodied_predictive_coding_interoceptive_inference",
        "3_distributed_criticality",
    }
    for spec in bundle.specs:
        assert spec.source_formulae
        assert (
            spec.claim_boundary
            == "source-bounded section 2 embodied predictive coding interoceptive inference source-accounting bridge; not validation evidence"
        )
        assert spec.hardware_status == "source_methodology_no_experiment"


def test_write_section_2_embodied_predictive_coding_interoceptive_inference_outputs(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(build_from_ledger(), output_dir=tmp_path, date_tag="2099-01-02")
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["summary"]["coverage_match"] is True
    assert (
        payload["summary"]["claim_boundary"]
        == "source-bounded section 2 embodied predictive coding interoceptive inference source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 " + "2. Embodied Predictive Coding (Interoceptive Inference)" + " Specs" in report
    )
    assert "P0R04967 - P0R04974" in report
