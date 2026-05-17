# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Part II: The Physical Sector (Field Theory & Quantization) > 2.4 The SSB Cascade: Origin of Mass & The Solitonic Self > The "Self" as a Soliton: Emergence of Localised Consciousness (Layer 5) > The Definition of the Self: The Triadic Solution builder tests
"""Tests for Paper 0 Part II: The Physical Sector (Field Theory & Quantization) > 2.4 The SSB Cascade: Origin of Mass & The Solitonic Self > The "Self" as a Soliton: Emergence of Localised Consciousness (Layer 5) > The Definition of the Self: The Triadic Solution source-accounting specs."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_paper0_part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad_specs import (
    build_from_ledger,
    write_outputs,
)


def test_build_part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad_specs_preserves_source_slice() -> (
    None
):
    bundle = build_from_ledger()
    assert bundle.summary["source_ledger_span"] == ["P0R01895", "P0R01958"]
    assert bundle.summary["source_record_count"] == 64
    assert bundle.summary["consumed_source_record_count"] == 64
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["unconsumed_source_ledger_ids"] == []
    assert bundle.summary["spec_count"] == 1
    assert bundle.summary["next_source_boundary"] == "P0R01959"
    assert bundle.summary["math_ids"] == []
    assert bundle.summary["image_ids"] == []
    assert bundle.summary["table_ids"] == []


def test_build_part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad_specs_preserves_component_source_formulae() -> (
    None
):
    bundle = build_from_ledger()
    by_context = {spec.context_id: spec for spec in bundle.specs}
    assert set(by_context) == {
        "part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad"
    }
    for spec in bundle.specs:
        assert spec.source_formulae
        assert (
            spec.claim_boundary
            == "source-bounded part ii the physical sector field theory quantization 2 4 the ssb cascad source-accounting bridge; not validation evidence"
        )
        assert spec.hardware_status == "source_methodology_no_experiment"


def test_write_part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad_outputs(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(build_from_ledger(), output_dir=tmp_path, date_tag="2099-01-02")
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["summary"]["coverage_match"] is True
    assert (
        payload["summary"]["claim_boundary"]
        == "source-bounded part ii the physical sector field theory quantization 2 4 the ssb cascad source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 "
        + 'Part II: The Physical Sector (Field Theory & Quantization) > 2.4 The SSB Cascade: Origin of Mass & The Solitonic Self > The "Self" as a Soliton: Emergence of Localised Consciousness (Layer 5) > The Definition of the Self: The Triadic Solution'
        + " Specs"
        in report
    )
    assert "P0R01895 - P0R01958" in report
