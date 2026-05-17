# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Part II: The Physical Sector (Field Theory & Quantization) > 2.4 The SSB Cascade: Origin of Mass & The Solitonic Self > The "Self" as a Soliton: Emergence of Localised Consciousness (Layer 5) > The Definition of the Self: The Triadic Solution runner tests
"""Tests for the Paper 0 Part II: The Physical Sector (Field Theory & Quantization) > 2.4 The SSB Cascade: Origin of Mass & The Solitonic Self > The "Self" as a Soliton: Emergence of Localised Consciousness (Layer 5) > The Definition of the Self: The Triadic Solution fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad_fixture import (
    render_report,
    write_outputs,
)


def test_run_part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad_fixture_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json", report_path=tmp_path / "fixture.md"
    )
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["source_ledger_span"] == ["P0R01895", "P0R01958"]
    assert payload["source_record_count"] == 64
    assert payload["component_count"] == 1
    assert payload["next_source_boundary"] == "P0R01959"
    assert (
        payload["claim_boundary"]
        == "source-bounded part ii the physical sector field theory quantization 2 4 the ssb cascad source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 "
        + 'Part II: The Physical Sector (Field Theory & Quantization) > 2.4 The SSB Cascade: Origin of Mass & The Solitonic Self > The "Self" as a Soliton: Emergence of Localised Consciousness (Layer 5) > The Definition of the Self: The Triadic Solution'
        + " Fixture"
        in report
    )
    assert (
        "source_part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad_only_no_experiment"
        in render_report(payload)
    )
