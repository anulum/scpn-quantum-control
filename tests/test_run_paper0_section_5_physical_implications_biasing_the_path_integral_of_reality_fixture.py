# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 5. Physical Implications: Biasing the Path Integral of Reality runner tests
"""Tests for the Paper 0 5. Physical Implications: Biasing the Path Integral of Reality fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_section_5_physical_implications_biasing_the_path_integral_of_reality_fixture import (
    render_report,
    write_outputs,
)


def test_run_section_5_physical_implications_biasing_the_path_integral_of_reality_fixture_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json", report_path=tmp_path / "fixture.md"
    )
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["source_ledger_span"] == ["P0R03848", "P0R03868"]
    assert payload["source_record_count"] == 21
    assert payload["component_count"] == 1
    assert payload["next_source_boundary"] == "P0R03869"
    assert (
        payload["claim_boundary"]
        == "source-bounded section 5 physical implications biasing the path integral of reality source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 " + "5. Physical Implications: Biasing the Path Integral of Reality" + " Fixture"
        in report
    )
    assert (
        "source_section_5_physical_implications_biasing_the_path_integral_of_reality_only_no_experiment"
        in render_report(payload)
    )
