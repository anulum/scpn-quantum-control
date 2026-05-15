# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 category universal grammar runner tests
"""Tests for the Paper 0 category/universal-grammar fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_category_universal_grammar_fixture import render_report, write_outputs


def test_category_universal_grammar_fixture_runner_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json",
        report_path=tmp_path / "fixture.md",
    )

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")

    assert payload["source_ledger_span"] == ["P0R00905", "P0R00986"]
    assert payload["hardware_status"] == "source_methodology_no_experiment"
    assert payload["source_record_count"] == 82
    assert payload["image_record_count"] == 5
    assert payload["blank_record_count"] == 3
    assert payload["next_source_boundary"] == "P0R00987"
    assert "Paper 0 Category Universal Grammar Fixture" in report
    assert "consciousness_physics_functors_and_natural_transformation" in report
    assert "source_category_universal_grammar_only_no_experiment" in render_report(payload)
