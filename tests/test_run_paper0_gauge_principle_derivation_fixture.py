# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 gauge-principle derivation runner tests
"""Tests for the Paper 0 gauge-principle-derivation fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_gauge_principle_derivation_fixture import render_report, write_outputs


def test_gauge_principle_derivation_fixture_runner_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json",
        report_path=tmp_path / "fixture.md",
    )

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")

    assert payload["source_ledger_span"] == ["P0R01018", "P0R01077"]
    assert payload["hardware_status"] == "source_methodology_no_experiment"
    assert payload["source_record_count"] == 60
    assert payload["blank_record_count"] == 1
    assert payload["image_record_count"] == 1
    assert payload["next_source_boundary"] == "P0R01078"
    assert "Paper 0 Gauge Principle Derivation Fixture" in report
    assert "covariant_derivative_gauge_transform_and_minimal_coupling" in report
    assert "source_gauge_principle_derivation_only_no_experiment" in render_report(payload)
