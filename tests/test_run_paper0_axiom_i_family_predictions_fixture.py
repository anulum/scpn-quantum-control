# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom I family predictions runner tests
"""Tests for the Paper 0 Axiom I family-predictions fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_axiom_i_family_predictions_fixture import (
    render_report,
    write_outputs,
)


def test_family_predictions_fixture_runner_writes_json_and_report(tmp_path: Path) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json",
        report_path=tmp_path / "fixture.md",
    )

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")

    assert payload["source_ledger_span"] == ["P0R00747", "P0R00756"]
    assert payload["hardware_status"] == "source_methodology_no_experiment"
    assert payload["conditional_prediction_count"] == 3
    assert payload["rejected_model_class_count"] == 5
    assert payload["blank_separator_count"] == 1
    assert payload["next_source_boundary"] == "P0R00757"
    assert "Paper 0 Axiom I Family Predictions Fixture" in report
    assert "ssb_mass_m_a_equals_g_v" in report
    assert "source_family_predictions_only_no_experiment" in render_report(payload)
