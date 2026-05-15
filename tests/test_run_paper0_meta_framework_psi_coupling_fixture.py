# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 meta-framework Psi coupling runner tests
"""Tests for the Paper 0 meta-framework/Psi-coupling fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_meta_framework_psi_coupling_fixture import (
    render_report,
    write_outputs,
)


def test_meta_framework_psi_coupling_fixture_runner_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json",
        report_path=tmp_path / "fixture.md",
    )

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")

    assert payload["source_ledger_span"] == ["P0R00838", "P0R00904"]
    assert payload["hardware_status"] == "source_methodology_no_experiment"
    assert payload["source_record_count"] == 67
    assert payload["image_or_figure_record_count"] == 6
    assert payload["blank_record_count"] == 2
    assert payload["next_source_boundary"] == "P0R00905"
    assert "Paper 0 Meta-Framework Psi Coupling Fixture" in report
    assert "h_int_minus_lambda_psis_sigma_coupling_statement" in report
    assert "source_meta_framework_psi_coupling_only_no_experiment" in render_report(payload)
