# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 U1 FIM multiscale dynamics runner tests
"""Tests for the Paper 0 U(1)/FIM multiscale-dynamics fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_u1_fim_multiscale_dynamics_fixture import write_outputs


def test_run_u1_fim_multiscale_fixture_writes_result_and_report(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    report = tmp_path / "report.md"

    payload = write_outputs(output_path=output, report_path=report)

    written = json.loads(output.read_text(encoding="utf-8"))
    report_text = report.read_text(encoding="utf-8")

    assert payload["source_ledger_span"] == ["P0R00506", "P0R00544"]
    assert written["upde_component_count"] == 3
    assert written["validation_boundary_count"] == 3
    assert written["hardware_status"] == "source_methodology_no_experiment"
    assert "Paper 0 U1 FIM Multiscale Dynamics Fixture" in report_text
    assert "requires_quantitative_biophysical_validation" in report_text
