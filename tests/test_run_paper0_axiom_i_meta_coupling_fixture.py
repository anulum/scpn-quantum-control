# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom I meta-coupling runner tests
"""Tests for the Paper 0 Axiom I meta-coupling fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_axiom_i_meta_coupling_fixture import render_report, write_outputs


def test_meta_coupling_fixture_runner_writes_json_and_report(tmp_path: Path) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json",
        report_path=tmp_path / "fixture.md",
    )

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")

    assert payload["source_ledger_span"] == ["P0R00717", "P0R00732"]
    assert payload["hardware_status"] == "source_methodology_no_experiment"
    assert payload["predictive_hardware_role_count"] == 3
    assert payload["interaction_component_count"] == 3
    assert payload["next_source_boundary"] == "P0R00733"
    assert "Paper 0 Axiom I Meta-Coupling Fixture" in report
    assert "local_well_behaved_infoton_mediation" in report
    assert "source_meta_coupling_only_no_experiment" in render_report(payload)
