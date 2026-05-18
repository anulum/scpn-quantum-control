# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Glial Scaffolding: Astrocytic Regulation of Neural Synchrony and Criticality runner tests
"""Tests for the Paper 0 Glial Scaffolding: Astrocytic Regulation of Neural Synchrony and Criticality fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_glial_scaffolding_astrocytic_regulation_of_neural_synchrony_and_critical_fixture import (
    render_report,
    write_outputs,
)


def test_run_glial_scaffolding_astrocytic_regulation_of_neural_synchrony_and_critical_fixture_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json", report_path=tmp_path / "fixture.md"
    )
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["source_ledger_span"] == ["P0R05455", "P0R05478"]
    assert payload["source_record_count"] == 24
    assert payload["component_count"] == 2
    assert payload["next_source_boundary"] == "P0R05479"
    assert (
        payload["claim_boundary"]
        == "source-bounded glial scaffolding astrocytic regulation of neural synchrony and critical source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 "
        + "Glial Scaffolding: Astrocytic Regulation of Neural Synchrony and Criticality"
        + " Fixture"
        in report
    )
    assert (
        "source_glial_scaffolding_astrocytic_regulation_of_neural_synchrony_and_critical_only_no_experiment"
        in render_report(payload)
    )
