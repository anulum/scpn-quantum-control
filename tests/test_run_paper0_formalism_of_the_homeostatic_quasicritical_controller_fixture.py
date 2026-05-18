# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Formalism of the Homeostatic Quasicritical Controller runner tests
"""Tests for the Paper 0 Formalism of the Homeostatic Quasicritical Controller fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_formalism_of_the_homeostatic_quasicritical_controller_fixture import (
    render_report,
    write_outputs,
)


def test_run_formalism_of_the_homeostatic_quasicritical_controller_fixture_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json", report_path=tmp_path / "fixture.md"
    )
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["source_ledger_span"] == ["P0R02869", "P0R02893"]
    assert payload["source_record_count"] == 25
    assert payload["component_count"] == 1
    assert payload["next_source_boundary"] == "P0R02894"
    assert (
        payload["claim_boundary"]
        == "source-bounded formalism of the homeostatic quasicritical controller source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 " + "Formalism of the Homeostatic Quasicritical Controller" + " Fixture" in report
    )
    assert (
        "source_formalism_of_the_homeostatic_quasicritical_controller_only_no_experiment"
        in render_report(payload)
    )
