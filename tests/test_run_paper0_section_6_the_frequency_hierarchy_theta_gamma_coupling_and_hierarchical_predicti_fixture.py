# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 6. The Frequency Hierarchy: Theta-Gamma Coupling and Hierarchical Predictive Coding runner tests
"""Tests for the Paper 0 6. The Frequency Hierarchy: Theta-Gamma Coupling and Hierarchical Predictive Coding fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_section_6_the_frequency_hierarchy_theta_gamma_coupling_and_hierarchical_predicti_fixture import (
    render_report,
    write_outputs,
)


def test_run_section_6_the_frequency_hierarchy_theta_gamma_coupling_and_hierarchical_predicti_fixture_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json", report_path=tmp_path / "fixture.md"
    )
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["source_ledger_span"] == ["P0R04507", "P0R04516"]
    assert payload["source_record_count"] == 10
    assert payload["component_count"] == 2
    assert payload["next_source_boundary"] == "P0R04517"
    assert (
        payload["claim_boundary"]
        == "source-bounded section 6 the frequency hierarchy theta gamma coupling and hierarchical predicti source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 "
        + "6. The Frequency Hierarchy: Theta-Gamma Coupling and Hierarchical Predictive Coding"
        + " Fixture"
        in report
    )
    assert (
        "source_section_6_the_frequency_hierarchy_theta_gamma_coupling_and_hierarchical_predicti_only_no_experiment"
        in render_report(payload)
    )
