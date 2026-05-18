# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 15-Layer Summary Table > 3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE) > The Unified Phase Dynamics Equation (UPDE) > One Spine, Many Couplings - UPDE Scope Constraint runner tests
"""Tests for the Paper 0 15-Layer Summary Table > 3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE) > The Unified Phase Dynamics Equation (UPDE) > One Spine, Many Couplings - UPDE Scope Constraint fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_fixture import (
    render_report,
    write_outputs,
)


def test_run_section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_fixture_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json", report_path=tmp_path / "fixture.md"
    )
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["source_ledger_span"] == ["P0R02746", "P0R02809"]
    assert payload["source_record_count"] == 64
    assert payload["component_count"] == 1
    assert payload["next_source_boundary"] == "P0R02810"
    assert (
        payload["claim_boundary"]
        == "source-bounded section 15 layer summary table 3 2 the dynamic spine the unified phase dynamics source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 "
        + "15-Layer Summary Table > 3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE) > The Unified Phase Dynamics Equation (UPDE) > One Spine, Many Couplings - UPDE Scope Constraint"
        + " Fixture"
        in report
    )
    assert (
        "source_section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_only_no_experiment"
        in render_report(payload)
    )
