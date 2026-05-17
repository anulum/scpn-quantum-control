# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Timing the Engine: UPDE Phase-Lags ($\tau_{ij}$) and Physiological Delays runner tests
"""Tests for the Paper 0 Timing the Engine: UPDE Phase-Lags ($\tau_{ij}$) and Physiological Delays fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_fixture import (
    render_report,
    write_outputs,
)


def test_run_timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_fixture_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json", report_path=tmp_path / "fixture.md"
    )
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["source_ledger_span"] == ["P0R02223", "P0R02236"]
    assert payload["source_record_count"] == 14
    assert payload["component_count"] == 1
    assert payload["next_source_boundary"] == "P0R02237"
    assert (
        payload["claim_boundary"]
        == "source-bounded timing the engine upde phase lags tau ij and physiological delays source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 "
        + "Timing the Engine: UPDE Phase-Lags ($\\tau_{ij}$) and Physiological Delays"
        + " Fixture"
        in report
    )
    assert (
        "source_timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_only_no_experiment"
        in render_report(payload)
    )
