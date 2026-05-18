# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 III. The SCPN Measurement Postulate (IIT-OR and QZE) runner tests
"""Tests for the Paper 0 III. The SCPN Measurement Postulate (IIT-OR and QZE) fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_iii_the_scpn_measurement_postulate_iit_or_and_qze_fixture import (
    render_report,
    write_outputs,
)


def test_run_iii_the_scpn_measurement_postulate_iit_or_and_qze_fixture_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json", report_path=tmp_path / "fixture.md"
    )
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["source_ledger_span"] == ["P0R05986", "P0R05993"]
    assert payload["source_record_count"] == 8
    assert payload["component_count"] == 3
    assert payload["next_source_boundary"] == "P0R05994"
    assert (
        payload["claim_boundary"]
        == "source-bounded iii the scpn measurement postulate iit or and qze source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 " + "III. The SCPN Measurement Postulate (IIT-OR and QZE)" + " Fixture" in report
    )
    assert (
        "source_iii_the_scpn_measurement_postulate_iit_or_and_qze_only_no_experiment"
        in render_report(payload)
    )
