# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Quasicriticality with MS-QEC: Two-Timescale Control and Stability Certificates (revision 11.07) runner tests
"""Tests for the Paper 0 Quasicriticality with MS-QEC: Two-Timescale Control and Stability Certificates (revision 11.07) fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_fixture import (
    render_report,
    write_outputs,
)


def test_run_quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_fixture_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json", report_path=tmp_path / "fixture.md"
    )
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["source_ledger_span"] == ["P0R02983", "P0R02990"]
    assert payload["source_record_count"] == 8
    assert payload["component_count"] == 3
    assert payload["next_source_boundary"] == "P0R02991"
    assert (
        payload["claim_boundary"]
        == "source-bounded quasicriticality with ms qec two timescale control and stability certifi source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 "
        + "Quasicriticality with MS-QEC: Two-Timescale Control and Stability Certificates (revision 11.07)"
        + " Fixture"
        in report
    )
    assert (
        "source_quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_only_no_experiment"
        in render_report(payload)
    )
