# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 IV. Sub-Synaptic and Axonal Architecture (L1-L3) runner tests
"""Tests for the Paper 0 IV. Sub-Synaptic and Axonal Architecture (L1-L3) fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_iv_sub_synaptic_and_axonal_architecture_l1_l3_fixture import (
    render_report,
    write_outputs,
)


def test_run_iv_sub_synaptic_and_axonal_architecture_l1_l3_fixture_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json", report_path=tmp_path / "fixture.md"
    )
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["source_ledger_span"] == ["P0R04786", "P0R04793"]
    assert payload["source_record_count"] == 8
    assert payload["component_count"] == 4
    assert payload["next_source_boundary"] == "P0R04794"
    assert (
        payload["claim_boundary"]
        == "source-bounded iv sub synaptic and axonal architecture l1 l3 source-accounting bridge; not validation evidence"
    )
    assert "Paper 0 " + "IV. Sub-Synaptic and Axonal Architecture (L1-L3)" + " Fixture" in report
    assert (
        "source_iv_sub_synaptic_and_axonal_architecture_l1_l3_only_no_experiment"
        in render_report(payload)
    )
