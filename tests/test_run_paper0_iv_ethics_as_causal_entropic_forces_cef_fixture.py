# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 IV. Ethics as Causal Entropic Forces (CEF): runner tests
"""Tests for the Paper 0 IV. Ethics as Causal Entropic Forces (CEF): fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_iv_ethics_as_causal_entropic_forces_cef_fixture import (
    render_report,
    write_outputs,
)


def test_run_iv_ethics_as_causal_entropic_forces_cef_fixture_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json", report_path=tmp_path / "fixture.md"
    )
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["source_ledger_span"] == ["P0R06107", "P0R06114"]
    assert payload["source_record_count"] == 8
    assert payload["component_count"] == 2
    assert payload["next_source_boundary"] == "P0R06115"
    assert (
        payload["claim_boundary"]
        == "source-bounded iv ethics as causal entropic forces cef source-accounting bridge; not validation evidence"
    )
    assert "Paper 0 " + "IV. Ethics as Causal Entropic Forces (CEF):" + " Fixture" in report
    assert "source_iv_ethics_as_causal_entropic_forces_cef_only_no_experiment" in render_report(
        payload
    )
