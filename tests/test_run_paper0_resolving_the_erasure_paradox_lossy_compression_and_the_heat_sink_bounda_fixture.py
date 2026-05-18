# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Resolving the Erasure Paradox: Lossy Compression and the Heat Sink Boundary runner tests
"""Tests for the Paper 0 Resolving the Erasure Paradox: Lossy Compression and the Heat Sink Boundary fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_resolving_the_erasure_paradox_lossy_compression_and_the_heat_sink_bounda_fixture import (
    render_report,
    write_outputs,
)


def test_run_resolving_the_erasure_paradox_lossy_compression_and_the_heat_sink_bounda_fixture_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json", report_path=tmp_path / "fixture.md"
    )
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["source_ledger_span"] == ["P0R05964", "P0R05985"]
    assert payload["source_record_count"] == 22
    assert payload["component_count"] == 1
    assert payload["next_source_boundary"] == "P0R05986"
    assert (
        payload["claim_boundary"]
        == "source-bounded resolving the erasure paradox lossy compression and the heat sink bounda source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 "
        + "Resolving the Erasure Paradox: Lossy Compression and the Heat Sink Boundary"
        + " Fixture"
        in report
    )
    assert (
        "source_resolving_the_erasure_paradox_lossy_compression_and_the_heat_sink_bounda_only_no_experiment"
        in render_report(payload)
    )
