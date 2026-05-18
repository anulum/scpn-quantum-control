# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 4.5 The Strange Loop of Closure - Meta-Layer 16 and The Anulum runner tests
"""Tests for the Paper 0 4.5 The Strange Loop of Closure - Meta-Layer 16 and The Anulum fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_section_4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum_fixture import (
    render_report,
    write_outputs,
)


def test_run_section_4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum_fixture_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json", report_path=tmp_path / "fixture.md"
    )
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["source_ledger_span"] == ["P0R04216", "P0R04223"]
    assert payload["source_record_count"] == 8
    assert payload["component_count"] == 2
    assert payload["next_source_boundary"] == "P0R04224"
    assert (
        payload["claim_boundary"]
        == "source-bounded section 4 5 the strange loop of closure meta layer 16 and the anulum source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 " + "4.5 The Strange Loop of Closure - Meta-Layer 16 and The Anulum" + " Fixture"
        in report
    )
    assert (
        "source_section_4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum_only_no_experiment"
        in render_report(payload)
    )
