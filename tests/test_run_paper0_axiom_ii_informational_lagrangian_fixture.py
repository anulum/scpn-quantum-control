# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom II informational Lagrangian runner tests
"""Tests for the Paper 0 Axiom II informational-Lagrangian fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_axiom_ii_informational_lagrangian_fixture import (
    render_report,
    write_outputs,
)


def test_informational_lagrangian_fixture_runner_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json",
        report_path=tmp_path / "fixture.md",
    )

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")

    assert payload["source_ledger_span"] == ["P0R00782", "P0R00790"]
    assert payload["hardware_status"] == "source_methodology_no_experiment"
    assert payload["source_record_count"] == 9
    assert payload["gauge_equation_count"] == 2
    assert payload["pullback_protocol_count"] == 1
    assert payload["next_source_boundary"] == "P0R00791"
    assert "Paper 0 Axiom II Informational Lagrangian Fixture" in report
    assert "chapter6_pullback_protocol_falsifiability_bridge" in report
    assert "source_axiom_ii_informational_lagrangian_only_no_experiment" in render_report(payload)
