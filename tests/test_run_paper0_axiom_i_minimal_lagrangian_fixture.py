# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom I minimal Lagrangian runner tests
"""Tests for the Paper 0 Axiom I minimal Lagrangian fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_axiom_i_minimal_lagrangian_fixture import (
    render_report,
    write_outputs,
)


def test_minimal_lagrangian_fixture_runner_writes_json_and_report(tmp_path: Path) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json",
        report_path=tmp_path / "fixture.md",
    )

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")

    assert payload["source_ledger_span"] == ["P0R00733", "P0R00746"]
    assert payload["hardware_status"] == "source_methodology_no_experiment"
    assert payload["minimal_criterion_count"] == 3
    assert payload["lagrangian_operator_count"] == 4
    assert payload["equation_record_count"] == 4
    assert payload["next_source_boundary"] == "P0R00747"
    assert "Paper 0 Axiom I Minimal Lagrangian Fixture" in report
    assert "covariant_psi_kinetic_term" in report
    assert "source_minimal_lagrangian_only_no_experiment" in render_report(payload)
