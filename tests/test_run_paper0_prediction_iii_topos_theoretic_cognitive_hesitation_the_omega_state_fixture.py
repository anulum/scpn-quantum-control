# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Prediction III: Topos-Theoretic Cognitive Hesitation (The $\Omega$-State) runner tests
r"""Tests for the Paper 0 Prediction III: Topos-Theoretic Cognitive Hesitation (The $\Omega$-State) fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state_fixture import (
    render_report,
    write_outputs,
)


def test_run_prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state_fixture_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json", report_path=tmp_path / "fixture.md"
    )
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["source_ledger_span"] == ["P0R05124", "P0R05142"]
    assert payload["source_record_count"] == 19
    assert payload["component_count"] == 1
    assert payload["next_source_boundary"] == "P0R05143"
    assert (
        payload["claim_boundary"]
        == "source-bounded prediction iii topos theoretic cognitive hesitation the omega state source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 "
        + "Prediction III: Topos-Theoretic Cognitive Hesitation (The $\\Omega$-State)"
        + " Fixture"
        in report
    )
    assert (
        "source_prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state_only_no_experiment"
        in render_report(payload)
    )
