# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Psi-Field as a Negentropy Source and the Landauer Cost of Coherence runner tests
"""Tests for the Paper 0 The Psi-Field as a Negentropy Source and the Landauer Cost of Coherence fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_the_psi_field_as_a_negentropy_source_and_the_landauer_cost_of_coherence_fixture import (
    render_report,
    write_outputs,
)


def test_run_the_psi_field_as_a_negentropy_source_and_the_landauer_cost_of_coherence_fixture_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json", report_path=tmp_path / "fixture.md"
    )
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["source_ledger_span"] == ["P0R05953", "P0R05963"]
    assert payload["source_record_count"] == 11
    assert payload["component_count"] == 3
    assert payload["next_source_boundary"] == "P0R05964"
    assert (
        payload["claim_boundary"]
        == "source-bounded the psi field as a negentropy source and the landauer cost of coherence source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 "
        + "The Psi-Field as a Negentropy Source and the Landauer Cost of Coherence"
        + " Fixture"
        in report
    )
    assert (
        "source_the_psi_field_as_a_negentropy_source_and_the_landauer_cost_of_coherence_only_no_experiment"
        in render_report(payload)
    )
