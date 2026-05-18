# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Physics of Teleology: A Derivation of the Ethical Functional runner tests
"""Tests for the Paper 0 The Physics of Teleology: A Derivation of the Ethical Functional fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_the_physics_of_teleology_a_derivation_of_the_ethical_functional_fixture import (
    render_report,
    write_outputs,
)


def test_run_the_physics_of_teleology_a_derivation_of_the_ethical_functional_fixture_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json", report_path=tmp_path / "fixture.md"
    )
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["source_ledger_span"] == ["P0R03581", "P0R03602"]
    assert payload["source_record_count"] == 22
    assert payload["component_count"] == 1
    assert payload["next_source_boundary"] == "P0R03603"
    assert (
        payload["claim_boundary"]
        == "source-bounded the physics of teleology a derivation of the ethical functional source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 "
        + "The Physics of Teleology: A Derivation of the Ethical Functional"
        + " Fixture"
        in report
    )
    assert (
        "source_the_physics_of_teleology_a_derivation_of_the_ethical_functional_only_no_experiment"
        in render_report(payload)
    )
