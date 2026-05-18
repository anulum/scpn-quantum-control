# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Physical Basis of the Ethical Functional: Causal Entropy and Computable Qualia runner tests
"""Tests for the Paper 0 The Physical Basis of the Ethical Functional: Causal Entropy and Computable Qualia fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_fixture import (
    render_report,
    write_outputs,
)


def test_run_the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_fixture_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json", report_path=tmp_path / "fixture.md"
    )
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["source_ledger_span"] == ["P0R04115", "P0R04122"]
    assert payload["source_record_count"] == 8
    assert payload["component_count"] == 2
    assert payload["next_source_boundary"] == "P0R04123"
    assert (
        payload["claim_boundary"]
        == "source-bounded the physical basis of the ethical functional causal entropy and computab source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 "
        + "The Physical Basis of the Ethical Functional: Causal Entropy and Computable Qualia"
        + " Fixture"
        in report
    )
    assert (
        "source_the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_only_no_experiment"
        in render_report(payload)
    )
