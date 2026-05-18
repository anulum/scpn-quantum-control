# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 II. The Discrete-Continuous Interface (HHDS) builder tests
"""Tests for Paper 0 II. The Discrete-Continuous Interface (HHDS) source-accounting specs."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_paper0_ii_the_discrete_continuous_interface_hhds_specs import (
    build_from_ledger,
    write_outputs,
)


def test_build_ii_the_discrete_continuous_interface_hhds_specs_preserves_source_slice() -> None:
    bundle = build_from_ledger()
    assert bundle.summary["source_ledger_span"] == ["P0R03260", "P0R03268"]
    assert bundle.summary["source_record_count"] == 9
    assert bundle.summary["consumed_source_record_count"] == 9
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["unconsumed_source_ledger_ids"] == []
    assert bundle.summary["spec_count"] == 4
    assert bundle.summary["next_source_boundary"] == "P0R03269"
    assert bundle.summary["math_ids"] == []
    assert bundle.summary["image_ids"] == []
    assert bundle.summary["table_ids"] == []


def test_build_ii_the_discrete_continuous_interface_hhds_specs_preserves_component_source_formulae() -> (
    None
):
    bundle = build_from_ledger()
    by_context = {spec.context_id: spec for spec in bundle.specs}
    assert set(by_context) == {
        "iii_formalising_emergence_ginzburg_landau",
        "v_the_observer_loop_and_the_strange_loop_of_self_l5",
        "ii_the_discrete_continuous_interface_hhds",
        "iv_the_physics_of_meaning_semantic_transduction",
    }
    for spec in bundle.specs:
        assert spec.source_formulae
        assert (
            spec.claim_boundary
            == "source-bounded ii the discrete continuous interface hhds source-accounting bridge; not validation evidence"
        )
        assert spec.hardware_status == "source_methodology_no_experiment"


def test_write_ii_the_discrete_continuous_interface_hhds_outputs(tmp_path: Path) -> None:
    outputs = write_outputs(build_from_ledger(), output_dir=tmp_path, date_tag="2099-01-02")
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["summary"]["coverage_match"] is True
    assert (
        payload["summary"]["claim_boundary"]
        == "source-bounded ii the discrete continuous interface hhds source-accounting bridge; not validation evidence"
    )
    assert "Paper 0 " + "II. The Discrete-Continuous Interface (HHDS)" + " Specs" in report
    assert "P0R03260 - P0R03268" in report
