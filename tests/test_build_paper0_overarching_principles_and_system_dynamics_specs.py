# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Overarching Principles and System Dynamics builder tests
"""Tests for Paper 0 Overarching Principles and System Dynamics source-accounting specs."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_paper0_overarching_principles_and_system_dynamics_specs import (
    build_from_ledger,
    write_outputs,
)


def test_build_overarching_principles_and_system_dynamics_specs_preserves_source_slice() -> None:
    bundle = build_from_ledger()
    assert bundle.summary["source_ledger_span"] == ["P0R05919", "P0R05927"]
    assert bundle.summary["source_record_count"] == 9
    assert bundle.summary["consumed_source_record_count"] == 9
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["unconsumed_source_ledger_ids"] == []
    assert bundle.summary["spec_count"] == 3
    assert bundle.summary["next_source_boundary"] == "P0R05928"
    assert bundle.summary["math_ids"] == []
    assert bundle.summary["image_ids"] == []
    assert bundle.summary["table_ids"] == []


def test_build_overarching_principles_and_system_dynamics_specs_preserves_component_source_formulae() -> (
    None
):
    bundle = build_from_ledger()
    by_context = {spec.context_id: spec for spec in bundle.specs}
    assert set(by_context) == {
        "computational_unifier",
        "layer_5_strange_loop_as_active_inference_engine_sn_precision_control_oct",
        "overarching_principles_and_system_dynamics",
    }
    for spec in bundle.specs:
        assert spec.source_formulae
        assert (
            spec.claim_boundary
            == "source-bounded overarching principles and system dynamics source-accounting bridge; not validation evidence"
        )
        assert spec.hardware_status == "source_methodology_no_experiment"


def test_write_overarching_principles_and_system_dynamics_outputs(tmp_path: Path) -> None:
    outputs = write_outputs(build_from_ledger(), output_dir=tmp_path, date_tag="2099-01-02")
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["summary"]["coverage_match"] is True
    assert (
        payload["summary"]["claim_boundary"]
        == "source-bounded overarching principles and system dynamics source-accounting bridge; not validation evidence"
    )
    assert "Paper 0 " + "Overarching Principles and System Dynamics" + " Specs" in report
    assert "P0R05919 - P0R05927" in report
