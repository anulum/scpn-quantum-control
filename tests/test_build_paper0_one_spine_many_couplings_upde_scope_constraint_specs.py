# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 One Spine, Many Couplings - UPDE Scope Constraint builder tests
"""Tests for Paper 0 One Spine, Many Couplings - UPDE Scope Constraint source-accounting specs."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_paper0_one_spine_many_couplings_upde_scope_constraint_specs import (
    build_from_ledger,
    write_outputs,
)


def test_build_one_spine_many_couplings_upde_scope_constraint_specs_preserves_source_slice() -> (
    None
):
    bundle = build_from_ledger()
    assert bundle.summary["source_ledger_span"] == ["P0R02682", "P0R02745"]
    assert bundle.summary["source_record_count"] == 64
    assert bundle.summary["consumed_source_record_count"] == 64
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["unconsumed_source_ledger_ids"] == []
    assert bundle.summary["spec_count"] == 1
    assert bundle.summary["next_source_boundary"] == "P0R02746"
    assert bundle.summary["math_ids"] == []
    assert bundle.summary["image_ids"] == []
    assert bundle.summary["table_ids"] == ["TBL010"]


def test_build_one_spine_many_couplings_upde_scope_constraint_specs_preserves_component_source_formulae() -> (
    None
):
    bundle = build_from_ledger()
    by_context = {spec.context_id: spec for spec in bundle.specs}
    assert set(by_context) == {"one_spine_many_couplings_upde_scope_constraint"}
    for spec in bundle.specs:
        assert spec.source_formulae
        assert (
            spec.claim_boundary
            == "source-bounded one spine many couplings upde scope constraint source-accounting bridge; not validation evidence"
        )
        assert spec.hardware_status == "source_methodology_no_experiment"


def test_write_one_spine_many_couplings_upde_scope_constraint_outputs(tmp_path: Path) -> None:
    outputs = write_outputs(build_from_ledger(), output_dir=tmp_path, date_tag="2099-01-02")
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["summary"]["coverage_match"] is True
    assert (
        payload["summary"]["claim_boundary"]
        == "source-bounded one spine many couplings upde scope constraint source-accounting bridge; not validation evidence"
    )
    assert "Paper 0 " + "One Spine, Many Couplings - UPDE Scope Constraint" + " Specs" in report
    assert "P0R02682 - P0R02745" in report
