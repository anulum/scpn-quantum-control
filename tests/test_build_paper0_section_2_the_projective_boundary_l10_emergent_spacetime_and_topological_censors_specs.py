# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. The Projective Boundary (L10): Emergent Spacetime and Topological Censorship builder tests
"""Tests for Paper 0 2. The Projective Boundary (L10): Emergent Spacetime and Topological Censorship source-accounting specs."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_paper0_section_2_the_projective_boundary_l10_emergent_spacetime_and_topological_censors_specs import (
    build_from_ledger,
    write_outputs,
)


def test_build_section_2_the_projective_boundary_l10_emergent_spacetime_and_topological_censors_specs_preserves_source_slice() -> (
    None
):
    bundle = build_from_ledger()
    assert bundle.summary["source_ledger_span"] == ["P0R04454", "P0R04461"]
    assert bundle.summary["source_record_count"] == 8
    assert bundle.summary["consumed_source_record_count"] == 8
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["unconsumed_source_ledger_ids"] == []
    assert bundle.summary["spec_count"] == 3
    assert bundle.summary["next_source_boundary"] == "P0R04462"
    assert bundle.summary["math_ids"] == []
    assert bundle.summary["image_ids"] == []
    assert bundle.summary["table_ids"] == []


def test_build_section_2_the_projective_boundary_l10_emergent_spacetime_and_topological_censors_specs_preserves_component_source_formulae() -> (
    None
):
    bundle = build_from_ledger()
    by_context = {spec.context_id: spec for spec in bundle.specs}
    assert set(by_context) == {
        "the_neurobiological_architecture_of_the_scpn",
        "2_the_projective_boundary_l10_emergent_spacetime_and_topological_censors",
        "vii_synthesis_the_scpn_torus",
    }
    for spec in bundle.specs:
        assert spec.source_formulae
        assert (
            spec.claim_boundary
            == "source-bounded section 2 the projective boundary l10 emergent spacetime and topological censors source-accounting bridge; not validation evidence"
        )
        assert spec.hardware_status == "source_methodology_no_experiment"


def test_write_section_2_the_projective_boundary_l10_emergent_spacetime_and_topological_censors_outputs(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(build_from_ledger(), output_dir=tmp_path, date_tag="2099-01-02")
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["summary"]["coverage_match"] is True
    assert (
        payload["summary"]["claim_boundary"]
        == "source-bounded section 2 the projective boundary l10 emergent spacetime and topological censors source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 "
        + "2. The Projective Boundary (L10): Emergent Spacetime and Topological Censorship"
        + " Specs"
        in report
    )
    assert "P0R04454 - P0R04461" in report
