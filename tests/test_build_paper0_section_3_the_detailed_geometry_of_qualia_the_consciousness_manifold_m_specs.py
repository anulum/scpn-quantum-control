# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 3. The Detailed Geometry of Qualia (The Consciousness Manifold M): builder tests
"""Tests for Paper 0 3. The Detailed Geometry of Qualia (The Consciousness Manifold M): source-accounting specs."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_paper0_section_3_the_detailed_geometry_of_qualia_the_consciousness_manifold_m_specs import (
    build_from_ledger,
    write_outputs,
)


def test_build_section_3_the_detailed_geometry_of_qualia_the_consciousness_manifold_m_specs_preserves_source_slice() -> (
    None
):
    bundle = build_from_ledger()
    assert bundle.summary["source_ledger_span"] == ["P0R04684", "P0R04692"]
    assert bundle.summary["source_record_count"] == 9
    assert bundle.summary["consumed_source_record_count"] == 9
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["unconsumed_source_ledger_ids"] == []
    assert bundle.summary["spec_count"] == 3
    assert bundle.summary["next_source_boundary"] == "P0R04693"
    assert bundle.summary["math_ids"] == []
    assert bundle.summary["image_ids"] == []
    assert bundle.summary["table_ids"] == []


def test_build_section_3_the_detailed_geometry_of_qualia_the_consciousness_manifold_m_specs_preserves_component_source_formulae() -> (
    None
):
    bundle = build_from_ledger()
    by_context = {spec.context_id: spec for spec in bundle.specs}
    assert set(by_context) == {
        "3_the_detailed_geometry_of_qualia_the_consciousness_manifold_m",
        "v_integrative_systems_the_embodied_brain",
        "4_the_binding_problem_and_the_role_of_central_hubs",
    }
    for spec in bundle.specs:
        assert spec.source_formulae
        assert (
            spec.claim_boundary
            == "source-bounded section 3 the detailed geometry of qualia the consciousness manifold m source-accounting bridge; not validation evidence"
        )
        assert spec.hardware_status == "source_methodology_no_experiment"


def test_write_section_3_the_detailed_geometry_of_qualia_the_consciousness_manifold_m_outputs(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(build_from_ledger(), output_dir=tmp_path, date_tag="2099-01-02")
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["summary"]["coverage_match"] is True
    assert (
        payload["summary"]["claim_boundary"]
        == "source-bounded section 3 the detailed geometry of qualia the consciousness manifold m source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 "
        + "3. The Detailed Geometry of Qualia (The Consciousness Manifold M):"
        + " Specs"
        in report
    )
    assert "P0R04684 - P0R04692" in report
