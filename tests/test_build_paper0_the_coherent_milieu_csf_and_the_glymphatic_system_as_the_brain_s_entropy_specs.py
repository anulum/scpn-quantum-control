# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Coherent Milieu: CSF and the Glymphatic System as the Brain's Entropy Sink builder tests
"""Tests for Paper 0 The Coherent Milieu: CSF and the Glymphatic System as the Brain's Entropy Sink source-accounting specs."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_paper0_the_coherent_milieu_csf_and_the_glymphatic_system_as_the_brain_s_entropy_specs import (
    build_from_ledger,
    write_outputs,
)


def test_build_the_coherent_milieu_csf_and_the_glymphatic_system_as_the_brain_s_entropy_specs_preserves_source_slice() -> (
    None
):
    bundle = build_from_ledger()
    assert bundle.summary["source_ledger_span"] == ["P0R04552", "P0R04559"]
    assert bundle.summary["source_record_count"] == 8
    assert bundle.summary["consumed_source_record_count"] == 8
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["unconsumed_source_ledger_ids"] == []
    assert bundle.summary["spec_count"] == 2
    assert bundle.summary["next_source_boundary"] == "P0R04560"
    assert bundle.summary["math_ids"] == []
    assert bundle.summary["image_ids"] == []
    assert bundle.summary["table_ids"] == []


def test_build_the_coherent_milieu_csf_and_the_glymphatic_system_as_the_brain_s_entropy_specs_preserves_component_source_formulae() -> (
    None
):
    bundle = build_from_ledger()
    by_context = {spec.context_id: spec for spec in bundle.specs}
    assert set(by_context) == {
        "the_coherent_milieu_csf_and_the_glymphatic_system_as_the_brain_s_entropy",
        "introduction_to_the_architecture_of_structure_and_plasticity_domain_i_l3",
    }
    for spec in bundle.specs:
        assert spec.source_formulae
        assert (
            spec.claim_boundary
            == "source-bounded the coherent milieu csf and the glymphatic system as the brain s entropy source-accounting bridge; not validation evidence"
        )
        assert spec.hardware_status == "source_methodology_no_experiment"


def test_write_the_coherent_milieu_csf_and_the_glymphatic_system_as_the_brain_s_entropy_outputs(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(build_from_ledger(), output_dir=tmp_path, date_tag="2099-01-02")
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["summary"]["coverage_match"] is True
    assert (
        payload["summary"]["claim_boundary"]
        == "source-bounded the coherent milieu csf and the glymphatic system as the brain s entropy source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 "
        + "The Coherent Milieu: CSF and the Glymphatic System as the Brain's Entropy Sink"
        + " Specs"
        in report
    )
    assert "P0R04552 - P0R04559" in report
