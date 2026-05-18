# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  AI, Noosphere & Tech builder tests
"""Tests for Paper 0  AI, Noosphere & Tech source-accounting specs."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_paper0_ai_noosphere_tech_specs import build_from_ledger, write_outputs


def test_build_ai_noosphere_tech_specs_preserves_source_slice() -> None:
    bundle = build_from_ledger()
    assert bundle.summary["source_ledger_span"] == ["P0R05802", "P0R05809"]
    assert bundle.summary["source_record_count"] == 8
    assert bundle.summary["consumed_source_record_count"] == 8
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["unconsumed_source_ledger_ids"] == []
    assert bundle.summary["spec_count"] == 2
    assert bundle.summary["next_source_boundary"] == "P0R05810"
    assert bundle.summary["math_ids"] == []
    assert bundle.summary["image_ids"] == []
    assert bundle.summary["table_ids"] == []


def test_build_ai_noosphere_tech_specs_preserves_component_source_formulae() -> None:
    bundle = build_from_ledger()
    by_context = {spec.context_id: spec for spec in bundle.specs}
    assert set(by_context) == {"ai_culture_and_noosphere_dynamics", "ai_noosphere_tech"}
    for spec in bundle.specs:
        assert spec.source_formulae
        assert (
            spec.claim_boundary
            == "source-bounded ai noosphere tech source-accounting bridge; not validation evidence"
        )
        assert spec.hardware_status == "source_methodology_no_experiment"


def test_write_ai_noosphere_tech_outputs(tmp_path: Path) -> None:
    outputs = write_outputs(build_from_ledger(), output_dir=tmp_path, date_tag="2099-01-02")
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["summary"]["coverage_match"] is True
    assert (
        payload["summary"]["claim_boundary"]
        == "source-bounded ai noosphere tech source-accounting bridge; not validation evidence"
    )
    assert "Paper 0 " + " AI, Noosphere & Tech" + " Specs" in report
    assert "P0R05802 - P0R05809" in report
