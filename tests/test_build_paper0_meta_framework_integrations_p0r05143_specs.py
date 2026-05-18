# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Meta-Framework Integrations builder tests
"""Tests for Paper 0 Meta-Framework Integrations source-accounting specs."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_paper0_meta_framework_integrations_p0r05143_specs import (
    build_from_ledger,
    write_outputs,
)


def test_build_meta_framework_integrations_p0r05143_specs_preserves_source_slice() -> None:
    bundle = build_from_ledger()
    assert bundle.summary["source_ledger_span"] == ["P0R05143", "P0R05151"]
    assert bundle.summary["source_record_count"] == 9
    assert bundle.summary["consumed_source_record_count"] == 9
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["unconsumed_source_ledger_ids"] == []
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["next_source_boundary"] == "P0R05152"
    assert bundle.summary["math_ids"] == []
    assert bundle.summary["image_ids"] == []
    assert bundle.summary["table_ids"] == []


def test_build_meta_framework_integrations_p0r05143_specs_preserves_component_source_formulae() -> (
    None
):
    bundle = build_from_ledger()
    by_context = {spec.context_id: spec for spec in bundle.specs}
    assert set(by_context) == {
        "predictive_coding_integration",
        "prediction_i_nv_mea_tests_the_model_s_geometry",
        "psis_field_coupling_integration",
        "meta_framework_integrations",
        "prediction_ii_qrng_tests_the_model_s_deepest_prior",
    }
    for spec in bundle.specs:
        assert spec.source_formulae
        assert (
            spec.claim_boundary
            == "source-bounded meta framework integrations p0r05143 source-accounting bridge; not validation evidence"
        )
        assert spec.hardware_status == "source_methodology_no_experiment"


def test_write_meta_framework_integrations_p0r05143_outputs(tmp_path: Path) -> None:
    outputs = write_outputs(build_from_ledger(), output_dir=tmp_path, date_tag="2099-01-02")
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["summary"]["coverage_match"] is True
    assert (
        payload["summary"]["claim_boundary"]
        == "source-bounded meta framework integrations p0r05143 source-accounting bridge; not validation evidence"
    )
    assert "Paper 0 " + "Meta-Framework Integrations" + " Specs" in report
    assert "P0R05143 - P0R05151" in report
