# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 meta-framework integrations builder tests
"""Tests for Paper 0 meta-framework integrations source-accounting specs."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_paper0_meta_framework_integrations_specs import build_from_ledger, write_outputs


def test_build_meta_framework_integrations_specs_preserves_source_slice() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R01714", "P0R01726"]
    assert bundle.summary["source_record_count"] == 13
    assert bundle.summary["consumed_source_record_count"] == 13
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["unconsumed_source_ledger_ids"] == []
    assert bundle.summary["spec_count"] == 3
    assert bundle.summary["next_source_boundary"] == "P0R01727"
    assert bundle.summary["math_ids"] == []
    assert bundle.summary["image_ids"] == []
    assert bundle.summary["table_ids"] == []
    assert bundle.summary["spec_keys"] == [
        "meta_framework_integrations.predictive_coding_flat_prior",
        "meta_framework_integrations.psi_s_field_coupling",
        "meta_framework_integrations.differentiated_sigma_interface",
    ]


def test_build_meta_framework_integrations_specs_preserves_source_formulae() -> None:
    bundle = build_from_ledger()
    formulae = {spec.context_id: spec.source_formulae for spec in bundle.specs}

    assert (
        "Source-Field begins as a flat prior with maximum uncertainty and minimal structure"
        in formulae["predictive_coding_flat_prior"]
    )
    assert (
        "universal Psi_s field couples via H_int = -lambda * Psi_s * sigma"
        in formulae["psi_s_field_coupling"]
    )
    assert (
        "SSB-2 differentiates sigma_collective into localized sigma_individual solitons"
        in formulae["differentiated_sigma_interface"]
    )


def test_write_meta_framework_integrations_outputs(tmp_path: Path) -> None:
    outputs = write_outputs(build_from_ledger(), output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")

    assert payload["summary"]["coverage_match"] is True
    assert (
        payload["summary"]["claim_boundary"]
        == "source-bounded meta-framework integrations bridge; not validation evidence"
    )
    assert "Paper 0 Meta-Framework Integrations Specs" in report
    assert "P0R01714 - P0R01726" in report
