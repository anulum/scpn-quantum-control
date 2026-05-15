# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom II infoton geometry builder tests
"""Tests for Paper 0 Axiom II infoton-geometry spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_axiom_ii_infoton_geometry_validation_spec,
)
from scripts.build_paper0_axiom_ii_infoton_geometry_specs import (
    SOURCE_LEDGER_IDS,
    build_axiom_ii_infoton_geometry_specs,
    build_from_ledger,
    render_report,
    write_outputs,
)


def test_infoton_geometry_builder_preserves_contiguous_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R00770", "P0R00774"]
    assert bundle.summary["source_record_count"] == 5
    assert bundle.summary["consumed_source_record_count"] == 5
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 4
    assert bundle.summary["gauge_necessity_count"] == 1
    assert bundle.summary["baseline_lagrangian_count"] == 1
    assert bundle.summary["fim_claim_count"] == 1
    assert bundle.summary["next_source_boundary"] == "P0R00775"
    assert [spec.key for spec in bundle.specs] == [
        "axiom_ii_infoton_geometry.problem_heading",
        "axiom_ii_infoton_geometry.gauge_necessity",
        "axiom_ii_infoton_geometry.spacetime_metric_baseline",
        "axiom_ii_infoton_geometry.fim_dynamics_claim",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_infoton_geometry_builder_keeps_source_formulae_and_boundaries() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        'The Central Problem: The Geometry of the "Infoton"'
        in specs["axiom_ii_infoton_geometry.problem_heading"].source_formulae
    )
    assert (
        "U(1) gauge principle is a mathematical necessity"
        in specs["axiom_ii_infoton_geometry.gauge_necessity"].source_formulae
    )
    assert (
        "local complex consciousness field requires a mediating gauge field"
        in specs["axiom_ii_infoton_geometry.gauge_necessity"].source_formulae
    )
    assert (
        "spin-1 vector boson infoton"
        in specs["axiom_ii_infoton_geometry.gauge_necessity"].source_formulae
    )
    assert (
        "L_EM = -1/4 F_mu_nu F^mu_nu"
        in specs["axiom_ii_infoton_geometry.spacetime_metric_baseline"].source_formulae
    )
    assert (
        "Fisher Information Metric g_FIM"
        in specs["axiom_ii_infoton_geometry.fim_dynamics_claim"].source_formulae
    )


def test_infoton_geometry_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R00773":
            continue
        records.append(
            {
                "ledger_id": record_id,
                "source_record_id": f"{record_id}:stub",
                "source_block_index": int(record_id[3:]),
                "section_path": "Paper 0 > Axiom II > Infoton Geometry",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_axiom_ii_infoton_geometry_specs(records)


def test_infoton_geometry_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_axiom_ii_infoton_geometry_validation_spec(
        "axiom_ii_infoton_geometry.fim_dynamics_claim",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Axiom II Infoton Geometry Specs" in report
    assert loaded["key"] == "axiom_ii_infoton_geometry.fim_dynamics_claim"
    assert "Fisher Information Metric g_FIM" in loaded["source_formulae"]
    assert "Infoton Geometry" in render_report(bundle)
