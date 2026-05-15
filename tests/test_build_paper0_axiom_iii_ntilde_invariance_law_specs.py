# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom III Ntilde invariance-law builder tests
"""Tests for Paper 0 Axiom III Ntilde-invariance-law spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_axiom_iii_ntilde_invariance_law_validation_spec,
)
from scripts.build_paper0_axiom_iii_ntilde_invariance_law_specs import (
    SOURCE_LEDGER_IDS,
    build_axiom_iii_ntilde_invariance_law_specs,
    build_from_ledger,
    render_report,
    write_outputs,
)


def test_ntilde_invariance_law_builder_preserves_contiguous_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R00800", "P0R00810"]
    assert bundle.summary["source_record_count"] == 11
    assert bundle.summary["consumed_source_record_count"] == 11
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 4
    assert bundle.summary["invariant_definition_count"] == 3
    assert bundle.summary["variable_definition_count"] == 3
    assert bundle.summary["threshold_equation_count"] == 1
    assert bundle.summary["reversible_limit_count"] == 1
    assert bundle.summary["next_source_boundary"] == "P0R00811"
    assert [spec.key for spec in bundle.specs] == [
        "axiom_iii_ntilde_invariance_law.physical_law_identification",
        "axiom_iii_ntilde_invariance_law.invariant_ratio_equation",
        "axiom_iii_ntilde_invariance_law.variable_definitions",
        "axiom_iii_ntilde_invariance_law.unity_threshold_limit",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_ntilde_invariance_law_builder_keeps_equations_and_boundaries() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "Axiom III is a fundamental falsifiable physical law"
        in specs["axiom_iii_ntilde_invariance_law.physical_law_identification"].source_formulae
    )
    assert (
        "dimensionless measurable invariant linking energy information and time"
        in specs["axiom_iii_ntilde_invariance_law.physical_law_identification"].source_formulae
    )
    assert (
        "tilde_N_t = P / (epsilon_b dot_I) = (E/t) / ((Delta F_rev / Delta I) dot_I)"
        in specs["axiom_iii_ntilde_invariance_law.invariant_ratio_equation"].source_formulae
    )
    assert (
        "actual power energy flux over minimum reversible free-energy cost for information flow"
        in specs["axiom_iii_ntilde_invariance_law.invariant_ratio_equation"].source_formulae
    )
    assert (
        "P = E/t is actual power or energy flux"
        in specs["axiom_iii_ntilde_invariance_law.variable_definitions"].source_formulae
    )
    assert (
        "dot_I is rate of reliably processed information bit/s"
        in specs["axiom_iii_ntilde_invariance_law.variable_definitions"].source_formulae
    )
    assert (
        "epsilon_b = Delta F_rev / Delta I is reversible free-energy cost per bit"
        in specs["axiom_iii_ntilde_invariance_law.variable_definitions"].source_formulae
    )
    assert (
        "tilde_N_t -> 1"
        in specs["axiom_iii_ntilde_invariance_law.unity_threshold_limit"].source_formulae
    )
    assert (
        "tilde_N_t = 1 is the reversible limit"
        in specs["axiom_iii_ntilde_invariance_law.unity_threshold_limit"].source_formulae
    )
    assert (
        "maximum thermodynamic efficiency minimum irreversibility or entropy production"
        in specs["axiom_iii_ntilde_invariance_law.unity_threshold_limit"].source_formulae
    )


def test_ntilde_invariance_law_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R00803":
            continue
        records.append(
            {
                "ledger_id": record_id,
                "source_record_id": f"{record_id}:stub",
                "source_block_index": int(record_id[3:]),
                "section_path": "Paper 0 > Axiom III > Ntilde Invariance Law",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_axiom_iii_ntilde_invariance_law_specs(records)


def test_ntilde_invariance_law_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_axiom_iii_ntilde_invariance_law_validation_spec(
        "axiom_iii_ntilde_invariance_law.invariant_ratio_equation",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Axiom III Ntilde Invariance Law Specs" in report
    assert loaded["key"] == "axiom_iii_ntilde_invariance_law.invariant_ratio_equation"
    assert (
        "tilde_N_t = P / (epsilon_b dot_I) = (E/t) / ((Delta F_rev / Delta I) dot_I)"
        in loaded["source_formulae"]
    )
    assert "Ntilde Invariance Law" in render_report(bundle)
