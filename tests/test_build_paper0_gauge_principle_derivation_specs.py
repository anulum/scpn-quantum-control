# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 gauge-principle derivation builder tests
"""Tests for Paper 0 gauge-principle derivation spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_gauge_principle_derivation_validation_spec,
)
from scripts.build_paper0_gauge_principle_derivation_specs import (
    SOURCE_LEDGER_IDS,
    build_from_ledger,
    build_gauge_principle_derivation_specs,
    render_report,
    write_outputs,
)


def test_gauge_principle_derivation_builder_preserves_full_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R01018", "P0R01077"]
    assert bundle.summary["source_record_count"] == 60
    assert bundle.summary["consumed_source_record_count"] == 60
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 6
    assert bundle.summary["blank_record_count"] == 1
    assert bundle.summary["image_record_count"] == 1
    assert bundle.summary["phenomenology_symmetry_record_count"] == 17
    assert bundle.summary["free_scalar_record_count"] == 12
    assert bundle.summary["local_u1_record_count"] == 7
    assert bundle.summary["covariant_derivative_record_count"] == 14
    assert bundle.summary["fim_dynamics_record_count"] == 10
    assert bundle.summary["next_source_boundary"] == "P0R01078"
    assert [spec.key for spec in bundle.specs] == [
        "gauge_principle_derivation.derivation_boundary",
        "gauge_principle_derivation.phenomenology_symmetry_roadmap",
        "gauge_principle_derivation.free_scalar_global_u1",
        "gauge_principle_derivation.local_u1_derivative_failure",
        "gauge_principle_derivation.covariant_derivative_minimal_coupling",
        "gauge_principle_derivation.fim_gauge_dynamics",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_gauge_principle_derivation_builder_keeps_equations_and_boundaries() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "A Gauge-Principle Derivation of the Psi-Field Interaction Lagrangian"
        in specs["gauge_principle_derivation.derivation_boundary"].source_formulae
    )
    assert (
        "L_Int = L_Geometric + L_Informational"
        in specs["gauge_principle_derivation.phenomenology_symmetry_roadmap"].source_formulae
    )
    assert (
        "L_Geometric = g_PsiG f(Psi) R"
        in specs["gauge_principle_derivation.phenomenology_symmetry_roadmap"].source_formulae
    )
    assert (
        "L_Informational = g_PsiI Psi det(g_mu_nu(x))"
        in specs["gauge_principle_derivation.phenomenology_symmetry_roadmap"].source_formulae
    )
    assert (
        "L_Psi = (partial_mu Psi)* (partial^mu Psi) - V(|Psi|)"
        in specs["gauge_principle_derivation.free_scalar_global_u1"].source_formulae
    )
    assert (
        "Psi(x) -> Psi'(x) = exp(i alpha) Psi(x)"
        in specs["gauge_principle_derivation.free_scalar_global_u1"].source_formulae
    )
    assert (
        "P0R01046 is blank after Noether-current context"
        in specs["gauge_principle_derivation.free_scalar_global_u1"].source_formulae
    )
    assert (
        "Psi(x) -> Psi'(x) = exp(i alpha(x)) Psi(x)"
        in specs["gauge_principle_derivation.local_u1_derivative_failure"].source_formulae
    )
    assert (
        "ordinary derivative failure introduces i(partial_mu alpha(x)) term"
        in specs["gauge_principle_derivation.local_u1_derivative_failure"].source_formulae
    )
    assert (
        "D_mu = partial_mu - i g A_mu"
        in specs[
            "gauge_principle_derivation.covariant_derivative_minimal_coupling"
        ].source_formulae
    )
    assert (
        "A_mu' = A_mu + (1/g) partial_mu alpha(x)"
        in specs[
            "gauge_principle_derivation.covariant_derivative_minimal_coupling"
        ].source_formulae
    )
    assert (
        "minimal coupling is an unavoidable consequence of local phase invariance"
        in specs[
            "gauge_principle_derivation.covariant_derivative_minimal_coupling"
        ].source_formulae
    )
    assert (
        "F_mu_nu = partial_mu A_nu - partial_nu A_mu"
        in specs["gauge_principle_derivation.fim_gauge_dynamics"].source_formulae
    )
    assert (
        "L_Informational' = L_Interaction + L_gauge"
        in specs["gauge_principle_derivation.fim_gauge_dynamics"].source_formulae
    )
    assert (
        "P0R01076 is image placeholder not validation evidence"
        in specs["gauge_principle_derivation.fim_gauge_dynamics"].source_formulae
    )
    assert (
        "next boundary is P0R01078 Lorentz covariance EFT resolution"
        in specs["gauge_principle_derivation.fim_gauge_dynamics"].source_formulae
    )


def test_gauge_principle_derivation_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R01056":
            continue
        records.append(
            {
                "ledger_id": record_id,
                "source_record_id": f"{record_id}:stub",
                "source_block_index": int(record_id[3:]),
                "section_path": "Paper 0 > Gauge Principle Derivation",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_gauge_principle_derivation_specs(records)


def test_gauge_principle_derivation_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_gauge_principle_derivation_validation_spec(
        "gauge_principle_derivation.covariant_derivative_minimal_coupling",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Gauge Principle Derivation Specs" in report
    assert loaded["key"] == "gauge_principle_derivation.covariant_derivative_minimal_coupling"
    assert "D_mu = partial_mu - i g A_mu" in loaded["source_formulae"]
    assert "Gauge Principle Derivation" in render_report(bundle)
