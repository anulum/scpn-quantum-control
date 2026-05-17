# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 foundational strengths phase boundary builder tests
"""Tests for Paper 0 foundational-strengths and phase-boundary spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_foundational_strengths_phase_boundary_validation_spec,
)
from scripts.build_paper0_foundational_strengths_phase_boundary_specs import (
    SOURCE_LEDGER_IDS,
    build_foundational_strengths_phase_boundary_specs,
    build_from_ledger,
    render_report,
    write_outputs,
)


def test_foundational_strengths_phase_boundary_builder_preserves_full_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R01189", "P0R01241"]
    assert bundle.summary["source_record_count"] == 53
    assert bundle.summary["consumed_source_record_count"] == 53
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 6
    assert bundle.summary["math_ids"] == []
    assert bundle.summary["image_ids"] == []
    assert bundle.summary["table_ids"] == []
    assert bundle.summary["next_source_boundary"] == "P0R01242"
    assert [spec.key for spec in bundle.specs] == [
        "foundational_strengths_phase_boundary.foundational_strengths",
        "foundational_strengths_phase_boundary.architecture_integration",
        "foundational_strengths_phase_boundary.future_research_and_parameter_constraints",
        "foundational_strengths_phase_boundary.modulus_phase_decomposition",
        "foundational_strengths_phase_boundary.axion_analogy_and_em_interface",
        "foundational_strengths_phase_boundary.gauge_choice_and_kinetic_phase_boundary",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_foundational_strengths_phase_boundary_builder_keeps_equations_and_boundaries() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "massless spin-1 informational gauge boson remains a prediction target"
        in specs["foundational_strengths_phase_boundary.foundational_strengths"].source_formulae
    )
    assert (
        "15-layer SCPN architecture is claimed to be strengthened by the interaction Lagrangian"
        in specs["foundational_strengths_phase_boundary.architecture_integration"].source_formulae
    )
    assert (
        "m_A = g v for the Infoton"
        in specs[
            "foundational_strengths_phase_boundary.future_research_and_parameter_constraints"
        ].source_formulae
    )
    assert (
        "V(Psi) = -(mu^2/2)|Psi|^2 + (lambda/4)|Psi|^4, mu^2 > 0"
        in specs[
            "foundational_strengths_phase_boundary.modulus_phase_decomposition"
        ].source_formulae
    )
    assert (
        "L_a_gamma_gamma = g_a_gamma_gamma a F_mu_nu F_tilde^mu_nu = g_a_gamma_gamma a (E dot B)"
        in specs[
            "foundational_strengths_phase_boundary.axion_analogy_and_em_interface"
        ].source_formulae
    )
    assert (
        "rho_kinetic_phase = one-half R^2 theta_dot^2"
        in specs[
            "foundational_strengths_phase_boundary.gauge_choice_and_kinetic_phase_boundary"
        ].source_formulae
    )


def test_foundational_strengths_phase_boundary_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R01231":
            continue
        records.append(
            {
                "ledger_id": record_id,
                "source_record_id": f"{record_id}:stub",
                "source_block_index": int(record_id[3:]),
                "canonical_category": "mechanism",
                "block_type": "Para",
                "math_ids": [],
                "image_ids": [],
                "table_id": None,
                "section_path": "Paper 0 > Foundational Strengths Phase Boundary",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_foundational_strengths_phase_boundary_specs(records)


def test_foundational_strengths_phase_boundary_outputs_and_loader_round_trip(
    tmp_path: Path,
) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_foundational_strengths_phase_boundary_validation_spec(
        "foundational_strengths_phase_boundary.modulus_phase_decomposition",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Foundational Strengths Phase Boundary Specs" in report
    assert loaded["key"] == "foundational_strengths_phase_boundary.modulus_phase_decomposition"
    assert "Psi(x) = (v + h(x)) exp(i theta(x))" in loaded["source_formulae"]
    assert "Foundational Strengths Phase Boundary" in render_report(bundle)
