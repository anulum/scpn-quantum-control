# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Paper 0 equation register tests
"""Tests for the Paper 0 source-anchored equation register."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.equation_register import (
    Paper0EquationRecord,
    get_paper0_equation_record,
    iter_paper0_equation_records,
    paper0_upde_records,
)


def test_upde_base_record_is_source_anchored_and_parameter_complete() -> None:
    record = get_paper0_equation_record("upde.base_phase")

    assert isinstance(record, Paper0EquationRecord)
    assert record.source_equation_ids == ("EQ0003", "EQ0032", "EQ0037", "EQ0039", "EQ0129")
    assert "K_{ij}^{L}" in record.canonical_latex
    assert "C_{\\mathrm{InterLayer}}" in record.canonical_latex
    assert "C_{\\mathrm{Field}}" in record.canonical_latex
    assert set(record.variables) >= {
        "theta_i_L",
        "omega_i_L",
        "K_ij_L",
        "C_interlayer",
        "C_field",
        "eta_i_L",
    }
    assert "finite oscillator layer" in record.assumptions
    assert "UPDE-to-XY gradient check" in record.validation_targets


def test_upde_family_register_contains_required_mechanism_records() -> None:
    records = paper0_upde_records()
    keys = {record.key for record in records}

    assert keys == {
        "upde.base_phase",
        "upde.interlayer_coupling",
        "upde.field_coupling",
        "upde.natural_gradient",
        "upde.adaptive_coupling",
    }
    for record in records:
        assert record.manuscript == "Paper 0 foundational framework"
        assert record.section_path
        assert "=" in record.canonical_latex
        assert "\\" in record.canonical_latex
        assert record.source_equation_ids
        assert record.variables
        assert record.validation_targets


def test_macro_transition_records_are_source_anchored() -> None:
    spin_glass = get_paper0_equation_record("nths.spin_glass_hamiltonian")
    rg_flow = get_paper0_equation_record("macro_transition.effective_coupling_rg")

    assert spin_glass.source_equation_ids == ("EQ0113",)
    assert "J_{ij}" in spin_glass.canonical_latex
    assert "sigma_i" in spin_glass.variables
    assert "Edwards-Anderson" in " ".join(spin_glass.validation_targets)
    assert "spin glass" in spin_glass.themes

    assert rg_flow.source_equation_ids == ("EQ0114",)
    assert "\\mu\\frac{dK_{\\mathrm{eff}}}{d\\mu}" in rg_flow.canonical_latex
    assert "K_eff" in rg_flow.variables
    assert "fixed point" in " ".join(rg_flow.validation_targets)
    assert "renormalisation group" in rg_flow.themes


def test_neurovascular_record_is_source_anchored() -> None:
    record = get_paper0_equation_record("embodied.neurovascular_phase_coupling")

    assert record.source_equation_ids == ("EQ0093",)
    assert "\\dot{\\theta}_{\\mathrm{Neural}}" in record.canonical_latex
    assert "K_{NH}" in record.canonical_latex
    assert set(record.variables) >= {
        "theta_neural",
        "theta_hemo",
        "omega_N",
        "K_NH",
    }
    assert "Mayer-wave" in " ".join(record.validation_targets)
    assert "neurovascular" in record.themes


def test_glial_and_immune_records_are_source_anchored() -> None:
    immune = get_paper0_equation_record("embodied.quantum_immune_interface")
    glial = get_paper0_equation_record("embodied.glial_sigma_control")

    assert immune.source_equation_ids == ("EQ0105",)
    assert "H_{\\mathrm{int}}" in immune.canonical_latex
    assert set(immune.variables) >= {"lambda", "Psi_s", "C_cyto", "sigma_x"}
    assert "cytokine" in " ".join(immune.validation_targets)
    assert "immune-interface" in immune.themes

    assert glial.source_equation_ids == (
        "EQ0106",
        "EQ0107",
        "EQ0108",
        "EQ0109",
        "EQ0110",
        "EQ0111",
        "EQ0112",
    )
    assert "\\dot{\\sigma}" in glial.canonical_latex
    assert "\\dot{G}" in glial.canonical_latex
    assert set(glial.variables) >= {"sigma", "G", "Ca_A", "kappa", "alpha", "beta"}
    assert "gliotransmitter blockade" in " ".join(glial.validation_targets)
    assert "glial-control" in glial.themes


def test_iter_records_filters_by_theme_without_synthetic_entries() -> None:
    upde = list(iter_paper0_equation_records(theme="UPDE"))
    fim = list(iter_paper0_equation_records(theme="FIM"))
    macro = list(iter_paper0_equation_records(theme="macro-transition"))
    neurovascular = list(iter_paper0_equation_records(theme="neurovascular"))
    glial = list(iter_paper0_equation_records(theme="glial-control"))

    assert {record.key for record in upde} >= {
        "upde.base_phase",
        "upde.natural_gradient",
    }
    assert {record.key for record in fim} == {"upde.natural_gradient"}
    assert all(record.provenance_status == "paper0_extracted" for record in upde)
    assert {record.key for record in macro} == {
        "nths.spin_glass_hamiltonian",
        "macro_transition.effective_coupling_rg",
    }
    assert {record.key for record in neurovascular} == {
        "embodied.neurovascular_phase_coupling",
    }
    assert {record.key for record in glial} == {
        "embodied.glial_sigma_control",
    }


def test_get_record_rejects_unknown_key() -> None:
    with pytest.raises(KeyError, match="unknown Paper 0 equation record"):
        get_paper0_equation_record("paper27.synthetic_matrix")
