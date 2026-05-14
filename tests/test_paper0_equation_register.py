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


def test_computational_unifier_records_are_source_anchored() -> None:
    cyclic = get_paper0_equation_record("computational.cyclic_operator_boundary")
    tsvf = get_paper0_equation_record("computational.tsvf_abl_boundary")
    record = get_paper0_equation_record("computational.info_thermodynamics")
    iit_or = get_paper0_equation_record("computational.iit_or_threshold")
    noether = get_paper0_equation_record("computational.coherence_noether_current")
    iet = get_paper0_equation_record("computational.information_energy_transduction")

    assert cyclic.source_equation_ids == ("EQ0115",)
    assert "O_{\\mathrm{MMC}}" in cyclic.canonical_latex
    assert "boundary-only" in " ".join(cyclic.validation_targets)
    assert "temporal-boundary" in cyclic.themes

    assert tsvf.source_equation_ids == ("EQ0116",)
    assert "P(A=a|t)" in tsvf.canonical_latex
    assert "normalisation" in " ".join(tsvf.validation_targets)
    assert "temporal-boundary" in tsvf.themes

    assert record.source_equation_ids == ("EQ0117", "EQ0118")
    assert "\\frac{dS_{\\mathrm{Total}}}{dt}" in record.canonical_latex
    assert "I(\\Psi;B)" in record.canonical_latex
    assert set(record.variables) >= {
        "S_total",
        "S_thermo",
        "S_info",
        "N_Psi",
        "I_Psi_B",
    }
    assert "Landauer" in " ".join(record.validation_targets)
    assert "information-thermodynamics" in record.themes

    assert iit_or.source_equation_ids == ("EQ0119",)
    assert "E_{\\Phi}" in iit_or.canonical_latex
    assert "threshold crossing" in " ".join(iit_or.validation_targets)
    assert "classifier-boundary" in iit_or.themes

    assert noether.source_equation_ids == ("EQ0120",)
    assert "J_{\\Psi}^{\\mu}" in noether.canonical_latex
    assert "\\partial_{\\mu}J_{\\Psi}^{\\mu}=0" in noether.canonical_latex
    assert "Noether" in noether.themes

    assert iet.source_equation_ids == ("EQ0121", "EQ0122")
    assert "\\nabla^2\\sqrt{\\rho}" in iet.canonical_latex
    assert "constant-density zero-potential" in " ".join(iet.validation_targets)
    assert "quantum-potential" in iet.themes


def test_ethical_gauge_records_are_source_anchored() -> None:
    action = get_paper0_equation_record("computational.ethical_yang_mills_action")
    boundary = get_paper0_equation_record("computational.ethical_connection_boundary")
    cef = get_paper0_equation_record("computational.causal_entropic_force")

    assert action.source_equation_ids == ("EQ0123", "EQ0124")
    assert "\\mathrm{Tr}(F\\wedge\\star F)" in action.canonical_latex
    assert "\\delta S_{\\mathrm{Ethical}}=0" in action.canonical_latex
    assert "gauge-action-boundary" in action.themes

    assert boundary.source_equation_ids == ("EQ0125", "EQ0126", "EQ0127")
    assert "D^{\\dagger}F=J_{\\mathrm{CEF}}" in boundary.canonical_latex
    assert "\\Phi_{\\partial M}" in boundary.canonical_latex
    assert "boundary-flux" in " ".join(boundary.validation_targets)
    assert "ethical-connection" in boundary.themes

    assert cef.source_equation_ids == ("EQ0128",)
    assert "F_{\\mathrm{Causal}}" in cef.canonical_latex
    assert "causal entropy" in " ".join(cef.validation_targets)
    assert "CEF" in cef.themes


def test_iter_records_filters_by_theme_without_synthetic_entries() -> None:
    upde = list(iter_paper0_equation_records(theme="UPDE"))
    fim = list(iter_paper0_equation_records(theme="FIM"))
    macro = list(iter_paper0_equation_records(theme="macro-transition"))
    neurovascular = list(iter_paper0_equation_records(theme="neurovascular"))
    glial = list(iter_paper0_equation_records(theme="glial-control"))
    info_thermo = list(iter_paper0_equation_records(theme="information-thermodynamics"))
    temporal_boundary = list(iter_paper0_equation_records(theme="temporal-boundary"))
    noether = list(iter_paper0_equation_records(theme="Noether"))
    quantum_potential = list(iter_paper0_equation_records(theme="quantum-potential"))
    ethical_connection = list(iter_paper0_equation_records(theme="ethical-connection"))
    cef = list(iter_paper0_equation_records(theme="CEF"))

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
    assert {record.key for record in info_thermo} == {
        "computational.info_thermodynamics",
    }
    assert {record.key for record in temporal_boundary} == {
        "computational.cyclic_operator_boundary",
        "computational.tsvf_abl_boundary",
    }
    assert {record.key for record in noether} == {
        "computational.coherence_noether_current",
    }
    assert {record.key for record in quantum_potential} == {
        "computational.information_energy_transduction",
    }
    assert {record.key for record in ethical_connection} == {
        "computational.ethical_connection_boundary",
    }
    assert {record.key for record in cef} == {
        "computational.causal_entropic_force",
    }


def test_get_record_rejects_unknown_key() -> None:
    with pytest.raises(KeyError, match="unknown Paper 0 equation record"):
        get_paper0_equation_record("paper27.synthetic_matrix")
