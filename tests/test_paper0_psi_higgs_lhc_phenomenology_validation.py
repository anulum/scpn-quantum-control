# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Psi-Higgs LHC phenomenology validation tests
"""Tests for Paper 0 Psi-Higgs LHC phenomenology validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.psi_higgs_lhc_phenomenology_validation import (
    PsiHiggsLHCPhenomenologyConfig,
    classify_psi_higgs_lhc_phenomenology_component,
    psi_higgs_lhc_phenomenology_labels,
    validate_psi_higgs_lhc_phenomenology_fixture,
)


def test_psi_higgs_lhc_phenomenology_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 14"):
        PsiHiggsLHCPhenomenologyConfig(expected_source_record_count=13)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        PsiHiggsLHCPhenomenologyConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01669"):
        PsiHiggsLHCPhenomenologyConfig(next_source_boundary="P0R01668")


def test_psi_higgs_lhc_phenomenology_classifiers_are_source_bounded() -> None:
    assert (
        classify_psi_higgs_lhc_phenomenology_component("phenomenology_bridge")
        == "psi_higgs_lhc_phenomenology_bridge_boundary"
    )
    assert (
        classify_psi_higgs_lhc_phenomenology_component("scalar_mixing_mechanism")
        == "psi_higgs_sm_higgs_scalar_mixing_boundary"
    )
    assert (
        classify_psi_higgs_lhc_phenomenology_component("scalar_potential_and_cross_term")
        == "higgs_portal_potential_cross_term_boundary"
    )
    with pytest.raises(ValueError, match="unknown Psi-Higgs LHC phenomenology component"):
        classify_psi_higgs_lhc_phenomenology_component("mass_eigenstates")


def test_psi_higgs_lhc_phenomenology_fixture_preserves_claim_boundary() -> None:
    result = validate_psi_higgs_lhc_phenomenology_fixture()

    assert result.source_ledger_span == ("P0R01655", "P0R01668")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 14
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R01669"
    assert result.null_controls == {
        "lhc_phenomenology_bridge_is_not_observed_lhc_signal": 1.0,
        "scalar_mixing_claim_is_not_measured_higgs_admixture": 1.0,
        "higgs_portal_potential_is_not_fitted_collider_model": 1.0,
    }
    assert (
        result.problem_metadata["protocol_state"]
        == "source_psi_higgs_lhc_phenomenology_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R01655"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R01668"


def test_psi_higgs_lhc_phenomenology_labels_name_next_mixing_angle_boundary() -> None:
    labels = psi_higgs_lhc_phenomenology_labels()

    assert (
        labels["section"]
        == "The Psi-Higgs Boson: Phenomenology and Experimental Signatures at the LHC"
    )
    assert labels["mechanism"] == "Psi-Higgs mechanism and scalar mixing"
    assert labels["portal"] == "V_mix = lambda_mix (H^dagger H) |Psi|^2"
    assert labels["cross_term"] == "lambda_mix v_h v_psi h_bare h_Psi,bare"
    assert labels["next_boundary"] == "Mass Eigenstates and the Mixing Angle (theta)"
