# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 infoton-properties derivation validation tests
"""Tests for Paper 0 infoton-properties derivation validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.derivation_infoton_properties_validation import (
    DerivationInfotonPropertiesConfig,
    classify_derivation_infoton_properties_component,
    derivation_infoton_properties_labels,
    validate_derivation_infoton_properties_fixture,
)


def test_derivation_infoton_properties_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 15"):
        DerivationInfotonPropertiesConfig(expected_source_record_count=14)
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        DerivationInfotonPropertiesConfig(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01638"):
        DerivationInfotonPropertiesConfig(next_source_boundary="P0R01637")


def test_derivation_infoton_properties_classifiers_are_source_bounded() -> None:
    assert (
        classify_derivation_infoton_properties_component("lagrangian_and_potential")
        == "u1_lagrangian_potential_derivation_boundary"
    )
    assert (
        classify_derivation_infoton_properties_component("vev_and_goldstone_absorption")
        == "vev_goldstone_absorption_source_boundary"
    )
    assert (
        classify_derivation_infoton_properties_component("mass_identification")
        == "infoton_mass_identification_source_boundary"
    )
    assert (
        classify_derivation_infoton_properties_component("range_consequence")
        == "short_range_informational_force_consequence_boundary"
    )
    with pytest.raises(ValueError, match="unknown infoton-properties derivation component"):
        classify_derivation_infoton_properties_component("psi_higgs_particle")


def test_derivation_infoton_properties_fixture_preserves_claim_boundary() -> None:
    result = validate_derivation_infoton_properties_fixture()

    assert result.source_ledger_span == ("P0R01623", "P0R01637")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 15
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R01638"
    assert result.null_controls == {
        "lagrangian_derivation_is_not_detector_evidence": 1.0,
        "goldstone_absorption_is_source_derivation_not_observed_event": 1.0,
        "infoton_mass_relation_is_not_measured_mass": 1.0,
        "force_range_relation_is_not_measured_force_profile": 1.0,
    }
    assert (
        result.problem_metadata["protocol_state"]
        == "source_derivation_infoton_properties_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R01623"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R01637"


def test_derivation_infoton_properties_labels_name_next_psi_higgs_boundary() -> None:
    labels = derivation_infoton_properties_labels()

    assert labels["section"] == "Derivation of the Infoton's Properties"
    assert labels["lagrangian"] == "L = |D_mu Psi|^2 - V(|Psi|) - 1/4 F_mu_nu F^mu_nu"
    assert labels["potential"] == "V(|Psi|) = -mu^2 |Psi|^2 + lambda |Psi|^4"
    assert labels["mass"] == "m_A = g v"
    assert labels["range"] == "lambda_range ~= hbar / (m_A c)"
    assert labels["next_boundary"] == "The Psi-Higgs Boson: A New Scalar Particle"
