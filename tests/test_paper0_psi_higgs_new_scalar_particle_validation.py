# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Psi-Higgs scalar validation tests
"""Tests for Paper 0 Psi-Higgs new-scalar-particle validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.psi_higgs_new_scalar_particle_validation import (
    PsiHiggsNewScalarParticleConfig,
    classify_psi_higgs_new_scalar_particle_component,
    psi_higgs_new_scalar_particle_labels,
    validate_psi_higgs_new_scalar_particle_fixture,
)


def test_psi_higgs_new_scalar_particle_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        PsiHiggsNewScalarParticleConfig(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        PsiHiggsNewScalarParticleConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01647"):
        PsiHiggsNewScalarParticleConfig(next_source_boundary="P0R01646")


def test_psi_higgs_new_scalar_particle_classifiers_are_source_bounded() -> None:
    assert (
        classify_psi_higgs_new_scalar_particle_component("scalar_remnant_identity")
        == "psi_higgs_scalar_remnant_identity_boundary"
    )
    assert (
        classify_psi_higgs_new_scalar_particle_component("potential_mass_term")
        == "psi_higgs_potential_mass_term_source_boundary"
    )
    assert (
        classify_psi_higgs_new_scalar_particle_component("mass_and_detection_boundary")
        == "psi_higgs_mass_relation_future_discovery_boundary"
    )
    with pytest.raises(ValueError, match="unknown Psi-Higgs new-scalar-particle component"):
        classify_psi_higgs_new_scalar_particle_component("lhc_signature")


def test_psi_higgs_new_scalar_particle_fixture_preserves_claim_boundary() -> None:
    result = validate_psi_higgs_new_scalar_particle_fixture()

    assert result.source_ledger_span == ("P0R01638", "P0R01646")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 9
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R01647"
    assert result.null_controls == {
        "scalar_remnant_identity_is_not_particle_detection": 1.0,
        "potential_mass_term_is_not_collider_mass_reconstruction": 1.0,
        "future_discovery_clause_is_not_current_evidence": 1.0,
    }
    assert (
        result.problem_metadata["protocol_state"]
        == "source_psi_higgs_new_scalar_particle_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R01638"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R01646"


def test_psi_higgs_new_scalar_particle_labels_name_next_signature_boundary() -> None:
    labels = psi_higgs_new_scalar_particle_labels()

    assert labels["section"] == "The Psi-Higgs Boson: A New Scalar Particle"
    assert labels["identity"] == "h(x) radial fluctuation of the Psi-field"
    assert labels["mass_term"] == "L_mass,h = -lambda v^2 h^2"
    assert labels["mass"] == "m_h = sqrt(2 lambda) v"
    assert labels["next_boundary"] == "Experimental Signatures and Search Strategies"
