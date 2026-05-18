# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 III. The Dynamics of the Coherent Brain (Domain I: L4) validation tests
"""Tests for Paper 0 III. The Dynamics of the Coherent Brain (Domain I: L4) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.iii_the_dynamics_of_the_coherent_brain_domain_i_l4_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    IiiTheDynamicsOfTheCoherentBrainDomainIL4Config,
    classify_iii_the_dynamics_of_the_coherent_brain_domain_i_l4_component,
    iii_the_dynamics_of_the_coherent_brain_domain_i_l4_labels,
    validate_iii_the_dynamics_of_the_coherent_brain_domain_i_l4_fixture,
)


def test_iii_the_dynamics_of_the_coherent_brain_domain_i_l4_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_iii_the_dynamics_of_the_coherent_brain_domain_i_l4_fixture()
    assert result.source_ledger_span == ("P0R04666", "P0R04673")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04674"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_iii_the_dynamics_of_the_coherent_brain_domain_i_l4_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04666"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04673"


def test_iii_the_dynamics_of_the_coherent_brain_domain_i_l4_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "iii_the_dynamics_of_the_coherent_brain_domain_i_l4",
        "1_the_upde_and_oscillatory_hierarchies",
        "2_the_mechanisms_of_self_organised_criticality_soc",
    ):
        assert (
            classify_iii_the_dynamics_of_the_coherent_brain_domain_i_l4_component(component)
            == f"{component}_source_boundary"
        )
    labels = iii_the_dynamics_of_the_coherent_brain_domain_i_l4_labels()
    assert labels["section"] == "III. The Dynamics of the Coherent Brain (Domain I: L4)"
    assert labels["next_boundary"] == "P0R04674"


def test_iii_the_dynamics_of_the_coherent_brain_domain_i_l4_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        IiiTheDynamicsOfTheCoherentBrainDomainIL4Config(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        IiiTheDynamicsOfTheCoherentBrainDomainIL4Config(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04674"):
        IiiTheDynamicsOfTheCoherentBrainDomainIL4Config(next_source_boundary="P0R04673")
    with pytest.raises(
        ValueError, match="unknown iii_the_dynamics_of_the_coherent_brain_domain_i_l4 component"
    ):
        classify_iii_the_dynamics_of_the_coherent_brain_domain_i_l4_component(
            "empirical_validation_claim"
        )
