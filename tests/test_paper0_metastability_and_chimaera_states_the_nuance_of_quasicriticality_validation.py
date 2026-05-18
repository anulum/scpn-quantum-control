# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Metastability and Chimaera States: The Nuance of Quasicriticality validation tests
"""Tests for Paper 0 Metastability and Chimaera States: The Nuance of Quasicriticality source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.metastability_and_chimaera_states_the_nuance_of_quasicriticality_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    MetastabilityAndChimaeraStatesTheNuanceOfQuasicriticalityConfig,
    classify_metastability_and_chimaera_states_the_nuance_of_quasicriticality_component,
    metastability_and_chimaera_states_the_nuance_of_quasicriticality_labels,
    validate_metastability_and_chimaera_states_the_nuance_of_quasicriticality_fixture,
)


def test_metastability_and_chimaera_states_the_nuance_of_quasicriticality_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_metastability_and_chimaera_states_the_nuance_of_quasicriticality_fixture()
    assert result.source_ledger_span == ("P0R04581", "P0R04588")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R04589"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_metastability_and_chimaera_states_the_nuance_of_quasicriticality_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04581"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04588"


def test_metastability_and_chimaera_states_the_nuance_of_quasicriticality_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "metastability_and_chimaera_states_the_nuance_of_quasicriticality",
        "the_dynamic_connectome_functional_reconfiguration_on_a_static_scaffold",
    ):
        assert (
            classify_metastability_and_chimaera_states_the_nuance_of_quasicriticality_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = metastability_and_chimaera_states_the_nuance_of_quasicriticality_labels()
    assert labels["section"] == "Metastability and Chimaera States: The Nuance of Quasicriticality"
    assert labels["next_boundary"] == "P0R04589"


def test_metastability_and_chimaera_states_the_nuance_of_quasicriticality_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        MetastabilityAndChimaeraStatesTheNuanceOfQuasicriticalityConfig(
            expected_source_record_count=7
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        MetastabilityAndChimaeraStatesTheNuanceOfQuasicriticalityConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04589"):
        MetastabilityAndChimaeraStatesTheNuanceOfQuasicriticalityConfig(
            next_source_boundary="P0R04588"
        )
    with pytest.raises(
        ValueError,
        match="unknown metastability_and_chimaera_states_the_nuance_of_quasicriticality component",
    ):
        classify_metastability_and_chimaera_states_the_nuance_of_quasicriticality_component(
            "empirical_validation_claim"
        )
