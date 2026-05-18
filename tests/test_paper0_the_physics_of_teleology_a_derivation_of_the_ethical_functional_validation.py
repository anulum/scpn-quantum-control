# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Physics of Teleology: A Derivation of the Ethical Functional validation tests
"""Tests for Paper 0 The Physics of Teleology: A Derivation of the Ethical Functional source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_physics_of_teleology_a_derivation_of_the_ethical_functional_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ThePhysicsOfTeleologyADerivationOfTheEthicalFunctionalConfig,
    classify_the_physics_of_teleology_a_derivation_of_the_ethical_functional_component,
    the_physics_of_teleology_a_derivation_of_the_ethical_functional_labels,
    validate_the_physics_of_teleology_a_derivation_of_the_ethical_functional_fixture,
)


def test_the_physics_of_teleology_a_derivation_of_the_ethical_functional_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_the_physics_of_teleology_a_derivation_of_the_ethical_functional_fixture()
    assert result.source_ledger_span == ("P0R03581", "P0R03602")
    assert result.source_record_count == 22
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R03603"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_physics_of_teleology_a_derivation_of_the_ethical_functional_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03581"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03602"


def test_the_physics_of_teleology_a_derivation_of_the_ethical_functional_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("the_physics_of_teleology_a_derivation_of_the_ethical_functional",):
        assert (
            classify_the_physics_of_teleology_a_derivation_of_the_ethical_functional_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = the_physics_of_teleology_a_derivation_of_the_ethical_functional_labels()
    assert labels["section"] == "The Physics of Teleology: A Derivation of the Ethical Functional"
    assert labels["next_boundary"] == "P0R03603"


def test_the_physics_of_teleology_a_derivation_of_the_ethical_functional_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 22"):
        ThePhysicsOfTeleologyADerivationOfTheEthicalFunctionalConfig(
            expected_source_record_count=21
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        ThePhysicsOfTeleologyADerivationOfTheEthicalFunctionalConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03603"):
        ThePhysicsOfTeleologyADerivationOfTheEthicalFunctionalConfig(
            next_source_boundary="P0R03602"
        )
    with pytest.raises(
        ValueError,
        match="unknown the_physics_of_teleology_a_derivation_of_the_ethical_functional component",
    ):
        classify_the_physics_of_teleology_a_derivation_of_the_ethical_functional_component(
            "empirical_validation_claim"
        )
