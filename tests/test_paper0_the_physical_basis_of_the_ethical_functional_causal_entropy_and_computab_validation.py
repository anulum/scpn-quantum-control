# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Physical Basis of the Ethical Functional: Causal Entropy and Computable Qualia validation tests
"""Tests for Paper 0 The Physical Basis of the Ethical Functional: Causal Entropy and Computable Qualia source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ThePhysicalBasisOfTheEthicalFunctionalCausalEntropyAndComputabConfig,
    classify_the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_component,
    the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_labels,
    validate_the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_fixture,
)


def test_the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_fixture()
    )
    assert result.source_ledger_span == ("P0R04115", "P0R04122")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R04123"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04115"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04122"


def test_the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab",
        "meta_framework_integrations",
    ):
        assert (
            classify_the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_labels()
    assert (
        labels["section"]
        == "The Physical Basis of the Ethical Functional: Causal Entropy and Computable Qualia"
    )
    assert labels["next_boundary"] == "P0R04123"


def test_the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        ThePhysicalBasisOfTheEthicalFunctionalCausalEntropyAndComputabConfig(
            expected_source_record_count=7
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        ThePhysicalBasisOfTheEthicalFunctionalCausalEntropyAndComputabConfig(
            expected_component_count=3
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04123"):
        ThePhysicalBasisOfTheEthicalFunctionalCausalEntropyAndComputabConfig(
            next_source_boundary="P0R04122"
        )
    with pytest.raises(
        ValueError,
        match="unknown the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab component",
    ):
        classify_the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_component(
            "empirical_validation_claim"
        )
