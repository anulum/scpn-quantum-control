# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 II. The Principle of Ethical Least Action (PELA) validation tests
"""Tests for Paper 0 II. The Principle of Ethical Least Action (PELA) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.ii_the_principle_of_ethical_least_action_pela_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    IiThePrincipleOfEthicalLeastActionPelaConfig,
    classify_ii_the_principle_of_ethical_least_action_pela_component,
    ii_the_principle_of_ethical_least_action_pela_labels,
    validate_ii_the_principle_of_ethical_least_action_pela_fixture,
)


def test_ii_the_principle_of_ethical_least_action_pela_fixture_preserves_source_boundary() -> None:
    result = validate_ii_the_principle_of_ethical_least_action_pela_fixture()
    assert result.source_ledger_span == ("P0R04029", "P0R04074")
    assert result.source_record_count == 46
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R04075"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_ii_the_principle_of_ethical_least_action_pela_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04029"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04074"


def test_ii_the_principle_of_ethical_least_action_pela_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("ii_the_principle_of_ethical_least_action_pela",):
        assert (
            classify_ii_the_principle_of_ethical_least_action_pela_component(component)
            == f"{component}_source_boundary"
        )
    labels = ii_the_principle_of_ethical_least_action_pela_labels()
    assert labels["section"] == "II. The Principle of Ethical Least Action (PELA)"
    assert labels["next_boundary"] == "P0R04075"


def test_ii_the_principle_of_ethical_least_action_pela_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 46"):
        IiThePrincipleOfEthicalLeastActionPelaConfig(expected_source_record_count=45)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        IiThePrincipleOfEthicalLeastActionPelaConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04075"):
        IiThePrincipleOfEthicalLeastActionPelaConfig(next_source_boundary="P0R04074")
    with pytest.raises(
        ValueError, match="unknown ii_the_principle_of_ethical_least_action_pela component"
    ):
        classify_ii_the_principle_of_ethical_least_action_pela_component(
            "empirical_validation_claim"
        )
