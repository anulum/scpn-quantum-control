# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Integration with Integrated Information Theory (IIT) 4.0 validation tests
"""Tests for Paper 0 Integration with Integrated Information Theory (IIT) 4.0 source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.integration_with_integrated_information_theory_iit_4_0_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    IntegrationWithIntegratedInformationTheoryIit40Config,
    classify_integration_with_integrated_information_theory_iit_4_0_component,
    integration_with_integrated_information_theory_iit_4_0_labels,
    validate_integration_with_integrated_information_theory_iit_4_0_fixture,
)


def test_integration_with_integrated_information_theory_iit_4_0_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_integration_with_integrated_information_theory_iit_4_0_fixture()
    assert result.source_ledger_span == ("P0R03521", "P0R03529")
    assert result.source_record_count == 9
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R03530"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_integration_with_integrated_information_theory_iit_4_0_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03521"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03529"


def test_integration_with_integrated_information_theory_iit_4_0_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("integration_with_integrated_information_theory_iit_4_0",):
        assert (
            classify_integration_with_integrated_information_theory_iit_4_0_component(component)
            == f"{component}_source_boundary"
        )
    labels = integration_with_integrated_information_theory_iit_4_0_labels()
    assert labels["section"] == "Integration with Integrated Information Theory (IIT) 4.0"
    assert labels["next_boundary"] == "P0R03530"


def test_integration_with_integrated_information_theory_iit_4_0_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        IntegrationWithIntegratedInformationTheoryIit40Config(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        IntegrationWithIntegratedInformationTheoryIit40Config(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03530"):
        IntegrationWithIntegratedInformationTheoryIit40Config(next_source_boundary="P0R03529")
    with pytest.raises(
        ValueError,
        match="unknown integration_with_integrated_information_theory_iit_4_0 component",
    ):
        classify_integration_with_integrated_information_theory_iit_4_0_component(
            "empirical_validation_claim"
        )
