# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Formalism of the Homeostatic Quasicritical Controller validation tests
"""Tests for Paper 0 Formalism of the Homeostatic Quasicritical Controller source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.formalism_of_the_homeostatic_quasicritical_controller_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    FormalismOfTheHomeostaticQuasicriticalControllerConfig,
    classify_formalism_of_the_homeostatic_quasicritical_controller_component,
    formalism_of_the_homeostatic_quasicritical_controller_labels,
    validate_formalism_of_the_homeostatic_quasicritical_controller_fixture,
)


def test_formalism_of_the_homeostatic_quasicritical_controller_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_formalism_of_the_homeostatic_quasicritical_controller_fixture()
    assert result.source_ledger_span == ("P0R02869", "P0R02893")
    assert result.source_record_count == 25
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R02894"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_formalism_of_the_homeostatic_quasicritical_controller_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02869"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02893"


def test_formalism_of_the_homeostatic_quasicritical_controller_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("formalism_of_the_homeostatic_quasicritical_controller",):
        assert (
            classify_formalism_of_the_homeostatic_quasicritical_controller_component(component)
            == f"{component}_source_boundary"
        )
    labels = formalism_of_the_homeostatic_quasicritical_controller_labels()
    assert labels["section"] == "Formalism of the Homeostatic Quasicritical Controller"
    assert labels["next_boundary"] == "P0R02894"


def test_formalism_of_the_homeostatic_quasicritical_controller_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 25"):
        FormalismOfTheHomeostaticQuasicriticalControllerConfig(expected_source_record_count=24)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        FormalismOfTheHomeostaticQuasicriticalControllerConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02894"):
        FormalismOfTheHomeostaticQuasicriticalControllerConfig(next_source_boundary="P0R02893")
    with pytest.raises(
        ValueError, match="unknown formalism_of_the_homeostatic_quasicritical_controller component"
    ):
        classify_formalism_of_the_homeostatic_quasicritical_controller_component(
            "empirical_validation_claim"
        )
