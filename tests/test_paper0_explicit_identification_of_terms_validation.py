# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Explicit Identification of Terms: validation tests
"""Tests for Paper 0 Explicit Identification of Terms: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.explicit_identification_of_terms_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ExplicitIdentificationOfTermsConfig,
    classify_explicit_identification_of_terms_component,
    explicit_identification_of_terms_labels,
    validate_explicit_identification_of_terms_fixture,
)


def test_explicit_identification_of_terms_fixture_preserves_source_boundary() -> None:
    result = validate_explicit_identification_of_terms_fixture()
    assert result.source_ledger_span == ("P0R04283", "P0R04290")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04291"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_explicit_identification_of_terms_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04283"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04290"


def test_explicit_identification_of_terms_classification_and_labels_are_explicit() -> None:
    for component in (
        "explicit_identification_of_terms",
        "the_nature_of_the_interaction",
        "the_psi_field_electromagnetic_interface_the_role_of_axion_like_particles",
    ):
        assert (
            classify_explicit_identification_of_terms_component(component)
            == f"{component}_source_boundary"
        )
    labels = explicit_identification_of_terms_labels()
    assert labels["section"] == "Explicit Identification of Terms:"
    assert labels["next_boundary"] == "P0R04291"


def test_explicit_identification_of_terms_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        ExplicitIdentificationOfTermsConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        ExplicitIdentificationOfTermsConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04291"):
        ExplicitIdentificationOfTermsConfig(next_source_boundary="P0R04290")
    with pytest.raises(ValueError, match="unknown explicit_identification_of_terms component"):
        classify_explicit_identification_of_terms_component("empirical_validation_claim")
