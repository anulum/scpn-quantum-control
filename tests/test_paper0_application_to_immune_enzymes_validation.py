# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Application to Immune Enzymes validation tests
"""Tests for Paper 0 Application to Immune Enzymes source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.application_to_immune_enzymes_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ApplicationToImmuneEnzymesConfig,
    application_to_immune_enzymes_labels,
    classify_application_to_immune_enzymes_component,
    validate_application_to_immune_enzymes_fixture,
)


def test_application_to_immune_enzymes_fixture_preserves_source_boundary() -> None:
    result = validate_application_to_immune_enzymes_fixture()
    assert result.source_ledger_span == ("P0R05517", "P0R05527")
    assert result.source_record_count == 11
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R05528"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_application_to_immune_enzymes_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05517"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05527"


def test_application_to_immune_enzymes_classification_and_labels_are_explicit() -> None:
    for component in (
        "application_to_immune_enzymes",
        "the_biophysics_of_coherence_a_scale_invariant_cybernetic_thread",
        "micro_scale_homeostasis_glial_control_of_neuronal_criticality",
    ):
        assert (
            classify_application_to_immune_enzymes_component(component)
            == f"{component}_source_boundary"
        )
    labels = application_to_immune_enzymes_labels()
    assert labels["section"] == "Application to Immune Enzymes"
    assert labels["next_boundary"] == "P0R05528"


def test_application_to_immune_enzymes_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 11"):
        ApplicationToImmuneEnzymesConfig(expected_source_record_count=10)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        ApplicationToImmuneEnzymesConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05528"):
        ApplicationToImmuneEnzymesConfig(next_source_boundary="P0R05527")
    with pytest.raises(ValueError, match="unknown application_to_immune_enzymes component"):
        classify_application_to_immune_enzymes_component("empirical_validation_claim")
