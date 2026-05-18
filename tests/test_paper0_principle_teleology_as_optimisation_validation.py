# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Principle (teleology as optimisation). validation tests
"""Tests for Paper 0 Principle (teleology as optimisation). source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.principle_teleology_as_optimisation_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    PrincipleTeleologyAsOptimisationConfig,
    classify_principle_teleology_as_optimisation_component,
    principle_teleology_as_optimisation_labels,
    validate_principle_teleology_as_optimisation_fixture,
)


def test_principle_teleology_as_optimisation_fixture_preserves_source_boundary() -> None:
    result = validate_principle_teleology_as_optimisation_fixture()
    assert result.source_ledger_span == ("P0R04001", "P0R04008")
    assert result.source_record_count == 8
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R04009"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_principle_teleology_as_optimisation_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04001"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04008"


def test_principle_teleology_as_optimisation_classification_and_labels_are_explicit() -> None:
    for component in ("principle_teleology_as_optimisation",):
        assert (
            classify_principle_teleology_as_optimisation_component(component)
            == f"{component}_source_boundary"
        )
    labels = principle_teleology_as_optimisation_labels()
    assert labels["section"] == "Principle (teleology as optimisation)."
    assert labels["next_boundary"] == "P0R04009"


def test_principle_teleology_as_optimisation_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        PrincipleTeleologyAsOptimisationConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        PrincipleTeleologyAsOptimisationConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04009"):
        PrincipleTeleologyAsOptimisationConfig(next_source_boundary="P0R04008")
    with pytest.raises(ValueError, match="unknown principle_teleology_as_optimisation component"):
        classify_principle_teleology_as_optimisation_component("empirical_validation_claim")
