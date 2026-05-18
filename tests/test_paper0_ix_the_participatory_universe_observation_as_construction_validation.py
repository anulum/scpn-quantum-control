# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 IX. The Participatory Universe: Observation as Construction validation tests
"""Tests for Paper 0 IX. The Participatory Universe: Observation as Construction source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.ix_the_participatory_universe_observation_as_construction_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    IxTheParticipatoryUniverseObservationAsConstructionConfig,
    classify_ix_the_participatory_universe_observation_as_construction_component,
    ix_the_participatory_universe_observation_as_construction_labels,
    validate_ix_the_participatory_universe_observation_as_construction_fixture,
)


def test_ix_the_participatory_universe_observation_as_construction_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_ix_the_participatory_universe_observation_as_construction_fixture()
    assert result.source_ledger_span == ("P0R06047", "P0R06056")
    assert result.source_record_count == 10
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R06057"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_ix_the_participatory_universe_observation_as_construction_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R06047"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R06056"


def test_ix_the_participatory_universe_observation_as_construction_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "ix_the_participatory_universe_observation_as_construction",
        "x_symmetry_conservation_laws_and_the_coherence_current",
    ):
        assert (
            classify_ix_the_participatory_universe_observation_as_construction_component(component)
            == f"{component}_source_boundary"
        )
    labels = ix_the_participatory_universe_observation_as_construction_labels()
    assert labels["section"] == "IX. The Participatory Universe: Observation as Construction"
    assert labels["next_boundary"] == "P0R06057"


def test_ix_the_participatory_universe_observation_as_construction_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 10"):
        IxTheParticipatoryUniverseObservationAsConstructionConfig(expected_source_record_count=9)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        IxTheParticipatoryUniverseObservationAsConstructionConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R06057"):
        IxTheParticipatoryUniverseObservationAsConstructionConfig(next_source_boundary="P0R06056")
    with pytest.raises(
        ValueError,
        match="unknown ix_the_participatory_universe_observation_as_construction component",
    ):
        classify_ix_the_participatory_universe_observation_as_construction_component(
            "empirical_validation_claim"
        )
