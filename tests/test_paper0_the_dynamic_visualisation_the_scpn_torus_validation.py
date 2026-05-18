# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Dynamic Visualisation: The SCPN Torus validation tests
"""Tests for Paper 0 The Dynamic Visualisation: The SCPN Torus source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_dynamic_visualisation_the_scpn_torus_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheDynamicVisualisationTheScpnTorusConfig,
    classify_the_dynamic_visualisation_the_scpn_torus_component,
    the_dynamic_visualisation_the_scpn_torus_labels,
    validate_the_dynamic_visualisation_the_scpn_torus_fixture,
)


def test_the_dynamic_visualisation_the_scpn_torus_fixture_preserves_source_boundary() -> None:
    result = validate_the_dynamic_visualisation_the_scpn_torus_fixture()
    assert result.source_ledger_span == ("P0R02532", "P0R02541")
    assert result.source_record_count == 10
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R02542"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_dynamic_visualisation_the_scpn_torus_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02532"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02541"


def test_the_dynamic_visualisation_the_scpn_torus_classification_and_labels_are_explicit() -> None:
    for component in ("the_dynamic_visualisation_the_scpn_torus",):
        assert (
            classify_the_dynamic_visualisation_the_scpn_torus_component(component)
            == f"{component}_source_boundary"
        )
    labels = the_dynamic_visualisation_the_scpn_torus_labels()
    assert labels["section"] == "The Dynamic Visualisation: The SCPN Torus"
    assert labels["next_boundary"] == "P0R02542"


def test_the_dynamic_visualisation_the_scpn_torus_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 10"):
        TheDynamicVisualisationTheScpnTorusConfig(expected_source_record_count=9)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        TheDynamicVisualisationTheScpnTorusConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02542"):
        TheDynamicVisualisationTheScpnTorusConfig(next_source_boundary="P0R02541")
    with pytest.raises(
        ValueError, match="unknown the_dynamic_visualisation_the_scpn_torus component"
    ):
        classify_the_dynamic_visualisation_the_scpn_torus_component("empirical_validation_claim")
