# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Master Diagram: A Mandala of Consciousness validation tests
"""Tests for Paper 0 The Master Diagram: A Mandala of Consciousness source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_master_diagram_a_mandala_of_consciousness_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheMasterDiagramAMandalaOfConsciousnessConfig,
    classify_the_master_diagram_a_mandala_of_consciousness_component,
    the_master_diagram_a_mandala_of_consciousness_labels,
    validate_the_master_diagram_a_mandala_of_consciousness_fixture,
)


def test_the_master_diagram_a_mandala_of_consciousness_fixture_preserves_source_boundary() -> None:
    result = validate_the_master_diagram_a_mandala_of_consciousness_fixture()
    assert result.source_ledger_span == ("P0R02042", "P0R02049")
    assert result.source_record_count == 8
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R02050"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_master_diagram_a_mandala_of_consciousness_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02042"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02049"


def test_the_master_diagram_a_mandala_of_consciousness_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("the_master_diagram_a_mandala_of_consciousness",):
        assert (
            classify_the_master_diagram_a_mandala_of_consciousness_component(component)
            == f"{component}_source_boundary"
        )
    labels = the_master_diagram_a_mandala_of_consciousness_labels()
    assert labels["section"] == "The Master Diagram: A Mandala of Consciousness"
    assert labels["next_boundary"] == "P0R02050"


def test_the_master_diagram_a_mandala_of_consciousness_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        TheMasterDiagramAMandalaOfConsciousnessConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        TheMasterDiagramAMandalaOfConsciousnessConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02050"):
        TheMasterDiagramAMandalaOfConsciousnessConfig(next_source_boundary="P0R02049")
    with pytest.raises(
        ValueError, match="unknown the_master_diagram_a_mandala_of_consciousness component"
    ):
        classify_the_master_diagram_a_mandala_of_consciousness_component(
            "empirical_validation_claim"
        )
