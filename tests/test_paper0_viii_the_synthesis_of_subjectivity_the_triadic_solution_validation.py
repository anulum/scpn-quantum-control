# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 VIII. The Synthesis of Subjectivity (The Triadic Solution) validation tests
"""Tests for Paper 0 VIII. The Synthesis of Subjectivity (The Triadic Solution) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.viii_the_synthesis_of_subjectivity_the_triadic_solution_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ViiiTheSynthesisOfSubjectivityTheTriadicSolutionConfig,
    classify_viii_the_synthesis_of_subjectivity_the_triadic_solution_component,
    validate_viii_the_synthesis_of_subjectivity_the_triadic_solution_fixture,
    viii_the_synthesis_of_subjectivity_the_triadic_solution_labels,
)


def test_viii_the_synthesis_of_subjectivity_the_triadic_solution_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_viii_the_synthesis_of_subjectivity_the_triadic_solution_fixture()
    assert result.source_ledger_span == ("P0R06132", "P0R06146")
    assert result.source_record_count == 15
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R06147"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_viii_the_synthesis_of_subjectivity_the_triadic_solution_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R06132"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R06146"


def test_viii_the_synthesis_of_subjectivity_the_triadic_solution_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "viii_the_synthesis_of_subjectivity_the_triadic_solution",
        "ix_the_physics_of_teleology_and_ethics_the_teleological_engine",
        "x_the_dynamics_of_dissolution_death_and_transcendence",
        "the_physics_of_information_weaving_space_time_and_consciousness",
    ):
        assert (
            classify_viii_the_synthesis_of_subjectivity_the_triadic_solution_component(component)
            == f"{component}_source_boundary"
        )
    labels = viii_the_synthesis_of_subjectivity_the_triadic_solution_labels()
    assert labels["section"] == "VIII. The Synthesis of Subjectivity (The Triadic Solution)"
    assert labels["next_boundary"] == "P0R06147"


def test_viii_the_synthesis_of_subjectivity_the_triadic_solution_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 15"):
        ViiiTheSynthesisOfSubjectivityTheTriadicSolutionConfig(expected_source_record_count=14)
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        ViiiTheSynthesisOfSubjectivityTheTriadicSolutionConfig(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R06147"):
        ViiiTheSynthesisOfSubjectivityTheTriadicSolutionConfig(next_source_boundary="P0R06146")
    with pytest.raises(
        ValueError,
        match="unknown viii_the_synthesis_of_subjectivity_the_triadic_solution component",
    ):
        classify_viii_the_synthesis_of_subjectivity_the_triadic_solution_component(
            "empirical_validation_claim"
        )
