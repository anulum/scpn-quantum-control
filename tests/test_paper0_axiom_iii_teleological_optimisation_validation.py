# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom III teleological optimisation validation tests
"""Tests for Paper 0 Axiom III teleological-optimisation validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.axiom_iii_teleological_optimisation_validation import (
    AxiomIIITeleologicalOptimisationConfig,
    axiom_iii_teleological_optimisation_labels,
    classify_teleological_optimisation_component,
    validate_axiom_iii_teleological_optimisation_fixture,
)


def test_teleological_optimisation_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        AxiomIIITeleologicalOptimisationConfig(expected_source_record_count=8)

    with pytest.raises(ValueError, match="expected_sec_maximisation_count must equal 2"):
        AxiomIIITeleologicalOptimisationConfig(expected_sec_maximisation_count=1)

    with pytest.raises(ValueError, match="next_source_boundary must equal P0R00800"):
        AxiomIIITeleologicalOptimisationConfig(next_source_boundary="P0R00799")


def test_teleological_optimisation_classifiers_are_source_bounded() -> None:
    assert (
        classify_teleological_optimisation_component("opening_context")
        == "axiom_iii_headings_and_formal_law_pointers"
    )
    assert (
        classify_teleological_optimisation_component("source_material_telos")
        == "source_telos_maximal_sustainable_ethical_coherence"
    )
    assert (
        classify_teleological_optimisation_component("directional_purpose")
        == "axiom_iii_directional_purpose_and_sec_maximisation"
    )
    assert (
        classify_teleological_optimisation_component("ethical_functional_guidance")
        == "layer15_ethical_functionals_bias_temporal_evolution"
    )

    with pytest.raises(ValueError, match="unknown teleological-optimisation component"):
        classify_teleological_optimisation_component("ntilde_equation")


def test_teleological_optimisation_fixture_preserves_claim_boundary_and_null_controls() -> None:
    result = validate_axiom_iii_teleological_optimisation_fixture()

    assert result.source_ledger_span == ("P0R00791", "P0R00799")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 9
    assert result.axiom_heading_count == 3
    assert result.sec_maximisation_count == 2
    assert result.layer15_guidance_count == 1
    assert result.directionality_count == 2
    assert result.next_source_boundary == "P0R00800"
    assert result.null_controls == {
        "teleological_postulate_is_source_claim_not_empirical_evidence": 1.0,
        "ntilde_law_heading_is_pointer_not_equation_in_this_slice": 1.0,
        "layer15_ethical_functional_guidance_requires_downstream_operationalisation": 1.0,
    }
    assert result.problem_metadata["protocol_state"] == (
        "source_axiom_iii_teleological_optimisation_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R00791"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R00799"


def test_teleological_optimisation_labels_name_ntilde_boundary() -> None:
    labels = axiom_iii_teleological_optimisation_labels()

    assert labels["section"] == "Axiom III: The Drive of Teleological Optimisation"
    assert labels["telos"] == "maximal Sustainable Ethical Coherence"
    assert labels["architecture_layer"] == "Layer 15 ethical functionals"
    assert labels["next_boundary"] == "Formal Physical Definition: The tilde_N_t Invariance Law"
