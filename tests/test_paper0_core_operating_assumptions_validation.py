# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 core operating assumptions validation tests
"""Tests for source-accounting checks around SCPN core operating assumptions."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.core_operating_assumptions_validation import (
    CoreOperatingAssumptionsConfig,
    classify_assumption_role,
    classify_hint_context,
    core_operating_assumption_labels,
    validate_core_operating_assumptions_fixture,
)


def test_core_assumption_roles_preserve_source_categories() -> None:
    assert classify_assumption_role("consciousness_fundamentality") == (
        "ontological_primitive_generative_assumption"
    )
    assert classify_assumption_role("bidirectional_causality") == (
        "recursive_top_down_bottom_up_causality"
    )
    assert (
        classify_assumption_role("field_realism") == "physical_measurable_engineerable_field_claim"
    )
    assert classify_assumption_role("unified_phase_dynamics") == (
        "universal_phase_synchronisation_language"
    )
    assert classify_assumption_role("ethical_functionals") == (
        "teleological_layer15_objective_prior"
    )

    with pytest.raises(ValueError, match="unknown core assumption"):
        classify_assumption_role("unknown")


def test_hint_context_preserves_reciprocal_coupling_and_lambda_boundary() -> None:
    assert classify_hint_context("psi_s") == "real_physical_field_context"
    assert classify_hint_context("sigma") == "phase_coherence_or_synchrony_candidate"
    assert classify_hint_context("causality") == "reciprocal_top_down_bottom_up_interaction"
    assert classify_hint_context("lambda") == "ethical_functional_tunes_parameter_not_force"

    with pytest.raises(ValueError, match="unknown H_int context"):
        classify_hint_context("force")


def test_core_operating_assumptions_fixture_is_source_bounded() -> None:
    result = validate_core_operating_assumptions_fixture()

    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_ledger_span == ("P0R00635", "P0R00669")
    assert result.core_assumption_count == 5
    assert result.blank_separator_count == 3
    assert result.next_source_boundary == "P0R00670"
    assert result.null_controls["operating_assumptions_are_not_empirical_results"] == 1.0
    assert result.null_controls["field_realism_claim_requires_downstream_validation"] == 1.0
    assert result.null_controls["ethical_functional_does_not_add_force_to_h_int"] == 1.0

    labels = core_operating_assumption_labels()
    assert labels["section"] == "The SCPN: Core Operating Assumptions"
    assert labels["h_int"] == "H_int = -lambda * Psi_s * sigma"
    assert labels["next_boundary"] == "Axiom I: The Primacy of Consciousness"
    assert result.as_dict()["problem_metadata"]["protocol_state"] == (
        "source_assumption_map_only_no_experiment"
    )


def test_core_operating_assumptions_config_rejects_wrong_source_counts() -> None:
    with pytest.raises(ValueError, match="expected_core_assumption_count must equal 5"):
        CoreOperatingAssumptionsConfig(expected_core_assumption_count=4)
    with pytest.raises(ValueError, match="expected_blank_separator_count must equal 3"):
        CoreOperatingAssumptionsConfig(expected_blank_separator_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R00670"):
        CoreOperatingAssumptionsConfig(next_source_boundary="P0R00671")
