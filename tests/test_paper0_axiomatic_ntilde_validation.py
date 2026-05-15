# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 axiomatic Ntilde validation tests
"""Tests for source-accounting checks around the formal Logos/Ntilde slice."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.axiomatic_ntilde_validation import (
    AxiomaticNtildeConfig,
    classify_axiomatic_status,
    classify_ntilde_regime,
    irreversibility_delta,
    ntilde_ratio,
    validate_axiomatic_ntilde_fixture,
)


def test_axiomatic_status_classification_preserves_claim_boundary() -> None:
    assert classify_axiomatic_status("axiom_1") == "metaphysical_assumption_generative"
    assert classify_axiomatic_status("axiom_2") == "falsifiable_information_geometry_hypothesis"
    assert classify_axiomatic_status("axiom_3") == (
        "normative_teleology_with_proposed_falsifiable_ntilde_invariant"
    )
    assert classify_axiomatic_status("axiom_3_status_tension") == (
        "preserve_normative_to_physical_claim_transition"
    )

    with pytest.raises(ValueError, match="unknown axiomatic status key"):
        classify_axiomatic_status("axiom_4")


def test_ntilde_ratio_and_irreversibility_delta_are_numeric_and_guarded() -> None:
    assert ntilde_ratio(power=10.0, reversible_cost_per_bit=2.0, information_rate=5.0) == 1.0
    assert irreversibility_delta(1.25) == pytest.approx(0.25)
    assert classify_ntilde_regime(1.0) == "quasicritical_reversible_threshold"
    assert classify_ntilde_regime(1.2) == "irreversible_overhead"
    assert classify_ntilde_regime(0.8) == "underpowered_or_efficiency_claim_rejected"

    with pytest.raises(ValueError, match="power must be finite and positive"):
        ntilde_ratio(power=0.0, reversible_cost_per_bit=2.0, information_rate=5.0)
    with pytest.raises(ValueError, match="reversible_cost_per_bit must be finite and positive"):
        ntilde_ratio(power=1.0, reversible_cost_per_bit=-1.0, information_rate=5.0)
    with pytest.raises(ValueError, match="information_rate must be finite and positive"):
        ntilde_ratio(power=1.0, reversible_cost_per_bit=2.0, information_rate=0.0)


def test_axiomatic_ntilde_fixture_is_source_bounded() -> None:
    result = validate_axiomatic_ntilde_fixture()

    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_ledger_span == ("P0R00578", "P0R00609")
    assert result.axiom_count == 3
    assert result.ntilde_formula_count == 5
    assert result.next_source_boundary == "P0R00610"
    assert result.null_controls["figure_caption_is_not_validation_evidence"] == 1.0
    assert result.null_controls["status_transition_is_not_empirical_confirmation"] == 1.0
    assert result.null_controls["ntilde_unity_is_target_not_observed_result"] == 1.0

    payload = result.as_dict()
    assert (
        payload["problem_metadata"]["protocol_state"] == "source_invariant_map_only_no_experiment"
    )


def test_axiomatic_ntilde_config_rejects_wrong_source_counts() -> None:
    with pytest.raises(ValueError, match="expected_axiom_count must equal 3"):
        AxiomaticNtildeConfig(expected_axiom_count=2)
    with pytest.raises(ValueError, match="expected_ntilde_formula_count must equal 5"):
        AxiomaticNtildeConfig(expected_ntilde_formula_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R00610"):
        AxiomaticNtildeConfig(next_source_boundary="P0R00611")
