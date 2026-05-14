# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 advanced mechanisms fixture tests
"""Tests for Paper 0 advanced mechanisms simulator fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.advanced_mechanisms_validation import (
    AdvancedMechanismsConfig,
    consilium_pareto_support_score,
    gauge_transduction_score,
    holographic_encoding_score,
    holographic_retrieval_score,
    validate_advanced_mechanisms_fixture,
)


def test_gauge_transduction_score_requires_symbol_operator_and_local_gauge_channels() -> None:
    complete = gauge_transduction_score(
        symbol_operator_strength=0.84,
        psi_resonance=0.82,
        local_gauge_transformation=0.88,
        field_connection_shift=0.79,
    )
    missing_connection = gauge_transduction_score(
        symbol_operator_strength=0.84,
        psi_resonance=0.82,
        local_gauge_transformation=0.88,
        field_connection_shift=0.0,
    )

    assert complete > missing_connection
    assert complete > AdvancedMechanismsConfig().mechanism_threshold


def test_holographic_encoding_and_retrieval_scores_are_bounded() -> None:
    encoding = holographic_encoding_score(
        l4_synchronisation=0.83,
        l1_quantum_bias=0.78,
        mera_boundary_mapping=0.81,
        bulk_entanglement_storage=0.86,
    )
    retrieval = holographic_retrieval_score(
        cue_syndrome_match=0.8,
        qec_recovery_operator=0.84,
        geodesic_flow_trace=0.79,
        l1_l4_reconstruction_bias=0.82,
    )

    assert encoding > AdvancedMechanismsConfig().mechanism_threshold
    assert retrieval > AdvancedMechanismsConfig().mechanism_threshold


def test_consilium_pareto_support_score_requires_multiobjective_inputs() -> None:
    score = consilium_pareto_support_score(
        coherence=0.82,
        complexity=0.77,
        qualia=0.74,
        pareto_feasibility=0.86,
        dynamic_weighting=0.81,
        geodesic_dissonance_reduction=0.79,
    )

    assert score > AdvancedMechanismsConfig().consilium_threshold


def test_advanced_mechanisms_fixture_rejects_invalid_inputs_and_preserves_boundaries() -> None:
    with pytest.raises(ValueError, match="mechanism inputs must be in \\[0, 1\\]"):
        gauge_transduction_score(
            symbol_operator_strength=1.2,
            psi_resonance=0.8,
            local_gauge_transformation=0.8,
            field_connection_shift=0.8,
        )
    with pytest.raises(ValueError, match="mechanism_threshold must be finite and positive"):
        AdvancedMechanismsConfig(mechanism_threshold=0.0)

    result = validate_advanced_mechanisms_fixture()

    assert result.spec_keys == (
        "advanced_mechanisms.geometric_physical_transduction",
        "advanced_mechanisms.holographic_memory_encoding",
        "advanced_mechanisms.holographic_memory_retrieval",
        "advanced_mechanisms.consilium_multiobjective_optimisation",
    )
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.source_ledger_span == ("P0R06382", "P0R06401")
    assert result.gauge_transduction > result.config_thresholds["mechanism_threshold"]
    assert result.holographic_encoding > result.config_thresholds["mechanism_threshold"]
    assert result.holographic_retrieval > result.config_thresholds["mechanism_threshold"]
    assert result.consilium_pareto_support > result.config_thresholds["consilium_threshold"]
    assert result.null_controls["missing_connection_rejection_label"] == 1.0
    assert result.null_controls["incomplete_memory_path_rejection_label"] == 1.0
    assert result.null_controls["unsupported_empirical_evidence_rejection_label"] == 1.0
    assert "not empirical evidence" in result.claim_boundary
