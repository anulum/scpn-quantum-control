# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Gaian safety validation tests
"""Executable fixture tests for Paper 0 Gaian and societal safety records."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.gaian_safety_validation import (
    GaianSafetyConfig,
    classify_nths_phase,
    gaian_stability_index,
    safety_protocol_score,
    validate_gaian_safety_fixture,
    validate_governance_risk_safeguard_fixture,
)


def test_gaian_stability_increases_with_biodiversity_phi_and_sec() -> None:
    config = GaianSafetyConfig()

    protected = gaian_stability_index(0.82, 0.76, 0.71, config)
    degraded = gaian_stability_index(0.31, 0.28, 0.36, config)

    assert protected > degraded
    assert protected > config.gaian_stability_threshold
    assert degraded < config.gaian_stability_threshold


def test_nths_phase_classifier_separates_three_source_categories() -> None:
    config = GaianSafetyConfig()

    assert (
        classify_nths_phase(coherence=0.85, frustration=0.12, entropy_flux=0.25, config=config)
        == "ferromagnetic_coherence"
    )
    assert (
        classify_nths_phase(coherence=0.45, frustration=0.78, entropy_flux=0.42, config=config)
        == "spin_glass_fragmentation"
    )
    assert (
        classify_nths_phase(coherence=0.22, frustration=0.25, entropy_flux=0.88, config=config)
        == "paramagnetic_incoherence"
    )


def test_governance_risk_safeguard_protocol_scores_all_required_controls() -> None:
    config = GaianSafetyConfig()

    result = validate_governance_risk_safeguard_fixture(config)

    assert safety_protocol_score(config) > config.safety_protocol_threshold
    assert result.safeguard_score > result.risk_score
    assert result.null_controls["missing_entropy_budget_rejection_label"] == pytest.approx(1.0)
    assert result.null_controls["missing_layer_15_16_anchor_rejection_label"] == pytest.approx(1.0)
    assert "not empirical evidence" in result.claim_boundary


def test_gaian_safety_config_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        GaianSafetyConfig(entropy_budget_weight=-0.1)

    with pytest.raises(ValueError, match="finite and positive"):
        GaianSafetyConfig(gaian_stability_threshold=0.0)

    with pytest.raises(ValueError, match="threshold ordering"):
        GaianSafetyConfig(coherence_threshold=0.2, incoherence_entropy_threshold=0.1)


def test_gaian_safety_default_fixture_wires_all_boundaries() -> None:
    result = validate_gaian_safety_fixture()

    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.spec_keys == (
        "gaian_safety.biodiversity_phi_sec_boundary",
        "gaian_safety.ethical_functional_pela_boundary",
        "gaian_safety.nths_phase_category_validation",
        "gaian_safety.consciousness_engineering_safety_protocol",
        "gaian_safety.governance_risk_safeguard_protocol",
    )
    assert result.gaian_stability_delta > 0.0
    assert result.phase_categories == (
        "ferromagnetic_coherence",
        "spin_glass_fragmentation",
        "paramagnetic_incoherence",
    )
    assert result.governance.safeguard_score > result.governance.risk_score
    assert "not empirical evidence" in result.claim_boundary
