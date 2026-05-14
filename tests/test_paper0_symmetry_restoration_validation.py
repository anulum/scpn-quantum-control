# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 symmetry-restoration fixture tests
"""Tests for Paper 0 MMC symmetry-restoration simulator fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.symmetry_restoration_validation import (
    SymmetryRestorationConfig,
    conformal_boundary_violation_score,
    effective_quadratic_coefficient,
    massless_limit_scores,
    symmetry_restoration_score,
    validate_symmetry_restoration_fixture,
    validate_vev_melting_fixture,
)


def test_conformal_boundary_violation_scores_mass_retention() -> None:
    config = SymmetryRestorationConfig()

    violation = conformal_boundary_violation_score(
        psi_mass_retained=0.92,
        physical_scale_retained=0.88,
        conformal_invariance=0.16,
        config=config,
    )
    restored = conformal_boundary_violation_score(
        psi_mass_retained=0.08,
        physical_scale_retained=0.05,
        conformal_invariance=0.91,
        config=config,
    )

    assert violation > config.violation_threshold
    assert restored < config.violation_threshold


def test_effective_potential_flip_and_symmetry_restoration_score() -> None:
    config = SymmetryRestorationConfig()

    broken = effective_quadratic_coefficient(
        mu_squared=0.80,
        c1_tds_squared=0.18,
        c2_f_r=0.12,
    )
    restored = effective_quadratic_coefficient(
        mu_squared=0.80,
        c1_tds_squared=0.47,
        c2_f_r=0.42,
    )
    score = symmetry_restoration_score(
        thermal_correction=0.47,
        geometric_correction=0.42,
        negative_mass_squared=0.80,
        config=config,
    )

    assert broken < 0.0
    assert restored > 0.0
    assert score > config.restoration_threshold


def test_vev_melting_fixture_preserves_massless_limit_formulae() -> None:
    result = validate_vev_melting_fixture()

    assert result.vev_limit_score > result.problem_metadata["vev_limit_threshold"]
    assert result.infoton_mass_limit < result.problem_metadata["massless_tolerance"]
    assert result.psi_higgs_mass_limit < result.problem_metadata["massless_tolerance"]
    assert result.source_formulae == (
        "lim_{t -> infinity} v(t) = 0",
        "m_A = g v; m_h = sqrt(2 lambda) v",
    )
    assert result.null_controls["nonzero_vev_rejection_label"] == 1.0
    assert result.null_controls["negative_lambda_rejection_label"] == 1.0


def test_massless_limit_scores_reject_invalid_inputs() -> None:
    config = SymmetryRestorationConfig()

    with pytest.raises(ValueError, match="finite and non-negative"):
        SymmetryRestorationConfig(conformal_rescaling_weight=-0.1)
    with pytest.raises(ValueError, match="finite and positive"):
        SymmetryRestorationConfig(restoration_threshold=0.0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        massless_limit_scores(v=0.1, gauge_coupling=-0.2, lambda_coupling=0.4, config=config)


def test_symmetry_restoration_fixture_preserves_non_empirical_boundaries() -> None:
    result = validate_symmetry_restoration_fixture()

    assert result.spec_keys == (
        "symmetry_restoration.mmc_conformal_geometry_boundary",
        "symmetry_restoration.conformal_boundary_masslessness_constraint",
        "symmetry_restoration.effective_potential_flip_boundary",
        "symmetry_restoration.vev_melting_massless_limit",
        "symmetry_restoration.legal_conformal_rescaling_boundary",
    )
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.source_ledger_span == ("P0R06324", "P0R06338")
    assert result.conformal_score > result.config_thresholds["conformal_threshold"]
    assert result.violation_score > result.config_thresholds["violation_threshold"]
    assert result.restoration_score > result.config_thresholds["restoration_threshold"]
    assert result.vev_melting.vev_limit_score > result.config_thresholds["vev_limit_threshold"]
    assert result.legal_rescaling_score > result.config_thresholds["legal_rescaling_threshold"]
    assert "not empirical evidence" in result.claim_boundary
