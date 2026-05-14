# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 dark-sector fixture tests
"""Tests for Paper 0 dark-energy and psi-DM simulator fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.dark_sector_validation import (
    DarkSectorConfig,
    cosmic_reservoir_score,
    dark_energy_context_score,
    geometric_coupling_score,
    mmc_information_preservation_score,
    psi_dm_candidate_label,
    validate_dark_sector_fixture,
    validate_psi_dm_interaction_fixture,
)


def test_mmc_and_dark_energy_context_scores_are_bounded() -> None:
    config = DarkSectorConfig()

    mmc = mmc_information_preservation_score(
        conformal_rescaling=0.82,
        ethical_functional_conserved=0.91,
        entropy_reset=0.74,
        config=config,
    )
    dark_energy = dark_energy_context_score(
        lambda_potential=0.78,
        rg_flow_pressure=0.73,
        cosmic_attractor_drive=0.76,
        config=config,
    )

    assert mmc > config.mmc_threshold
    assert dark_energy > config.dark_energy_threshold


def test_psi_dm_candidate_labels_and_interactions_are_explicit() -> None:
    config = DarkSectorConfig()

    assert (
        psi_dm_candidate_label(
            ssb=True,
            alp_bec=True,
            q_ball=True,
            nonlinear_potential=True,
        )
        == "coherent_psi_field_dark_matter_candidate"
    )
    assert (
        psi_dm_candidate_label(
            ssb=True,
            alp_bec=False,
            q_ball=False,
            nonlinear_potential=True,
        )
        == "incomplete_psi_dm_candidate_boundary"
    )
    assert (
        geometric_coupling_score(
            stress_energy_tensor=0.84,
            curvature_coupling=0.81,
            weak_ordinary_matter_coupling=0.18,
            config=config,
        )
        > config.interaction_threshold
    )


def test_cosmic_reservoir_score_preserves_l8_l12_boundary() -> None:
    config = DarkSectorConfig()

    score = cosmic_reservoir_score(
        structure_scaffolding=0.77,
        halo_coherence=0.83,
        l8_phase_locking=0.71,
        l12_gaian_sync=0.69,
        config=config,
    )

    assert score > config.reservoir_threshold


def test_psi_dm_interaction_fixture_has_rejection_controls() -> None:
    result = validate_psi_dm_interaction_fixture()

    assert result.interaction_score > result.problem_metadata["interaction_threshold"]
    assert result.null_controls["missing_geometric_coupling_rejection_label"] == 1.0
    assert result.null_controls["missing_weak_coupling_boundary_rejection_label"] == 1.0
    assert result.null_controls["unsupported_dark_matter_evidence_rejection_label"] == 1.0
    assert result.source_formulae == ("L_Geometric proportional to -xi R Psi* Psi",)


def test_invalid_dark_sector_config_and_inputs_reject_bad_parameters() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        DarkSectorConfig(mmc_threshold=0.0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        DarkSectorConfig(conformal_rescaling_weight=-0.1)
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        geometric_coupling_score(
            stress_energy_tensor=1.2,
            curvature_coupling=0.5,
            weak_ordinary_matter_coupling=0.2,
            config=DarkSectorConfig(),
        )


def test_dark_sector_fixture_preserves_non_empirical_boundaries() -> None:
    result = validate_dark_sector_fixture()

    assert result.spec_keys == (
        "dark_sector.mmc_operator_information_preservation",
        "dark_sector.dark_energy_teleological_potential_boundary",
        "dark_sector.psi_dark_matter_hypothesis_boundary",
        "dark_sector.psi_dm_interaction_mechanisms",
        "dark_sector.cosmic_coherence_reservoir_boundary",
    )
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.source_ledger_span == ("P0R06311", "P0R06323")
    assert result.mmc_score > result.config_thresholds["mmc_threshold"]
    assert result.dark_energy_score > result.config_thresholds["dark_energy_threshold"]
    assert result.interaction.interaction_score > result.config_thresholds["interaction_threshold"]
    assert result.reservoir_score > result.config_thresholds["reservoir_threshold"]
    assert "not empirical evidence" in result.claim_boundary
