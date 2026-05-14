# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 t0-seeding fixture tests
"""Tests for Paper 0 t=0 SSB seeding simulator fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.t0_seeding_validation import (
    T0SeedingConfig,
    effective_tachyonic_mass_term,
    initial_value_need_score,
    j_sec_memory_bias_score,
    spin_torsion_bridge,
    validate_spin_torsion_bridge_fixture,
    validate_t0_seeding_fixture,
)


def test_initial_value_need_and_j_sec_memory_bias_are_bounded() -> None:
    config = T0SeedingConfig()

    need = initial_value_need_score(
        restored_massless_boundary=0.91,
        symmetric_vacuum=0.88,
        ssb_trigger_requirement=0.86,
        config=config,
    )
    memory = j_sec_memory_bias_score(
        preserved_j_sec=0.89,
        conformal_invariance=0.84,
        prior_aeon_geometry=0.81,
        config=config,
    )

    assert need > config.initial_value_threshold
    assert memory > config.memory_bias_threshold


def test_effective_tachyonic_mass_term_tips_potential() -> None:
    config = T0SeedingConfig()

    coefficient = effective_tachyonic_mass_term(
        j_sec=0.82,
        coupling=0.74,
        lambda_coupling=0.41,
        psi_abs=0.35,
        config=config,
    )
    no_seed = effective_tachyonic_mass_term(
        j_sec=0.0,
        coupling=0.74,
        lambda_coupling=0.41,
        psi_abs=0.35,
        config=config,
    )

    assert coefficient < 0.0
    assert no_seed > coefficient


def test_spin_torsion_bridge_preserves_equation_formulae() -> None:
    config = T0SeedingConfig()

    torsion = spin_torsion_bridge(
        gravitational_constant=0.67,
        spin_density=0.73,
        config=config,
    )
    psi_torsion = spin_torsion_bridge(
        gravitational_constant=0.67,
        spin_density=0.81,
        config=config,
    )
    result = validate_spin_torsion_bridge_fixture()

    assert torsion > 0.0
    assert psi_torsion > torsion
    assert result.torsion_bridge_score > result.problem_metadata["torsion_threshold"]
    assert result.source_formulae == (
        "torsion_ijk = 8 pi G s_ijk",
        "torsion_ijk = 8 pi G s_ijk_psi",
    )
    assert result.null_controls["negative_spin_density_rejection_label"] == 1.0
    assert result.null_controls["unsupported_torsion_evidence_rejection_label"] == 1.0


def test_invalid_t0_seeding_config_and_inputs_reject_bad_parameters() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        T0SeedingConfig(initial_value_threshold=0.0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        T0SeedingConfig(memory_weight=-0.1)
    with pytest.raises(ValueError, match="finite and non-negative"):
        spin_torsion_bridge(
            gravitational_constant=-0.1,
            spin_density=0.7,
            config=T0SeedingConfig(),
        )


def test_t0_seeding_fixture_preserves_non_empirical_boundaries() -> None:
    result = validate_t0_seeding_fixture()

    assert result.spec_keys == (
        "t0_seeding.initial_value_problem_boundary",
        "t0_seeding.j_sec_memory_bias_boundary",
        "t0_seeding.teleological_tachyonic_potential",
        "t0_seeding.spin_torsion_bridge_equations",
        "t0_seeding.conformal_invariant_torsion_boundary",
    )
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.source_ledger_span == ("P0R06339", "P0R06362")
    assert result.initial_value_score > result.config_thresholds["initial_value_threshold"]
    assert result.memory_bias_score > result.config_thresholds["memory_bias_threshold"]
    assert result.tachyonic_coefficient < 0.0
    assert result.spin_torsion.torsion_bridge_score > result.config_thresholds["torsion_threshold"]
    assert result.conformal_torsion_score > result.config_thresholds["conformal_torsion_threshold"]
    assert "not empirical evidence" in result.claim_boundary
