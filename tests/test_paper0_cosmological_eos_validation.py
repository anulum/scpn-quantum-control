# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 cosmological equation-of-state fixture tests
"""Tests for Paper 0 cosmological equation-of-state fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.cosmological_eos_validation import (
    CosmologicalEOSConfig,
    equation_of_state,
    observational_w0_consistency,
    scalar_field_density_pressure,
    stress_energy_split,
    validate_cosmological_eos_fixture,
)


def test_scalar_field_equation_of_state_matches_source_formulae() -> None:
    density, pressure = scalar_field_density_pressure(psi_dot=0.2, potential=2.0)
    w_value = equation_of_state(psi_dot=0.2, potential=2.0)

    assert density == pytest.approx(2.02)
    assert pressure == pytest.approx(-1.98)
    assert w_value == pytest.approx(-1.98 / 2.02)
    assert equation_of_state(psi_dot=0.0, potential=2.0) == pytest.approx(-1.0)
    assert equation_of_state(psi_dot=2.0, potential=0.0) == pytest.approx(1.0)


def test_observational_and_split_guards_are_bounded() -> None:
    config = CosmologicalEOSConfig()

    assert observational_w0_consistency(w0=-1.03, sigma=0.03, target=-1.0, config=config) is True
    assert observational_w0_consistency(w0=-0.88, sigma=0.03, target=-1.0, config=config) is False
    assert stress_energy_split(background=0.97, perturbation=0.03, config=config) == pytest.approx(
        (0.97, 0.03)
    )

    with pytest.raises(ValueError, match="density denominator must be positive"):
        equation_of_state(psi_dot=0.0, potential=0.0)
    with pytest.raises(ValueError, match="sigma must be finite and positive"):
        observational_w0_consistency(w0=-1.0, sigma=0.0, target=-1.0, config=config)
    with pytest.raises(ValueError, match="perturbation_fraction must be in \\[0, 1\\]"):
        CosmologicalEOSConfig(perturbation_fraction=1.2)


def test_cosmological_eos_fixture_preserves_claim_boundary() -> None:
    result = validate_cosmological_eos_fixture()

    assert result.spec_keys == (
        "cosmological_eos.chapter_boundary",
        "cosmological_eos.scalar_field_equations",
        "cosmological_eos.limiting_cases",
        "cosmological_eos.observational_constraint",
        "cosmological_eos.hybrid_split_and_homogeneity",
        "cosmological_eos.quintessence_detection_target",
    )
    assert result.hardware_status == "cosmological_constraint_fixture_no_execution"
    assert result.source_ledger_span == ("P0R06916", "P0R06948")
    assert result.slow_roll_w == pytest.approx(-1.0)
    assert result.kinetic_dominated_w == pytest.approx(1.0)
    assert result.observational_constraint_consistent is True
    assert result.background_fraction > result.perturbation_fraction
    assert result.null_controls["zero_density_rejection_label"] == 1.0
    assert result.null_controls["invalid_perturbation_fraction_rejection_label"] == 1.0
    assert result.null_controls["unsupported_cosmology_validation_rejection_label"] == 1.0
    assert "not empirical evidence" in result.claim_boundary
