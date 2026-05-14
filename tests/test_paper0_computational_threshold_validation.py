# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 computational-threshold validation tests
"""Executable fixture tests for Paper 0 EQ0119-EQ0122 anchors."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.paper0.computational_threshold_validation import (
    CoherenceCurrentConfig,
    IITThresholdConfig,
    InformationEnergyTransductionConfig,
    iit_threshold_energy,
    quantum_potential,
    validate_coherence_noether_current_fixture,
    validate_iit_or_threshold_fixture,
    validate_information_energy_transduction_fixture,
)


def test_iit_or_threshold_fixture_checks_linear_energy_and_labels() -> None:
    config = IITThresholdConfig(
        phi_values=np.array([0.1, 0.35, 0.8, 1.2], dtype=np.float64),
        alpha_phi=2.5,
        phi_crit=0.75,
    )

    energy = iit_threshold_energy(config)
    result = validate_iit_or_threshold_fixture(config)

    assert np.allclose(energy, np.array([0.25, 0.875, 2.0, 3.0]))
    assert result.spec_key == "computational.iit_or_threshold"
    assert result.source_equation_ids == ("EQ0119",)
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.proportionality_residual < 1.0e-12
    assert result.threshold_labels == (0, 0, 1, 1)
    assert result.null_controls["alpha_zero_energy_max_abs"] == pytest.approx(0.0)
    assert result.null_controls["subcritical_label_count"] == pytest.approx(0.0)


def test_noether_current_fixture_checks_phase_invariance_and_conservation() -> None:
    result = validate_coherence_noether_current_fixture(CoherenceCurrentConfig())

    assert result.spec_key == "computational.coherence_noether_current"
    assert result.source_equation_ids == ("EQ0120",)
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.global_phase_invariance_error < 1.0e-12
    assert result.divergence_residual < 1.0e-10
    assert result.null_controls["phase_broken_divergence_residual"] > result.divergence_residual
    assert result.null_controls["open_boundary_flux_label"] == pytest.approx(1.0)


def test_information_energy_transduction_fixture_checks_quantum_potential() -> None:
    config = InformationEnergyTransductionConfig(grid_points=401, domain_radius=6.0, sigma=1.3)

    result = validate_information_energy_transduction_fixture(config)

    assert result.spec_key == "computational.information_energy_transduction"
    assert result.source_equation_ids == ("EQ0121", "EQ0122")
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.constant_density_max_abs < 1.0e-12
    assert result.gaussian_residual_rms < 4.0e-4
    assert result.null_controls["non_positive_rho_rejection_label"] == pytest.approx(1.0)
    assert result.null_controls["grid_refinement_improvement_label"] == pytest.approx(1.0)


def test_quantum_potential_rejects_invalid_density() -> None:
    with pytest.raises(ValueError, match="rho must be strictly positive"):
        quantum_potential(np.array([1.0, 0.0, 1.0]), dx=1.0)

    with pytest.raises(ValueError, match="alpha_phi"):
        IITThresholdConfig(alpha_phi=-1.0)

    with pytest.raises(ValueError, match="grid_points"):
        InformationEnergyTransductionConfig(grid_points=10)
