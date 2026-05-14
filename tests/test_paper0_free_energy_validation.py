# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 free-energy validation tests
"""Executable fixture tests for Paper 0 EQ0130-EQ0131 anchors."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.paper0.free_energy_validation import (
    FreeEnergyConfig,
    free_energy_terms,
    validate_variational_free_energy_fixture,
)


def test_variational_free_energy_fixture_checks_decomposition_and_bound() -> None:
    config = FreeEnergyConfig(
        q_theta_mu=np.array([0.2, 0.5, 0.3], dtype=np.float64),
        p_theta_y=np.array([0.25, 0.45, 0.3], dtype=np.float64),
        p_y_theta=np.array([0.7, 0.4, 0.9], dtype=np.float64),
        evidence=0.7,
    )

    terms = free_energy_terms(config)
    result = validate_variational_free_energy_fixture(config)

    assert terms.complexity_kl >= 0.0
    assert terms.accuracy_loss > 0.0
    assert terms.total_free_energy == pytest.approx(terms.complexity_kl + terms.accuracy_loss)
    assert result.spec_key == "computational.variational_free_energy"
    assert result.source_equation_ids == ("EQ0130", "EQ0131")
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.decomposition_residual < 1.0e-12
    assert result.surprise_upper_bound_margin > 0.0
    assert result.null_controls["identical_density_kl_abs"] == pytest.approx(0.0)
    assert result.null_controls["support_mismatch_rejection_label"] == pytest.approx(1.0)


def test_free_energy_fixture_rejects_invalid_probability_inputs() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        FreeEnergyConfig(q_theta_mu=np.array([0.5, 0.5]), p_theta_y=np.array([0.5, 0.0]))

    with pytest.raises(ValueError, match="same shape"):
        FreeEnergyConfig(q_theta_mu=np.array([0.6, 0.4]), p_theta_y=np.array([0.2, 0.3, 0.5]))

    with pytest.raises(ValueError, match="evidence"):
        FreeEnergyConfig(evidence=0.0)
