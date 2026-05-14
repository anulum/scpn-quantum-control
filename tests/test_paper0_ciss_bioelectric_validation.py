# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 CISS-bioelectric fixture tests
"""Tests for Paper 0 Layer 3 CISS-bioelectric fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.paper0.ciss_bioelectric_validation import (
    CISSBioelectricConfig,
    bioelectric_cascade_drive,
    ciss_effective_field_t,
    ciss_spin_filter_hamiltonian,
    membrane_potential_derivative,
    radical_pair_hamiltonian,
    validate_ciss_bioelectric_fixture,
)


def test_ciss_spin_filter_and_effective_field_preserve_source_terms() -> None:
    total = ciss_spin_filter_hamiltonian(
        epsilon_0=0.2,
        delta=0.4,
        sigma_z=-1.0,
        spin_orbit_lambda=0.05,
        length_scale=2.0,
        sigma_dot_l=3.0,
        g_factor=2.1,
        s_dot_sigma=0.25,
    )

    assert total == pytest.approx(0.2 - 0.2 + (0.05 / 4.0) * 3.0 + 2.1 * 0.25)
    assert ciss_effective_field_t(spin_orbit_lambda=0.5, scale_t_per_lambda=80.0) == pytest.approx(
        40.0
    )


def test_radical_pair_and_bioelectric_feedback_terms_are_finite() -> None:
    zeeman = np.array([0.2, 0.3], dtype=np.float64)
    hyperfine = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float64)

    rp = radical_pair_hamiltonian(
        zeeman_terms=zeeman,
        hyperfine_terms=hyperfine,
        exchange_j=0.4,
        s1_dot_s2=-0.25,
    )
    derivative = membrane_potential_derivative(ionic_current=0.7, pump_current=0.2)
    cascade = bioelectric_cascade_drive(
        target_potential_gradient=1.5,
        cav_activation_gain=0.8,
        camkii_gain=0.6,
        chromatin_gain=0.5,
    )

    assert rp == pytest.approx(float(zeeman.sum() + hyperfine.sum() + 0.4 * (0.5 - 0.5)))
    assert derivative == pytest.approx(-0.5)
    assert cascade == pytest.approx(-1.5 * 0.8 * 0.6 * 0.5)


def test_ciss_bioelectric_fixture_preserves_boundaries_and_controls() -> None:
    with pytest.raises(ValueError, match="length_scale must be finite and positive"):
        ciss_spin_filter_hamiltonian(
            epsilon_0=0.0,
            delta=0.1,
            sigma_z=1.0,
            spin_orbit_lambda=0.2,
            length_scale=0.0,
            sigma_dot_l=1.0,
            g_factor=2.0,
            s_dot_sigma=0.1,
        )
    with pytest.raises(ValueError, match="zeeman_terms must be one-dimensional"):
        radical_pair_hamiltonian(
            zeeman_terms=np.ones((1, 2), dtype=np.float64),
            hyperfine_terms=np.ones((2, 2), dtype=np.float64),
            exchange_j=0.1,
            s1_dot_s2=0.0,
        )
    with pytest.raises(
        ValueError, match="effective_field_scale_t_per_lambda must be finite and positive"
    ):
        CISSBioelectricConfig(effective_field_scale_t_per_lambda=0.0)

    result = validate_ciss_bioelectric_fixture()

    assert result.spec_keys == (
        "ciss_bioelectric.layer3_framing",
        "ciss_bioelectric.ciss_spin_filter",
        "ciss_bioelectric.radical_pair_modulation",
        "ciss_bioelectric.bioelectric_cascade_feedback",
        "ciss_bioelectric.observable_predictions",
    )
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.source_ledger_span == ("P0R06560", "P0R06581")
    assert 10.0 <= result.effective_field_t <= 100.0
    assert result.feedback_derivative < 0.0
    assert result.radical_pair_yield_modulation > 0.0
    assert result.null_controls["non_positive_length_rejection_label"] == 1.0
    assert result.null_controls["shape_mismatch_rejection_label"] == 1.0
    assert result.null_controls["unsupported_morphogenesis_evidence_rejection_label"] == 1.0
    assert "not empirical evidence" in result.claim_boundary
