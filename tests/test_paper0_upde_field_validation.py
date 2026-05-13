# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Paper 0 UPDE field validation tests
"""Executable simulator fixtures for the Paper 0 UPDE field-coupling equation."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.paper0.upde_validation import (
    FieldCouplingConfig,
    field_alignment_projection,
    field_coupling_term,
    validate_upde_field_fixture,
)


def _theta_fixture() -> np.ndarray:
    return np.array([-0.72, -0.18, 0.24, 0.69, 1.12], dtype=np.float64)


def test_field_coupling_matches_paper0_cosine_formula() -> None:
    theta = _theta_fixture()
    config = FieldCouplingConfig(zeta_L=0.42, psi_global=1.7, theta_psi=0.31)

    term = field_coupling_term(theta, config=config)

    np.testing.assert_allclose(
        term,
        0.42 * 1.7 * np.cos(theta - 0.31),
        rtol=1e-14,
        atol=1e-14,
    )


def test_field_fixture_consumes_spec_and_records_null_controls() -> None:
    theta = _theta_fixture()
    config = FieldCouplingConfig(
        zeta_L=0.42,
        psi_global=1.7,
        theta_psi=0.31,
        random_phase_samples=256,
        random_seed=17,
    )

    result = validate_upde_field_fixture(theta, config=config)

    assert result.spec_key == "upde.field_coupling"
    assert result.validation_protocol == "paper0.upde.field.global_phase_coupling"
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.source_equation_ids == ("EQ0034", "EQ0041", "EQ0043")
    assert result.null_controls["zero_field_linf"] == pytest.approx(0.0)
    assert result.null_controls["randomised_phase_projection_abs_mean"] < (
        0.25 * result.field_alignment_projection
    )
    assert result.null_controls["bounded_amplitude"] == pytest.approx(0.42 * 1.7)


def test_field_fixture_exports_phase_artifact_and_metadata() -> None:
    result = validate_upde_field_fixture(_theta_fixture())
    payload = result.phase_artifact_payload

    assert payload["regime_id"] == "paper0_upde_field_fixture"
    assert payload["metadata"]["paper0_spec_key"] == "upde.field_coupling"
    assert payload["metadata"]["hardware_status"] == "simulator_only_no_provider_submission"
    assert payload["metadata"]["theta_psi"] == pytest.approx(result.theta_psi)
    assert len(payload["layers"]) == 1
    assert np.asarray(payload["cross_layer_alignment"]).shape == (1, 1)


def test_field_alignment_projection_is_positive_for_coherent_field() -> None:
    theta = _theta_fixture()
    config = FieldCouplingConfig(zeta_L=0.42, psi_global=1.7, theta_psi=0.31)
    term = field_coupling_term(theta, config=config)

    projection = field_alignment_projection(theta, term, theta_psi=config.theta_psi)

    assert projection > 0.0


def test_field_fixture_rejects_invalid_field_inputs() -> None:
    theta = _theta_fixture()

    bad_theta = theta.copy()
    bad_theta[2] = np.nan
    with pytest.raises(ValueError, match="theta must contain only finite values"):
        validate_upde_field_fixture(bad_theta)

    with pytest.raises(ValueError, match="zeta_L must be finite and non-negative"):
        validate_upde_field_fixture(theta, config=FieldCouplingConfig(zeta_L=-0.1))

    with pytest.raises(ValueError, match="psi_global must be finite and non-negative"):
        validate_upde_field_fixture(theta, config=FieldCouplingConfig(psi_global=np.inf))

    with pytest.raises(ValueError, match="random_phase_samples must be at least 2"):
        validate_upde_field_fixture(theta, config=FieldCouplingConfig(random_phase_samples=1))
