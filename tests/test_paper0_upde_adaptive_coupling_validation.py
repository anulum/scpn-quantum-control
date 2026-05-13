# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Paper 0 UPDE adaptive-coupling validation tests
"""Executable simulator fixture for the Paper 0 adaptive-coupling equation."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.paper0.upde_validation import (
    AdaptiveCouplingConfig,
    adaptive_coupling_rates,
    apply_adaptive_coupling_step,
    validate_upde_adaptive_coupling_fixture,
)


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    K = np.array(
        [
            [0.0, 0.31, 0.12],
            [0.31, 0.0, 0.24],
            [0.12, 0.24, 0.0],
        ],
        dtype=np.float64,
    )
    noise = np.array(
        [
            [0.0, 0.01, -0.005],
            [0.01, 0.0, 0.004],
            [-0.005, 0.004, 0.0],
        ],
        dtype=np.float64,
    )
    return K, noise


def test_adaptive_rates_match_paper0_source_equations() -> None:
    K, noise = _fixture()
    config = AdaptiveCouplingConfig(gamma_L=0.8, lambda_L=0.15, alpha_L=0.45)

    rates = adaptive_coupling_rates(
        K,
        R_L=0.62,
        R_L_star=0.75,
        sigma_L=1.18,
        noise_xi=noise,
        eta_L=0.4,
        config=config,
    )

    expected_K = 0.8 * (0.62 - 0.75) - 0.15 * K + noise
    np.fill_diagonal(expected_K, 0.0)
    expected_eta = -0.45 * (1.18 - 1.0)
    np.testing.assert_allclose(rates.K_dot, expected_K, atol=1e-15, rtol=1e-15)
    assert rates.eta_dot == pytest.approx(expected_eta)


def test_adaptive_step_preserves_symmetry_and_zero_diagonal() -> None:
    K, noise = _fixture()

    step = apply_adaptive_coupling_step(
        K,
        R_L=0.62,
        R_L_star=0.75,
        sigma_L=1.18,
        noise_xi=noise,
        eta_L=0.4,
        dt=0.05,
        config=AdaptiveCouplingConfig(gamma_L=0.8, lambda_L=0.15, alpha_L=0.45),
    )

    np.testing.assert_allclose(step.K_next, step.K_next.T)
    np.testing.assert_allclose(np.diag(step.K_next), 0.0)
    assert np.isfinite(step.eta_next)


def test_adaptive_fixture_consumes_spec_and_records_null_controls() -> None:
    K, noise = _fixture()
    config = AdaptiveCouplingConfig(gamma_L=0.8, lambda_L=0.15, alpha_L=0.45)

    result = validate_upde_adaptive_coupling_fixture(
        K,
        R_L=0.62,
        R_L_star=0.75,
        sigma_L=1.18,
        noise_xi=noise,
        eta_L=0.4,
        dt=0.05,
        config=config,
    )

    assert result.spec_key == "upde.adaptive_coupling"
    assert result.validation_protocol == "paper0.upde.adaptive_coupling.quasicritical_controller"
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.source_equation_ids == ("EQ0045",)
    assert result.null_controls["zero_gain_K_dot_linf"] == pytest.approx(0.0)
    assert result.null_controls["zero_gain_eta_dot_abs"] == pytest.approx(0.0)
    assert result.null_controls["wrong_sign_K_dot_l2"] > 0.0
    assert result.null_controls["wrong_sign_eta_dot_abs"] > 0.0
    assert result.bounded_update_linf <= config.max_abs_update


def test_adaptive_fixture_rejects_invalid_inputs_and_unbounded_gains() -> None:
    K, noise = _fixture()
    bad_K = K.copy()
    bad_K[0, 1] = np.nan

    with pytest.raises(ValueError, match="K_ij_L must contain only finite values"):
        validate_upde_adaptive_coupling_fixture(bad_K, 0.6, 0.7, 1.1, noise, 0.4)

    with pytest.raises(ValueError, match="noise_xi must be symmetric"):
        validate_upde_adaptive_coupling_fixture(K, 0.6, 0.7, 1.1, noise + np.triu(noise, 1), 0.4)

    with pytest.raises(ValueError, match="gamma_L exceeds max_abs_gain"):
        validate_upde_adaptive_coupling_fixture(
            K,
            0.6,
            0.7,
            1.1,
            noise,
            0.4,
            config=AdaptiveCouplingConfig(gamma_L=1e6, max_abs_gain=10.0),
        )

    with pytest.raises(ValueError, match="lambda_L must be finite and non-negative"):
        validate_upde_adaptive_coupling_fixture(
            K,
            0.6,
            0.7,
            1.1,
            noise,
            0.4,
            config=AdaptiveCouplingConfig(lambda_L=-0.1),
        )
