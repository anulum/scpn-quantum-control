# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Paper 0 UPDE natural-gradient validation tests
"""Executable simulator fixture for the Paper 0 UPDE natural-gradient equation."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.paper0.upde_validation import (
    NaturalGradientConfig,
    finite_difference_quadratic_gradient,
    natural_gradient_flow,
    quadratic_free_energy,
    validate_upde_natural_gradient_fixture,
)


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = np.array([0.42, -0.31, 0.88], dtype=np.float64)
    target = np.array([0.1, -0.05, 0.55], dtype=np.float64)
    precision = np.array(
        [
            [2.0, 0.25, 0.0],
            [0.25, 1.4, 0.15],
            [0.0, 0.15, 1.1],
        ],
        dtype=np.float64,
    )
    fisher = np.array(
        [
            [1.9, 0.12, 0.0],
            [0.12, 0.95, 0.08],
            [0.0, 0.08, 1.45],
        ],
        dtype=np.float64,
    )
    return theta, target, precision, fisher


def test_quadratic_free_energy_gradient_matches_finite_difference() -> None:
    theta, target, precision, _fisher = _fixture()

    analytic = precision @ (theta - target)
    finite_difference = finite_difference_quadratic_gradient(
        theta,
        target,
        precision,
        step=2.5e-6,
    )

    np.testing.assert_allclose(analytic, finite_difference, atol=2e-10, rtol=2e-10)
    assert quadratic_free_energy(theta, target, precision) > 0.0


def test_natural_gradient_flow_solves_fisher_preconditioned_descent() -> None:
    theta, target, precision, fisher = _fixture()
    gradient = precision @ (theta - target)
    config = NaturalGradientConfig(eta_L=0.37)

    drift = natural_gradient_flow(gradient, fisher, config=config)

    expected = -0.37 * np.linalg.solve(fisher, gradient)
    np.testing.assert_allclose(drift, expected, atol=1e-12, rtol=1e-12)


def test_natural_gradient_fixture_consumes_spec_and_reports_nulls() -> None:
    theta, target, precision, fisher = _fixture()
    config = NaturalGradientConfig(eta_L=0.37, finite_difference_step=2.5e-6)

    result = validate_upde_natural_gradient_fixture(
        theta, target, precision, fisher, config=config
    )

    assert result.spec_key == "upde.natural_gradient"
    assert result.validation_protocol == "paper0.upde.natural_gradient.fim_free_energy"
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.source_equation_ids == ("EQ0042",)
    assert result.gradient_error_linf < 1e-9
    assert result.metric_condition_number > 1.0
    assert result.null_controls["euclidean_vs_natural_l2"] > 0.0
    assert result.null_controls["identity_fim_matches_euclidean_linf"] < 1e-12
    assert result.null_controls["regularised_singular_metric_linf"] > 0.0


def test_natural_gradient_fixture_rejects_non_positive_definite_metric() -> None:
    theta, target, precision, fisher = _fixture()
    non_pd = fisher.copy()
    non_pd[0, 0] = -1.0

    with pytest.raises(ValueError, match="positive definite"):
        validate_upde_natural_gradient_fixture(theta, target, precision, non_pd)


def test_natural_gradient_fixture_rejects_invalid_shapes_and_nonfinite_values() -> None:
    theta, target, precision, fisher = _fixture()
    bad_theta = theta.copy()
    bad_theta[0] = np.nan

    with pytest.raises(ValueError, match="theta must contain only finite values"):
        validate_upde_natural_gradient_fixture(bad_theta, target, precision, fisher)

    with pytest.raises(ValueError, match="precision_matrix must have shape"):
        validate_upde_natural_gradient_fixture(theta, target, precision[:2, :2], fisher)

    with pytest.raises(ValueError, match="eta_L must be finite and positive"):
        validate_upde_natural_gradient_fixture(
            theta,
            target,
            precision,
            fisher,
            config=NaturalGradientConfig(eta_L=0.0),
        )
