# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the Wirtinger (CR) calculus module
"""Tests for wirtinger_calculus.py against analytic Wirtinger derivatives."""

import numpy as np
import pytest

from scpn_quantum_control.wirtinger_calculus import (
    holomorphic_gradient,
    is_holomorphic,
    minimise_real_objective,
    real_objective_gradient,
    wirtinger_partials,
)


# --------------------------------------------------------------------------- #
# Wirtinger partials against textbook results
# --------------------------------------------------------------------------- #
def test_holomorphic_square():
    z = np.array([1.3 - 0.7j])
    derivative = wirtinger_partials(lambda v: v[0] ** 2, z)
    assert np.allclose(derivative.df_dz, 2.0 * z, atol=1e-6)
    assert np.allclose(derivative.df_dconj_z, 0.0, atol=1e-6)
    assert derivative.holomorphic_residual < 1e-6


def test_modulus_squared_is_non_holomorphic():
    z = np.array([1.3 - 0.7j])
    derivative = wirtinger_partials(lambda v: np.abs(v[0]) ** 2, z)
    # d/dz |z|^2 = conj(z),  d/dconj_z |z|^2 = z
    assert np.allclose(derivative.df_dz, np.conj(z), atol=1e-6)
    assert np.allclose(derivative.df_dconj_z, z, atol=1e-6)


def test_conjugate_derivative():
    z = np.array([0.4 + 0.9j])
    derivative = wirtinger_partials(lambda v: np.conj(v[0]), z)
    assert np.allclose(derivative.df_dz, 0.0, atol=1e-6)
    assert np.allclose(derivative.df_dconj_z, 1.0, atol=1e-6)


def test_real_part_derivative():
    z = np.array([0.5 - 0.2j])
    derivative = wirtinger_partials(lambda v: np.real(v[0]) + 0.0j, z)
    # Re(z) = (z + conj z) / 2 -> both partials are 1/2.
    assert np.allclose(derivative.df_dz, 0.5, atol=1e-6)
    assert np.allclose(derivative.df_dconj_z, 0.5, atol=1e-6)


def test_multivariate_partials():
    z = np.array([0.5 + 0.2j, -0.3 + 0.9j])
    derivative = wirtinger_partials(lambda v: v[0] ** 2 + v[1] * np.conj(v[1]), z)
    assert np.allclose(derivative.df_dz, [2.0 * z[0], np.conj(z[1])], atol=1e-6)
    assert np.allclose(derivative.df_dconj_z, [0.0, z[1]], atol=1e-6)


def test_wirtinger_product_rule():
    z = np.array([0.7 + 0.3j])

    def f(v):
        return v[0] ** 2

    def g(v):
        return np.conj(v[0]) + 1.0

    df = wirtinger_partials(f, z).df_dz
    dg = wirtinger_partials(g, z).df_dz
    dfg = wirtinger_partials(lambda v: f(v) * g(v), z).df_dz
    # d(fg)/dz = f dg/dz + g df/dz.
    expected = f(z) * dg + g(z) * df
    assert np.allclose(dfg, expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# Holomorphicity test and gradient
# --------------------------------------------------------------------------- #
def test_is_holomorphic():
    z = np.array([1.0 + 0.5j, -0.4 + 0.2j])
    assert is_holomorphic(lambda v: np.exp(v[0]) * v[1], z)
    assert not is_holomorphic(lambda v: np.abs(v[0]) ** 2, z)


def test_holomorphic_gradient_matches_complex_derivative():
    z = np.array([1.0 + 1.0j])
    gradient = holomorphic_gradient(lambda v: v[0] ** 3, z)
    assert np.allclose(gradient, 3.0 * z**2, atol=1e-6)


def test_holomorphic_gradient_rejects_non_holomorphic():
    with pytest.raises(ValueError):
        holomorphic_gradient(lambda v: np.abs(v[0]) ** 2, np.array([1.0 + 1.0j]))


# --------------------------------------------------------------------------- #
# Real-valued objective gradient and descent
# --------------------------------------------------------------------------- #
def test_real_objective_gradient_is_conjugate_of_df_dz():
    z = np.array([0.6 - 0.2j, 0.1 + 0.4j])
    target = np.array([0.8 - 0.3j, -0.4 + 0.6j])

    def loss(v):
        return float(np.sum(np.abs(v - target) ** 2))

    gradient = real_objective_gradient(loss, z)
    # d/dconj_z |z - t|^2 = z - t.
    assert np.allclose(gradient, z - target, atol=1e-6)


def test_complex_descent_converges_to_target():
    target = np.array([0.8 - 0.3j, -0.4 + 0.6j])

    def loss(v):
        return float(np.sum(np.abs(v - target) ** 2))

    result = minimise_real_objective(loss, np.zeros(2, dtype=complex), learning_rate=0.3, steps=80)
    assert result.final_loss < result.loss_history[0]
    assert result.final_loss < 1e-10
    assert np.allclose(result.parameters, target, atol=1e-4)
    assert "claim_boundary" in result.provenance


def test_descent_reduces_holomorphic_modulus_objective():
    # L(z) = |z|^2 has minimum at 0; CR descent must drive z -> 0.
    result = minimise_real_objective(
        lambda v: float(np.sum(np.abs(v) ** 2)),
        np.array([1.5 - 0.8j]),
        learning_rate=0.4,
        steps=60,
    )
    assert np.allclose(result.parameters, 0.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# Fail-closed validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "z",
    [
        np.zeros((2, 2), dtype=complex),
        np.array([], dtype=complex),
        np.array([np.nan + 0j]),
        np.array([1j * np.inf]),
    ],
)
def test_wirtinger_rejects_bad_point(z):
    with pytest.raises(ValueError):
        wirtinger_partials(lambda v: v[0], z)


@pytest.mark.parametrize("step", [0.0, -1e-6, np.inf])
def test_wirtinger_rejects_bad_step(step):
    with pytest.raises(ValueError):
        wirtinger_partials(lambda v: v[0], np.array([1.0 + 0j]), step=step)


def test_is_holomorphic_rejects_negative_tolerance():
    with pytest.raises(ValueError):
        is_holomorphic(lambda v: v[0], np.array([1.0 + 0j]), tolerance=-1.0)


@pytest.mark.parametrize("kwargs", [{"learning_rate": 0.0}, {"learning_rate": -1.0}, {"steps": 0}])
def test_minimise_rejects_bad_args(kwargs):
    with pytest.raises(ValueError):
        minimise_real_objective(
            lambda v: float(np.sum(np.abs(v) ** 2)), np.array([1.0 + 0j]), **kwargs
        )
