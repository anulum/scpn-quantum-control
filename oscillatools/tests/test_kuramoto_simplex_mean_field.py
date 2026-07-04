# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the k-simplex mean-field Kuramoto force and Jacobian
r"""Tests for :mod:`oscillatools.accel.kuramoto_simplex_mean_field`.

The all-to-all ``p``-simplex force is checked against its closed form ``K r^p sin(pψ − pθ_i)`` and
against the lower-order reductions — the classic Kuramoto mean field at ``p = 1`` and the triadic
force at ``p = 2`` (the roadmap acceptance, an exact match to the standalone triadic module). The
Jacobian is checked against a central finite difference at several orders, against the triadic
Jacobian at ``p = 2``, and for the zero row sums of the global-phase Goldstone mode.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscillatools.accel.kuramoto_simplex_mean_field import (
    simplex_mean_field_force,
    simplex_mean_field_jacobian,
)
from oscillatools.accel.mean_phase_observables import mean_phase
from oscillatools.accel.order_parameter_observables import order_parameter
from oscillatools.accel.triadic_mean_field import (
    triadic_mean_field_force,
    triadic_mean_field_jacobian,
)


def _central_difference_jacobian(theta, coupling, order, step=1e-6):
    count = theta.size
    numeric = np.empty((count, count))
    for column in range(count):
        forward = theta.copy()
        forward[column] += step
        backward = theta.copy()
        backward[column] -= step
        numeric[:, column] = (
            simplex_mean_field_force(forward, coupling, order)
            - simplex_mean_field_force(backward, coupling, order)
        ) / (2.0 * step)
    return numeric


# --------------------------------------------------------------------------- force


def test_force_matches_closed_form() -> None:
    rng = np.random.default_rng(0)
    theta = rng.uniform(0.0, 2.0 * np.pi, 7)
    coupling, order = 1.4, 3
    radius, phase = order_parameter(theta), mean_phase(theta)
    expected = coupling * radius**order * np.sin(order * phase - order * theta)
    assert np.allclose(simplex_mean_field_force(theta, coupling, order), expected)


def test_first_order_reduces_to_classic_mean_field() -> None:
    rng = np.random.default_rng(1)
    theta = rng.uniform(0.0, 2.0 * np.pi, 8)
    coupling = 2.1
    radius, phase = order_parameter(theta), mean_phase(theta)
    classic = coupling * radius * np.sin(phase - theta)
    assert np.allclose(simplex_mean_field_force(theta, coupling, 1), classic)


def test_second_order_equals_triadic_force() -> None:
    rng = np.random.default_rng(2)
    theta = rng.uniform(0.0, 2.0 * np.pi, 9)
    coupling = 1.7
    assert np.allclose(
        simplex_mean_field_force(theta, coupling, 2), triadic_mean_field_force(theta, coupling)
    )


def test_force_rejects_non_vector_theta() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        simplex_mean_field_force(np.zeros((2, 2)), 1.0, 2)


def test_force_rejects_non_positive_order() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        simplex_mean_field_force(np.zeros(3), 1.0, 0)


def test_force_on_empty_is_empty() -> None:
    result = simplex_mean_field_force(np.empty(0), 1.0, 2)
    assert result.shape == (0,)


# --------------------------------------------------------------------------- Jacobian


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_jacobian_matches_central_difference(order: int) -> None:
    rng = np.random.default_rng(order)
    theta = rng.uniform(0.0, 2.0 * np.pi, 6)
    coupling = 1.3
    analytic = simplex_mean_field_jacobian(theta, coupling, order)
    numeric = _central_difference_jacobian(theta, coupling, order)
    assert np.allclose(analytic, numeric, atol=1e-7)


def test_jacobian_second_order_equals_triadic_jacobian() -> None:
    rng = np.random.default_rng(5)
    theta = rng.uniform(0.0, 2.0 * np.pi, 7)
    coupling = 1.9
    assert np.allclose(
        simplex_mean_field_jacobian(theta, coupling, 2),
        triadic_mean_field_jacobian(theta, coupling),
    )


def test_jacobian_rows_sum_to_zero() -> None:
    rng = np.random.default_rng(6)
    theta = rng.uniform(0.0, 2.0 * np.pi, 8)
    for order in (1, 2, 3):
        jacobian = simplex_mean_field_jacobian(theta, 1.5, order)
        assert np.allclose(jacobian.sum(axis=1), 0.0, atol=1e-12)


def test_jacobian_rejects_non_vector_theta() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        simplex_mean_field_jacobian(np.zeros((3, 3)), 1.0, 2)


def test_jacobian_rejects_non_positive_order() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        simplex_mean_field_jacobian(np.zeros(3), 1.0, -2)


def test_jacobian_on_empty_is_zero_by_zero() -> None:
    result = simplex_mean_field_jacobian(np.empty(0), 1.0, 2)
    assert result.shape == (0, 0)
