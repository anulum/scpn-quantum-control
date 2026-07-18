# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for the KYMA Kuramoto dynamics
"""Correctness of the differentiable Kuramoto RK4 integrator and readout."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")  # KYMA probe requires the optional [jax] extra, absent from the CI lane


import jax.numpy as jnp
import numpy as np
import pytest

from scpn_quantum_control.benchmarks.kyma.dynamics import (
    cluster_order_parameter,
    integrate_kuramoto,
    kuramoto_rhs,
)


def test_order_parameter_in_phase_is_one() -> None:
    theta = jnp.array([[0.3, 0.3, 0.3, 0.3]])
    r = cluster_order_parameter(theta, jnp.array([0, 1, 2, 3]))
    assert float(r[0]) == pytest.approx(1.0, abs=1e-5)


def test_order_parameter_balanced_anti_phase_is_zero() -> None:
    theta = jnp.array([[0.0, jnp.pi, 0.0, jnp.pi]])
    r = cluster_order_parameter(theta, jnp.array([0, 1, 2, 3]))
    assert float(r[0]) == pytest.approx(0.0, abs=1e-5)


def test_order_parameter_bounded_unit_interval() -> None:
    rng = np.random.default_rng(1)
    theta = jnp.asarray(rng.uniform(-np.pi, np.pi, size=(32, 8)))
    r = np.asarray(cluster_order_parameter(theta, jnp.arange(8)))
    assert np.all(r >= -1e-5) and np.all(r <= 1.0 + 1e-5)


def test_two_oscillators_synchronise_under_positive_coupling() -> None:
    theta0 = jnp.array([[0.0, 2.0]])
    omega = jnp.array([[1.0, 1.0]])
    coupling = jnp.array([[0.0, 2.0], [2.0, 0.0]])
    final = integrate_kuramoto(theta0, omega, coupling, dt=0.05, steps=200)
    r = cluster_order_parameter(final, jnp.array([0, 1]))
    assert float(r[0]) == pytest.approx(1.0, abs=1e-3)


def test_uncoupled_oscillators_drift_by_omega_t() -> None:
    # Zero coupling → pure phase accrual ω·T (RK4 exact for a constant RHS).
    theta0 = jnp.array([[0.0, 0.0]])
    omega = jnp.array([[1.0, -0.5]])
    coupling = jnp.zeros((2, 2))
    final = np.asarray(integrate_kuramoto(theta0, omega, coupling, dt=0.01, steps=100))
    # T = 1.0 → phases 1.0 and -0.5 (wrapped to (-π, π]).
    assert final[0, 0] == pytest.approx(1.0, abs=1e-4)
    assert final[0, 1] == pytest.approx(-0.5, abs=1e-4)


def test_rhs_zero_when_aligned_and_no_drive() -> None:
    theta = jnp.array([[0.7, 0.7, 0.7]])
    coupling = jnp.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    rhs = kuramoto_rhs(theta, jnp.zeros((1, 3)), coupling)
    assert np.allclose(np.asarray(rhs), 0.0, atol=1e-5)


def test_final_phases_wrapped_to_pi_interval() -> None:
    theta0 = jnp.array([[0.0]])
    omega = jnp.array([[10.0]])  # large drift → must wrap
    final = np.asarray(integrate_kuramoto(theta0, omega, jnp.zeros((1, 1)), dt=0.1, steps=50))
    assert -np.pi - 1e-4 <= final[0, 0] <= np.pi + 1e-4
