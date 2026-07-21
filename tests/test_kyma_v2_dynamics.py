# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — tests for KYMA v2 batched Kuramoto dynamics
"""Tests for the per-trial gated-coupling Kuramoto integrator and readouts."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from scpn_quantum_control.benchmarks.kyma_v2 import dynamics  # noqa: E402


def test_rhs_zero_coupling_is_pure_drive() -> None:
    theta = jnp.zeros((3, 4))
    omega = jnp.asarray(np.arange(12.0).reshape(3, 4))
    coupling = jnp.zeros((3, 4, 4))
    rhs = dynamics.kuramoto_rhs_batched(theta, omega, coupling)
    assert np.allclose(np.asarray(rhs), np.asarray(omega))


def test_rhs_per_trial_coupling_is_independent() -> None:
    # Two trials, different couplings → different interaction terms.
    theta = jnp.asarray([[0.0, np.pi / 2], [0.0, np.pi / 2]])
    omega = jnp.zeros((2, 2))
    coupling = jnp.asarray([[[0.0, 1.0], [1.0, 0.0]], [[0.0, 2.0], [2.0, 0.0]]])
    rhs = np.asarray(dynamics.kuramoto_rhs_batched(theta, omega, coupling))
    # Oscillator 0 pulled toward oscillator 1 (θ_1 − θ_0 = π/2): sin = +1.
    assert rhs[1, 0] == pytest.approx(2.0 * rhs[0, 0], rel=1e-6)


def test_attractive_coupling_synchronises() -> None:
    rng = np.random.default_rng(0)
    theta0 = jnp.asarray(rng.uniform(-np.pi, np.pi, size=(1, 6)))
    omega = jnp.zeros((1, 6))
    coupling = jnp.asarray((np.ones((6, 6)) - np.eye(6))[None])  # all-to-all attractive
    final = dynamics.integrate_kuramoto_batched(theta0, omega, coupling, dt=0.1, steps=100)
    r = np.asarray(dynamics.order_parameter(final, jnp.arange(6)))[0]
    assert r > 0.99


def test_zero_coupling_preserves_relative_phases() -> None:
    theta0 = jnp.asarray([[0.0, 1.0, 2.0]])
    omega = jnp.zeros((1, 3))
    coupling = jnp.zeros((1, 3, 3))
    final = np.asarray(
        dynamics.integrate_kuramoto_batched(theta0, omega, coupling, dt=0.1, steps=10)
    )
    assert np.allclose(final, np.asarray(theta0), atol=1e-6)


def test_order_parameter_in_and_anti_phase() -> None:
    in_phase = jnp.zeros((1, 4))
    anti = jnp.asarray([[0.0, 0.0, np.pi, np.pi]])
    assert np.asarray(dynamics.order_parameter(in_phase, jnp.arange(4)))[0] == pytest.approx(1.0)
    assert np.asarray(dynamics.order_parameter(anti, jnp.arange(4)))[0] == pytest.approx(
        0.0, abs=1e-6
    )


@pytest.mark.parametrize("n_bins", [2, 4, 8])
def test_phase_label_range_and_binning(n_bins: int) -> None:
    # Phases spread across the circle → labels cover the full range and stay in bounds.
    phi = jnp.asarray([np.linspace(-np.pi + 0.01, np.pi - 0.01, 32)]).T.reshape(1, 32)
    labels = np.asarray(dynamics.phase_label(phi, readout_oscillator=0, n_bins=n_bins))
    # single readout oscillator (index 0): one label.
    assert labels.shape == (1,)
    assert 0 <= int(labels[0]) < n_bins


def test_phase_label_specific_bins() -> None:
    # 4 bins of width π/2; φ = π/4 → bin 0, φ = 3π/4 → bin 1, φ = 5π/4 → bin 2.
    theta = jnp.asarray([[np.pi / 4], [3 * np.pi / 4], [5 * np.pi / 4]])
    labels = np.asarray(dynamics.phase_label(theta, readout_oscillator=0, n_bins=4))
    assert labels.tolist() == [0, 1, 2]


def test_phase_label_wraps_negative_angles() -> None:
    # −π/4 mod 2π = 7π/4 → bin 3 (last) of 4.
    theta = jnp.asarray([[-np.pi / 4]])
    label = int(np.asarray(dynamics.phase_label(theta, readout_oscillator=0, n_bins=4))[0])
    assert label == 3
