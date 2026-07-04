# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the Kuramoto frequency-locking diagnostics
r"""Tests for :mod:`oscillatools.accel.kuramoto_frequency_order`.

The effective frequency is checked against three exact analytic facts: a constant-rate
trajectory has frequency equal to that rate; the wrapped-difference estimate is invariant
to storing the trajectory modulo :math:`2\pi`; and it telescopes to the end-to-end slope
when no single step wraps. The synchronisation index is checked to vanish for a
frequency-locked ensemble and to equal the natural-frequency spread for a linear drift,
and the locked fraction is checked on planted single-, split- and partial-locking states.
A supercritical networked-Kuramoto integration confirms the roadmap acceptance — the index
collapses once the coupling locks a frequency-spread ensemble.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscillatools.accel.diff_kuramoto_rk4 import kuramoto_rk4_trajectory
from oscillatools.accel.kuramoto_frequency_order import (
    FrequencyOrder,
    effective_frequencies,
    frequency_locked_fraction,
    frequency_order_diagnostics,
    frequency_synchronisation_index,
    frequency_synchronisation_index_gradient,
)


def _wrap(values: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * values))


def _linear_trajectory(
    rates: np.ndarray, offsets: np.ndarray, *, dt: float, steps: int
) -> np.ndarray:
    """A trajectory whose oscillator ``i`` advances at a constant rate ``rates[i]``."""
    grid = np.arange(steps + 1) * dt
    return rates[None, :] * grid[:, None] + offsets[None, :]


# --------------------------------------------------------------------------- effective frequency


def test_constant_rate_recovers_the_rate() -> None:
    rates = np.array([1.3, 1.3, 1.3, 1.3])
    offsets = np.random.default_rng(0).uniform(0.0, 2.0 * np.pi, 4)
    trajectory = _linear_trajectory(rates, offsets, dt=0.01, steps=500)
    assert np.allclose(effective_frequencies(trajectory, dt=0.01), 1.3, atol=1e-10)


def test_distinct_rates_recovered_per_oscillator() -> None:
    rates = np.array([-2.0, -0.5, 0.0, 0.75, 3.1])
    offsets = np.random.default_rng(1).uniform(0.0, 2.0 * np.pi, 5)
    trajectory = _linear_trajectory(rates, offsets, dt=0.005, steps=800)
    assert np.allclose(effective_frequencies(trajectory, dt=0.005), rates, atol=1e-9)


def test_wrapped_storage_gives_same_frequencies() -> None:
    rates = np.array([-1.1, 0.4, 2.7])
    offsets = np.random.default_rng(2).uniform(0.0, 2.0 * np.pi, 3)
    trajectory = _linear_trajectory(rates, offsets, dt=0.01, steps=600)
    raw = effective_frequencies(trajectory, dt=0.01)
    wrapped = effective_frequencies(_wrap(trajectory), dt=0.01)
    assert np.allclose(raw, wrapped, atol=1e-9)


def test_telescopes_to_endpoint_slope_when_no_step_wraps() -> None:
    rates = np.array([0.3, -0.7, 1.2])
    offsets = np.zeros(3)
    trajectory = _linear_trajectory(rates, offsets, dt=0.01, steps=400)
    span = 400 * 0.01
    endpoint = (trajectory[-1] - trajectory[0]) / span
    assert np.allclose(effective_frequencies(trajectory, dt=0.01), endpoint, atol=1e-12)


def test_frequency_scales_inversely_with_dt() -> None:
    # One fixed phase array (identical per-step advance) read at two steps: doubling the
    # interpretation step halves the reported frequency.
    rates = np.array([0.5, 1.0, 2.0])
    phases = rates[None, :] * np.arange(401)[:, None] * 0.01
    assert np.allclose(
        effective_frequencies(phases, dt=0.01),
        2.0 * effective_frequencies(phases, dt=0.02),
        atol=1e-9,
    )


# --------------------------------------------------------------------------- synchronisation index


def test_index_vanishes_for_frequency_locked_state() -> None:
    rates = np.full(8, 2.4)
    offsets = np.random.default_rng(3).uniform(0.0, 2.0 * np.pi, 8)
    trajectory = _linear_trajectory(rates, offsets, dt=0.01, steps=500)
    assert frequency_synchronisation_index(trajectory, dt=0.01) < 1e-12


def test_index_equals_population_spread_for_linear_drift() -> None:
    rates = np.random.default_rng(4).standard_normal(16)
    offsets = np.random.default_rng(5).uniform(0.0, 2.0 * np.pi, 16)
    trajectory = _linear_trajectory(rates, offsets, dt=0.005, steps=900)
    assert np.isclose(
        frequency_synchronisation_index(trajectory, dt=0.005), float(np.std(rates)), atol=1e-9
    )


def test_index_grows_monotonically_with_spread() -> None:
    base = np.random.default_rng(6).standard_normal(12)
    offsets = np.zeros(12)
    previous = -1.0
    for spread in (0.0, 0.5, 1.0, 2.0, 4.0):
        trajectory = _linear_trajectory(spread * base, offsets, dt=0.01, steps=400)
        index = frequency_synchronisation_index(trajectory, dt=0.01)
        assert index >= previous - 1e-12
        previous = index


def test_index_is_zero_for_single_oscillator() -> None:
    trajectory = _linear_trajectory(np.array([1.7]), np.array([0.2]), dt=0.01, steps=300)
    assert frequency_synchronisation_index(trajectory, dt=0.01) == 0.0


# --------------------------------------------------------------------------- index gradient


def _smooth_trajectory(rates: np.ndarray, *, dt: float, steps: int, seed: int) -> np.ndarray:
    """A drift-plus-bounded-ripple trajectory with no per-step advance near ``±π``."""
    rng = np.random.default_rng(seed)
    grid = np.arange(steps + 1) * dt
    ripple = 0.05 * np.sin(
        2.0 * np.pi * grid[:, None] + rng.uniform(0.0, 1.0, rates.size)[None, :]
    )
    return (
        rates[None, :] * grid[:, None]
        + ripple
        + rng.uniform(0.0, 2.0 * np.pi, rates.size)[None, :]
    )


def test_index_gradient_is_nonzero_only_at_the_endpoints() -> None:
    rates = np.random.default_rng(20).standard_normal(9)
    trajectory = _smooth_trajectory(rates, dt=0.01, steps=300, seed=21)
    gradient = frequency_synchronisation_index_gradient(trajectory, dt=0.01)
    assert gradient.shape == trajectory.shape
    assert np.allclose(gradient[1:-1], 0.0, atol=1e-14)
    assert np.allclose(gradient[0], -gradient[-1], atol=1e-15)


def test_index_gradient_matches_central_difference() -> None:
    rates = np.random.default_rng(22).standard_normal(7)
    trajectory = _smooth_trajectory(rates, dt=0.01, steps=120, seed=23)
    analytic = frequency_synchronisation_index_gradient(trajectory, dt=0.01)
    numeric = np.zeros_like(trajectory)
    step = 1e-6
    flat_numeric = numeric.reshape(-1)
    base = trajectory.reshape(-1)
    for index in range(base.size):
        forward = base.copy()
        forward[index] += step
        backward = base.copy()
        backward[index] -= step
        flat_numeric[index] = (
            frequency_synchronisation_index(forward.reshape(trajectory.shape), dt=0.01)
            - frequency_synchronisation_index(backward.reshape(trajectory.shape), dt=0.01)
        ) / (2.0 * step)
    assert np.allclose(analytic, numeric, atol=1e-7)


def test_index_gradient_vanishes_for_a_frequency_locked_state() -> None:
    # An integrated, supercritically coupled ensemble is frequency-locked to floating
    # precision, so the standard deviation sits at its non-differentiable point.
    count = 10
    rng = np.random.default_rng(24)
    omega = rng.normal(0.0, 1.0, count)
    theta0 = rng.uniform(0.0, 2.0 * np.pi, count)
    trajectory = kuramoto_rk4_trajectory(
        theta0, omega, np.full((count, count), 8.0 / count), 0.01, 3000
    )
    gradient = frequency_synchronisation_index_gradient(trajectory[1500:], dt=0.01)
    assert np.array_equal(gradient, np.zeros_like(gradient))


def test_index_gradient_vanishes_for_a_single_oscillator() -> None:
    trajectory = _linear_trajectory(np.array([1.7]), np.array([0.4]), dt=0.01, steps=200)
    gradient = frequency_synchronisation_index_gradient(trajectory, dt=0.01)
    assert np.array_equal(gradient, np.zeros_like(gradient))


def test_index_gradient_rejects_non_two_dimensional_phases() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        frequency_synchronisation_index_gradient(np.zeros(5), dt=0.01)


# --------------------------------------------------------------------------- locked fraction


def test_locked_fraction_is_one_for_fully_locked() -> None:
    rates = np.full(10, 1.0)
    offsets = np.random.default_rng(7).uniform(0.0, 2.0 * np.pi, 10)
    trajectory = _linear_trajectory(rates, offsets, dt=0.01, steps=400)
    assert frequency_locked_fraction(trajectory, dt=0.01) == 1.0


def test_locked_fraction_is_zero_for_symmetric_split() -> None:
    # Half advance at +1, half at -1; the mean is 0 and neither group sits near it.
    rates = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    offsets = np.zeros(6)
    trajectory = _linear_trajectory(rates, offsets, dt=0.01, steps=400)
    assert frequency_locked_fraction(trajectory, dt=0.01, tolerance=1e-3) == 0.0


def test_locked_fraction_counts_the_cluster_about_the_mean() -> None:
    # Nine oscillators at 1.0 plus one modest outlier at 2.0: the mean is 1.1, so the nine
    # sit within tolerance and the single outlier does not.
    rates = np.concatenate([np.full(9, 1.0), np.array([2.0])])
    offsets = np.zeros(10)
    trajectory = _linear_trajectory(rates, offsets, dt=0.005, steps=800)
    fraction = frequency_locked_fraction(trajectory, dt=0.005, tolerance=0.5)
    assert fraction == pytest.approx(0.9)


def test_locked_fraction_is_outlier_sensitive_through_the_mean() -> None:
    # A large outlier drags the mean out of the cluster, so locking-to-the-mean reports no
    # locked oscillators — the documented behaviour of a mean-referenced index.
    rates = np.concatenate([np.full(9, 1.0), np.array([20.0])])
    offsets = np.zeros(10)
    trajectory = _linear_trajectory(rates, offsets, dt=0.005, steps=800)
    assert frequency_locked_fraction(trajectory, dt=0.005, tolerance=0.5) == 0.0


def test_locked_fraction_is_one_for_a_wide_tolerance() -> None:
    rates = np.random.default_rng(8).standard_normal(7)
    offsets = np.zeros(7)
    trajectory = _linear_trajectory(rates, offsets, dt=0.01, steps=300)
    assert frequency_locked_fraction(trajectory, dt=0.01, tolerance=1e6) == 1.0


# --------------------------------------------------------------------------- bundled diagnostics


def test_diagnostics_bundle_matches_individual_helpers() -> None:
    rates = np.random.default_rng(9).standard_normal(11)
    offsets = np.random.default_rng(10).uniform(0.0, 2.0 * np.pi, 11)
    trajectory = _linear_trajectory(rates, offsets, dt=0.01, steps=500)
    bundle = frequency_order_diagnostics(trajectory, dt=0.01, tolerance=0.25)
    assert isinstance(bundle, FrequencyOrder)
    assert np.allclose(bundle.effective_frequencies, effective_frequencies(trajectory, dt=0.01))
    assert bundle.synchronisation_index == pytest.approx(
        frequency_synchronisation_index(trajectory, dt=0.01)
    )
    assert bundle.locked_fraction == pytest.approx(
        frequency_locked_fraction(trajectory, dt=0.01, tolerance=0.25)
    )


def test_frequency_order_is_frozen() -> None:
    bundle = FrequencyOrder(
        effective_frequencies=np.zeros(3), synchronisation_index=0.0, locked_fraction=1.0
    )
    with pytest.raises(AttributeError):
        bundle.locked_fraction = 0.5  # type: ignore[misc]


# --------------------------------------------------------------------------- integrated trajectory


def test_supercritical_coupling_collapses_the_index() -> None:
    # A frequency-spread ensemble: weak coupling lets them drift (large index); strong
    # coupling locks them to a common rate (small index) — the roadmap acceptance.
    count = 12
    rng = np.random.default_rng(11)
    omega = rng.normal(0.0, 1.0, count)
    theta0 = rng.uniform(0.0, 2.0 * np.pi, count)
    dt, steps = 0.01, 4000
    burn = 2000

    weak = kuramoto_rk4_trajectory(theta0, omega, np.full((count, count), 0.05 / count), dt, steps)
    strong = kuramoto_rk4_trajectory(
        theta0, omega, np.full((count, count), 8.0 / count), dt, steps
    )

    weak_index = frequency_synchronisation_index(weak[burn:], dt=dt)
    strong_index = frequency_synchronisation_index(strong[burn:], dt=dt)

    assert strong_index < 1e-2
    assert weak_index > strong_index
    # Strong coupling locks essentially the whole ensemble to the mean frequency.
    assert frequency_locked_fraction(strong[burn:], dt=dt, tolerance=1e-2) == 1.0


# --------------------------------------------------------------------------- validation


def test_rejects_non_two_dimensional_phases() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        effective_frequencies(np.zeros(5), dt=0.01)


def test_rejects_too_few_time_samples() -> None:
    with pytest.raises(ValueError, match="at least two time samples"):
        effective_frequencies(np.zeros((1, 4)), dt=0.01)


def test_rejects_zero_oscillators() -> None:
    with pytest.raises(ValueError, match="at least one oscillator"):
        effective_frequencies(np.zeros((5, 0)), dt=0.01)


def test_rejects_non_positive_dt() -> None:
    with pytest.raises(ValueError, match="dt must be strictly positive"):
        effective_frequencies(np.zeros((3, 2)), dt=0.0)


def test_rejects_negative_tolerance_in_fraction() -> None:
    trajectory = _linear_trajectory(np.array([1.0, 2.0]), np.zeros(2), dt=0.01, steps=100)
    with pytest.raises(ValueError, match="tolerance must be non-negative"):
        frequency_locked_fraction(trajectory, dt=0.01, tolerance=-1e-9)


def test_rejects_negative_tolerance_in_diagnostics() -> None:
    trajectory = _linear_trajectory(np.array([1.0, 2.0]), np.zeros(2), dt=0.01, steps=100)
    with pytest.raises(ValueError, match="tolerance must be non-negative"):
        frequency_order_diagnostics(trajectory, dt=0.01, tolerance=-1.0)
