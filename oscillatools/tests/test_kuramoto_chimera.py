# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the chimera and metastability diagnostics
r"""Tests for :mod:`oscillatools.accel.kuramoto_chimera`.

The roadmap acceptance is checked directly: the chimera index and both metastability indices
vanish on a fully synchronised trajectory and the chimera index is positive on a constructed
state where one community locks while another drifts. The community order parameter is checked
to read ``1`` on a synchronised group and below ``1`` on a spread one, the metastability is
checked to grow with a wandering global coherence, and both index gradients are checked against
central finite differences including the zero subgradient at an incoherent point. The community
partition validation branches are covered explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscillatools.accel.diff_kuramoto_rk4 import kuramoto_rk4_trajectory
from oscillatools.accel.kuramoto_chimera import (
    ChimeraDiagnostics,
    chimera_diagnostics,
    chimera_index,
    chimera_index_gradient,
    community_metastability,
    community_order_parameters,
    metastability_index,
    metastability_index_gradient,
)

_TWO_GROUPS = [np.arange(0, 6), np.arange(6, 12)]


def _synchronised(steps: int, count: int, *, dt: float = 0.02) -> np.ndarray:
    """Build a fully phase-locked trajectory in which every oscillator shares one rotating phase."""
    grid = np.arange(steps + 1) * dt
    return np.repeat(grid[:, None] + 0.3, count, axis=1)


def _constructed_chimera(steps: int, *, seed: int) -> np.ndarray:
    """Group 0 synchronised, group 1 incoherent and time-varying."""
    rng = np.random.default_rng(seed)
    grid = np.arange(steps + 1) * 0.02
    trajectory = np.empty((steps + 1, 12))
    trajectory[:, :6] = grid[:, None] + 0.1
    spread = rng.uniform(0.0, 2.0 * np.pi, 6)
    wander = 0.7 * np.sin(np.linspace(0.0, 10.0, steps + 1))[:, None] * np.arange(1, 7)[None, :]
    trajectory[:, 6:] = grid[:, None] + spread[None, :] + wander
    return trajectory


def _smooth_generic(count: int, steps: int, *, seed: int) -> np.ndarray:
    """Build a generic differentiable trajectory away from any incoherent point."""
    rng = np.random.default_rng(seed)
    grid = np.arange(steps + 1) * 0.02
    drift = rng.normal(0.0, 1.0, count)[None, :] * grid[:, None]
    ripple = 0.1 * np.sin(2.0 * np.pi * grid[:, None] + np.arange(count)[None, :])
    return drift + ripple + rng.uniform(0.0, 2.0 * np.pi, count)[None, :]


# --------------------------------------------------------------------------- community order parameter


def test_community_order_parameters_shape_and_range() -> None:
    """Community order parameters have one column per community and lie in [0, 1]."""
    trajectory = _constructed_chimera(200, seed=0)
    phi = community_order_parameters(trajectory, _TWO_GROUPS)
    assert phi.shape == (201, 2)
    assert np.all(phi >= 0.0) and np.all(phi <= 1.0 + 1e-12)


def test_synchronised_group_reads_unit_coherence() -> None:
    """A synchronised group reads near-unit coherence and a spread group less."""
    trajectory = _constructed_chimera(200, seed=1)
    phi = community_order_parameters(trajectory, _TWO_GROUPS)
    assert phi[:, 0].mean() > 0.99  # synchronised cluster
    assert phi[:, 1].mean() < 0.9  # spread cluster


# --------------------------------------------------------------------------- chimera index


def test_chimera_index_is_zero_for_full_synchrony() -> None:
    """Full synchrony has chimera index zero."""
    trajectory = _synchronised(300, 12)
    assert chimera_index(trajectory, _TWO_GROUPS) == pytest.approx(0.0, abs=1e-18)


def test_chimera_index_is_positive_for_a_constructed_chimera() -> None:
    """A constructed chimera has a positive index."""
    trajectory = _constructed_chimera(300, seed=2)
    assert chimera_index(trajectory, _TWO_GROUPS) > 1e-3


def test_chimera_index_is_zero_for_a_single_community() -> None:
    """A single community has chimera index zero."""
    trajectory = _constructed_chimera(200, seed=3)
    assert chimera_index(trajectory, [np.arange(12)]) == 0.0


# --------------------------------------------------------------------------- metastability


def test_metastability_is_zero_for_full_synchrony() -> None:
    """Full synchrony has zero metastability."""
    trajectory = _synchronised(300, 12)
    assert metastability_index(trajectory) == pytest.approx(0.0, abs=1e-18)


def test_metastability_is_positive_for_a_wandering_global_coherence() -> None:
    """A wandering global coherence has positive metastability."""
    trajectory = _constructed_chimera(300, seed=4)
    assert metastability_index(trajectory) > 1e-4


def test_metastability_is_zero_for_a_single_time_sample() -> None:
    """A single time sample has zero metastability."""
    trajectory = _constructed_chimera(300, seed=5)[:1]
    assert metastability_index(trajectory) == 0.0


def test_community_metastability_zero_for_synchrony_positive_for_wandering() -> None:
    """Community metastability is zero for synchrony and positive for wandering."""
    assert community_metastability(_synchronised(300, 12), _TWO_GROUPS) == pytest.approx(
        0.0, abs=1e-18
    )
    assert community_metastability(_constructed_chimera(300, seed=6), _TWO_GROUPS) > 1e-4


# --------------------------------------------------------------------------- integrated trajectory


def test_supercritical_integration_is_neither_chimeric_nor_metastable() -> None:
    """A supercritical integration is neither chimeric nor metastable."""
    count = 12
    rng = np.random.default_rng(7)
    omega = rng.normal(0.0, 1.0, count)
    theta0 = rng.uniform(0.0, 2.0 * np.pi, count)
    trajectory = kuramoto_rk4_trajectory(
        theta0, omega, np.full((count, count), 8.0 / count), 0.01, 3000
    )
    tail = trajectory[1500:]
    assert chimera_index(tail, _TWO_GROUPS) < 1e-2
    assert metastability_index(tail) < 1e-3


# --------------------------------------------------------------------------- gradients


def _central_difference(function, trajectory: np.ndarray, step: float = 1e-6) -> np.ndarray:
    numeric = np.zeros_like(trajectory)
    flat_numeric = numeric.reshape(-1)
    base = trajectory.reshape(-1)
    for index in range(base.size):
        forward = base.copy()
        forward[index] += step
        backward = base.copy()
        backward[index] -= step
        flat_numeric[index] = (
            function(forward.reshape(trajectory.shape))
            - function(backward.reshape(trajectory.shape))
        ) / (2.0 * step)
    return numeric


def test_chimera_gradient_matches_central_difference() -> None:
    """The chimera gradient agrees with central differences."""
    trajectory = _smooth_generic(8, 40, seed=8)
    groups = [np.arange(0, 4), np.arange(4, 8)]
    analytic = chimera_index_gradient(trajectory, groups)
    numeric = _central_difference(lambda p: chimera_index(p, groups), trajectory)
    assert analytic.shape == trajectory.shape
    assert np.allclose(analytic, numeric, atol=1e-7)


def test_metastability_gradient_matches_central_difference() -> None:
    """The metastability gradient agrees with central differences."""
    trajectory = _smooth_generic(8, 40, seed=9)
    analytic = metastability_index_gradient(trajectory)
    numeric = _central_difference(metastability_index, trajectory)
    assert analytic.shape == trajectory.shape
    assert np.allclose(analytic, numeric, atol=1e-7)


def test_gradient_rows_for_oscillators_outside_communities_are_zero() -> None:
    # A partition that omits oscillators 6 and 7: their gradient columns must stay zero.
    """Oscillators outside every community get zero gradient."""
    trajectory = _smooth_generic(8, 30, seed=10)
    groups = [np.arange(0, 3), np.arange(3, 6)]
    gradient = chimera_index_gradient(trajectory, groups)
    assert np.array_equal(gradient[:, 6:], np.zeros((trajectory.shape[0], 2)))


def test_chimera_gradient_uses_zero_subgradient_at_incoherent_community() -> None:
    # A community whose two members are exactly antiphase has Phi = 0 at every time; the
    # gradient must fall back to the finite zero subgradient rather than divide by zero.
    """An incoherent community gets a zero subgradient."""
    grid = np.arange(21) * 0.05
    trajectory = np.empty((21, 4))
    trajectory[:, 0] = grid
    trajectory[:, 1] = grid + np.pi  # antiphase with member 0 -> incoherent community
    trajectory[:, 2] = grid + 0.05
    trajectory[:, 3] = grid + 0.1
    gradient = chimera_index_gradient(trajectory, [np.arange(0, 2), np.arange(2, 4)])
    assert np.all(np.isfinite(gradient))
    assert np.array_equal(gradient[:, :2], np.zeros((21, 2)))


def test_metastability_gradient_uses_zero_subgradient_at_incoherence() -> None:
    # Two antiphase oscillators give R = 0 throughout; the gradient stays finite and zero.
    """Incoherence gets a zero metastability subgradient."""
    grid = np.arange(21) * 0.05
    trajectory = np.stack([grid, grid + np.pi], axis=1)
    gradient = metastability_index_gradient(trajectory)
    assert np.all(np.isfinite(gradient))
    assert np.array_equal(gradient, np.zeros((21, 2)))


# --------------------------------------------------------------------------- bundled diagnostics


def test_diagnostics_bundle_matches_individual_helpers() -> None:
    """The diagnostics bundle matches the individual helpers."""
    trajectory = _constructed_chimera(250, seed=11)
    bundle = chimera_diagnostics(trajectory, _TWO_GROUPS)
    assert isinstance(bundle, ChimeraDiagnostics)
    assert np.allclose(
        bundle.community_order_parameters, community_order_parameters(trajectory, _TWO_GROUPS)
    )
    assert bundle.chimera_index == pytest.approx(chimera_index(trajectory, _TWO_GROUPS))
    assert bundle.metastability_index == pytest.approx(metastability_index(trajectory))
    assert bundle.community_metastability == pytest.approx(
        community_metastability(trajectory, _TWO_GROUPS)
    )


def test_chimera_diagnostics_is_frozen() -> None:
    """The diagnostics bundle cannot be modified."""
    bundle = ChimeraDiagnostics(
        community_order_parameters=np.zeros((2, 1)),
        chimera_index=0.0,
        metastability_index=0.0,
        community_metastability=0.0,
    )
    with pytest.raises(AttributeError):
        bundle.chimera_index = 1.0  # type: ignore[misc]


# --------------------------------------------------------------------------- validation


def test_rejects_non_two_dimensional_phases() -> None:
    """Phases that are not two-dimensional are refused."""
    with pytest.raises(ValueError, match="two-dimensional"):
        metastability_index(np.zeros(5))


def test_rejects_empty_time_axis() -> None:
    """A trajectory without time samples is refused."""
    with pytest.raises(ValueError, match="at least one time sample"):
        metastability_index(np.zeros((0, 4)))


def test_rejects_empty_oscillator_axis() -> None:
    """A trajectory without oscillators is refused."""
    with pytest.raises(ValueError, match="at least one oscillator"):
        metastability_index(np.zeros((4, 0)))


def test_rejects_empty_community_list() -> None:
    """An empty community list is refused."""
    with pytest.raises(ValueError, match="at least one community"):
        chimera_index(np.zeros((4, 3)), [])


def test_rejects_out_of_range_community_index() -> None:
    """A community index outside the oscillators is refused."""
    with pytest.raises(ValueError, match="indices outside"):
        chimera_index(np.zeros((4, 3)), [np.array([0, 3])])


def test_rejects_overlapping_communities() -> None:
    """Overlapping communities are refused."""
    with pytest.raises(ValueError, match="shares oscillator"):
        chimera_index(np.zeros((4, 4)), [np.array([0, 1]), np.array([1, 2])])


def test_rejects_repeated_index_within_a_community() -> None:
    """A community that repeats an oscillator is refused."""
    with pytest.raises(ValueError, match="repeats an oscillator"):
        chimera_index(np.zeros((4, 4)), [np.array([0, 0, 1])])


def test_rejects_non_one_dimensional_community() -> None:
    """A community that is not a non-empty vector is refused."""
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        chimera_index(np.zeros((4, 4)), [np.zeros((2, 2), dtype=int)])
