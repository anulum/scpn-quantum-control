# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the information-theoretic phase diagnostics
r"""Tests for :mod:`oscillatools.accel.kuramoto_phase_information`.

The roadmap acceptance is checked at both extremes: the phase entropy is maximal
(:math:`\ln \text{bins}`) for a uniform ensemble and zero for full synchrony, and grows
monotonically with the phase spread. The mutual information is checked to equal the marginal
entropy for a phase-locked pair (an exact whole-bin offset gives a deterministic bin map) and
to fall near zero for statistically independent random phases, with the matrix symmetric and
its diagonal equal to the per-oscillator entropy. The validation branches are covered.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscillatools.accel.kuramoto_phase_information import (
    mutual_information_matrix,
    normalised_phase_entropy,
    pairwise_mutual_information,
    phase_entropy,
    phase_entropy_series,
)

_TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------- phase entropy


def test_uniform_phases_reach_the_maximum_entropy() -> None:
    angles = np.random.default_rng(0).uniform(0.0, _TWO_PI, 200_000)
    assert phase_entropy(angles, bins=36) == pytest.approx(np.log(36), abs=0.02)


def test_identical_phases_have_zero_entropy() -> None:
    assert phase_entropy(np.full(5000, 1.234), bins=36) == 0.0


def test_entropy_grows_monotonically_with_spread() -> None:
    rng = np.random.default_rng(1)
    previous = -1.0
    for sigma in (0.05, 0.2, 0.5, 1.0, 3.0):
        entropy = phase_entropy(rng.normal(0.0, sigma, 50_000), bins=36)
        assert entropy >= previous - 1e-9
        previous = entropy


def test_normalised_entropy_spans_zero_to_one() -> None:
    rng = np.random.default_rng(2)
    assert normalised_phase_entropy(rng.uniform(0.0, _TWO_PI, 200_000), bins=24) == pytest.approx(
        1.0, abs=0.02
    )
    assert normalised_phase_entropy(np.full(1000, 0.7), bins=24) == 0.0


def test_entropy_series_shape_and_ordering() -> None:
    # A trajectory that starts spread and ends synchronised: entropy must fall.
    rng = np.random.default_rng(3)
    spread = rng.uniform(0.0, _TWO_PI, (1, 400))
    converged = np.full((1, 400), 0.5)
    trajectory = np.concatenate([spread, converged], axis=0)
    series = phase_entropy_series(trajectory, bins=36)
    assert series.shape == (2,)
    assert series[0] > series[1]
    assert series[1] == 0.0


# --------------------------------------------------------------------------- mutual information


def _phase_pair(steps: int, *, bins: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    grid = np.arange(steps) * 0.01
    locked_a = 1.3 * grid + rng.uniform(0.0, _TWO_PI)
    locked_b = locked_a + (_TWO_PI / bins) * 5  # whole-bin offset -> deterministic bin map
    return locked_a, locked_b


def test_phase_locked_pair_has_mutual_information_equal_to_marginal_entropy() -> None:
    locked_a, locked_b = _phase_pair(20_000, bins=36, seed=4)
    marginal = phase_entropy(locked_a, bins=36)
    assert pairwise_mutual_information(locked_a, locked_b, bins=36) == pytest.approx(
        marginal, rel=1e-9
    )


def test_independent_phases_have_near_zero_mutual_information() -> None:
    rng = np.random.default_rng(5)
    first = rng.uniform(0.0, _TWO_PI, 20_000)
    second = rng.uniform(0.0, _TWO_PI, 20_000)
    assert pairwise_mutual_information(first, second, bins=36) < 0.1


def test_mutual_information_is_symmetric() -> None:
    locked_a, locked_b = _phase_pair(15_000, bins=24, seed=6)
    forward = pairwise_mutual_information(locked_a, locked_b, bins=24)
    backward = pairwise_mutual_information(locked_b, locked_a, bins=24)
    assert forward == pytest.approx(backward, rel=1e-12)


# --------------------------------------------------------------------------- mutual-information matrix


def test_mutual_information_matrix_structure() -> None:
    rng = np.random.default_rng(7)
    steps, bins = 20_000, 24
    grid = np.arange(steps) * 0.01
    reference = 1.0 * grid + rng.uniform(0.0, _TWO_PI)
    phases = np.stack(
        [
            reference,
            reference + (_TWO_PI / bins) * 3,  # locked to oscillator 0
            rng.uniform(0.0, _TWO_PI, steps),  # independent
            0.4 * grid + rng.uniform(0.0, _TWO_PI),
        ],
        axis=1,
    )
    matrix = mutual_information_matrix(phases, bins=bins)
    assert matrix.shape == (4, 4)
    assert np.allclose(matrix, matrix.T, atol=1e-12)
    for index in range(4):
        assert matrix[index, index] == pytest.approx(phase_entropy(phases[:, index], bins=bins))
    # The locked pair shares much more information than the locked-vs-independent pair.
    assert matrix[0, 1] > 5.0 * matrix[0, 2]


# --------------------------------------------------------------------------- validation


def test_phase_entropy_rejects_too_few_bins() -> None:
    with pytest.raises(ValueError, match="bins must be at least 2"):
        phase_entropy(np.zeros(10), bins=1)


def test_phase_entropy_rejects_non_one_dimensional_angles() -> None:
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        phase_entropy(np.zeros((3, 3)))


def test_phase_entropy_rejects_empty_angles() -> None:
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        phase_entropy(np.empty(0))


def test_mutual_information_rejects_unequal_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        pairwise_mutual_information(np.zeros(5), np.zeros(6))


def test_entropy_series_rejects_non_two_dimensional() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        phase_entropy_series(np.zeros(5))


def test_entropy_series_rejects_empty_axes() -> None:
    with pytest.raises(ValueError, match="at least one time sample"):
        phase_entropy_series(np.zeros((0, 4)))


def test_matrix_rejects_non_two_dimensional() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        mutual_information_matrix(np.zeros(5))


def test_matrix_rejects_too_few_bins() -> None:
    with pytest.raises(ValueError, match="bins must be at least 2"):
        mutual_information_matrix(np.zeros((10, 3)), bins=1)


def test_matrix_rejects_empty_axes() -> None:
    with pytest.raises(ValueError, match="at least one time sample"):
        mutual_information_matrix(np.zeros((0, 4)))
