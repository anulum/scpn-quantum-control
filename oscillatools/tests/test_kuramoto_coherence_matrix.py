# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the pairwise phase-coherence matrices
r"""Tests for :mod:`oscillatools.accel.kuramoto_coherence_matrix`.

The instantaneous coherence is checked entry-by-entry against ``cos(theta_j - theta_k)`` and
for its symmetric unit-diagonal form; the time-averaged coherence is checked to keep the signed
block structure of a clustered state (the roadmap acceptance) while the phase-locking value is
checked to read one for a rigidly locked pair and near zero for an independent one. The spectral
helpers are checked to reproduce the matrix, sort eigenvalues, and split two antiphase clusters
by the sign of the leading eigenvector. The validation branches are covered.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscillatools.accel.kuramoto_coherence_matrix import (
    coherence_matrix,
    coherence_spectrum,
    leading_coherence_eigenvector,
    mean_coherence_matrix,
    phase_locking_matrix,
)

_TWO_PI = 2.0 * np.pi


def _clustered_trajectory(steps: int) -> np.ndarray:
    """Two rigidly antiphase clusters (3 + 3) drifting together at unit rate."""
    grid = np.arange(steps) * 0.02
    trajectory = np.empty((steps, 6))
    trajectory[:, :3] = grid[:, None]
    trajectory[:, 3:] = grid[:, None] + np.pi
    return trajectory


# --------------------------------------------------------------------------- instantaneous coherence


def test_coherence_matrix_matches_cosine_difference() -> None:
    theta = np.random.default_rng(0).uniform(0.0, _TWO_PI, 7)
    matrix = coherence_matrix(theta)
    assert matrix.shape == (7, 7)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1.0)
    expected = np.cos(theta[:, None] - theta[None, :])
    assert np.allclose(matrix, expected)


def test_coherence_matrix_blocks_on_antiphase_clusters() -> None:
    theta = np.array([0.0, 0.0, 0.0, np.pi, np.pi, np.pi])
    matrix = coherence_matrix(theta)
    assert np.allclose(matrix[:3, :3], 1.0)
    assert np.allclose(matrix[3:, 3:], 1.0)
    assert np.allclose(matrix[:3, 3:], -1.0)


# --------------------------------------------------------------------------- time-averaged coherence


def test_mean_coherence_is_unity_for_full_synchrony() -> None:
    grid = np.arange(300) * 0.02
    trajectory = np.repeat((grid + 0.3)[:, None], 5, axis=1)
    assert np.allclose(mean_coherence_matrix(trajectory), 1.0)


def test_mean_coherence_keeps_signed_block_structure() -> None:
    matrix = mean_coherence_matrix(_clustered_trajectory(400))
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1.0)
    assert np.allclose(matrix[:3, :3], 1.0)
    assert np.allclose(matrix[3:, 3:], 1.0)
    assert np.allclose(matrix[:3, 3:], -1.0)


# --------------------------------------------------------------------------- phase-locking value


def test_phase_locking_is_unity_for_a_rigidly_locked_ensemble() -> None:
    # Antiphase but rigidly locked: every relative phase is constant, so rho = 1 everywhere.
    assert np.allclose(phase_locking_matrix(_clustered_trajectory(400)), 1.0)


def test_phase_locking_in_unit_interval_with_unit_diagonal() -> None:
    rng = np.random.default_rng(1)
    grid = np.arange(500) * 0.02
    trajectory = np.stack(
        [
            1.3 * grid + rng.uniform(0.0, _TWO_PI),
            rng.uniform(0.0, _TWO_PI, 500),  # independent random member
            0.4 * grid + rng.uniform(0.0, _TWO_PI),
        ],
        axis=1,
    )
    matrix = phase_locking_matrix(trajectory)
    assert np.all(matrix >= -1e-12) and np.all(matrix <= 1.0 + 1e-12)
    assert np.allclose(np.diag(matrix), 1.0)
    assert np.allclose(matrix, matrix.T)
    assert matrix[0, 1] < 0.2  # the random member decorrelates from the drift


# --------------------------------------------------------------------------- spectral structure


def test_spectrum_reproduces_the_matrix_and_sorts_eigenvalues() -> None:
    matrix = mean_coherence_matrix(_clustered_trajectory(400))
    values, vectors = coherence_spectrum(matrix)
    assert np.all(np.diff(values) <= 1e-9)  # descending
    reconstructed = (vectors * values) @ vectors.T
    assert np.allclose(reconstructed, matrix, atol=1e-9)


def test_leading_eigenvector_splits_two_clusters_by_sign() -> None:
    matrix = mean_coherence_matrix(_clustered_trajectory(400))
    leading = leading_coherence_eigenvector(matrix)
    assert leading.shape == (6,)
    assert np.isclose(np.linalg.norm(leading), 1.0)
    signs = np.sign(leading)
    assert np.all(signs[:3] == signs[0])
    assert np.all(signs[3:] == signs[3])
    assert signs[0] == -signs[3]


def test_full_synchrony_yields_a_single_dominant_mode() -> None:
    grid = np.arange(300) * 0.02
    trajectory = np.repeat((grid + 0.2)[:, None], 5, axis=1)
    values, _ = coherence_spectrum(mean_coherence_matrix(trajectory))
    assert values[0] == pytest.approx(5.0)  # rank-one all-ones matrix, eigenvalue = N
    assert np.allclose(values[1:], 0.0, atol=1e-9)


# --------------------------------------------------------------------------- validation


def test_coherence_matrix_rejects_non_one_dimensional() -> None:
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        coherence_matrix(np.zeros((3, 3)))


def test_coherence_matrix_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        coherence_matrix(np.empty(0))


def test_mean_coherence_rejects_non_two_dimensional() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        mean_coherence_matrix(np.zeros(5))


def test_phase_locking_rejects_empty_axes() -> None:
    with pytest.raises(ValueError, match="at least one time sample"):
        phase_locking_matrix(np.zeros((0, 3)))


def test_spectrum_rejects_non_square_matrix() -> None:
    with pytest.raises(ValueError, match="non-empty square"):
        coherence_spectrum(np.zeros((3, 4)))


def test_leading_eigenvector_rejects_empty_matrix() -> None:
    with pytest.raises(ValueError, match="non-empty square"):
        leading_coherence_eigenvector(np.zeros((0, 0)))
