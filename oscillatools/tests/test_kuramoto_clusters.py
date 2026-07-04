# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the phase-cluster and partial-synchronisation metrics
r"""Tests for :mod:`oscillatools.accel.kuramoto_clusters`.

The roadmap acceptance is checked directly: full synchrony yields a single cluster of every
oscillator, and a planted multi-cluster state recovers the right cluster count and sizes. The
dependence on the input matrix is exercised — the signed coherence splits an antiphase pair of
clusters while the phase-locking value merges them — together with the singleton convention, the
descending-size ordering, the per-cluster coherence and the validation branch.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscillatools.accel.kuramoto_clusters import (
    ClusterPartition,
    cluster_count,
    cluster_partition,
    phase_clusters,
)
from oscillatools.accel.kuramoto_coherence_matrix import (
    mean_coherence_matrix,
    phase_locking_matrix,
)

_TWO_PI = 2.0 * np.pi


def _drift(steps: int) -> np.ndarray:
    return np.arange(steps) * 0.02


def _planted(sizes_and_offsets: list[tuple[int, float]], steps: int = 400) -> np.ndarray:
    """A trajectory of rigid clusters, each a block of a given size at a constant phase offset."""
    grid = _drift(steps)
    columns = []
    for size, offset in sizes_and_offsets:
        columns.append(np.repeat((grid + offset)[:, None], size, axis=1))
    return np.concatenate(columns, axis=1)


# --------------------------------------------------------------------------- cluster recovery


def test_full_synchrony_is_a_single_cluster() -> None:
    trajectory = np.repeat((_drift(300) + 0.3)[:, None], 8, axis=1)
    partition = cluster_partition(mean_coherence_matrix(trajectory))
    assert partition.count == 1
    assert partition.sizes.tolist() == [8]
    assert partition.coherences[0] == pytest.approx(1.0, abs=1e-6)


def test_planted_three_cluster_state_recovers_count_and_sizes() -> None:
    trajectory = _planted([(4, 0.0), (3, _TWO_PI / 3), (2, 2 * _TWO_PI / 3)])
    partition = cluster_partition(mean_coherence_matrix(trajectory))
    assert partition.count == 3
    assert partition.sizes.tolist() == [4, 3, 2]  # descending
    assert np.allclose(partition.coherences, 1.0, atol=1e-6)
    # every oscillator carries a label and the labels are dense in [0, count)
    assert set(partition.labels.tolist()) == {0, 1, 2}


def test_cluster_count_matches_partition() -> None:
    trajectory = _planted([(2, 0.0), (2, np.pi / 2), (2, np.pi)])
    matrix = mean_coherence_matrix(trajectory)
    assert cluster_count(matrix) == cluster_partition(matrix).count == 3


# --------------------------------------------------------------------------- matrix-dependent notion


def test_signed_coherence_splits_antiphase_clusters() -> None:
    trajectory = _planted([(3, 0.0), (3, np.pi)])
    partition = cluster_partition(mean_coherence_matrix(trajectory))
    assert partition.count == 2
    assert partition.sizes.tolist() == [3, 3]


def test_phase_locking_value_merges_antiphase_clusters() -> None:
    trajectory = _planted([(3, 0.0), (3, np.pi)])
    partition = cluster_partition(phase_locking_matrix(trajectory))
    assert partition.count == 1
    assert partition.sizes.tolist() == [6]


# --------------------------------------------------------------------------- singletons & ordering


def test_isolated_oscillator_forms_a_singleton_with_unit_coherence() -> None:
    grid = _drift(400)
    trajectory = np.empty((400, 5))
    trajectory[:, :4] = grid[:, None]
    trajectory[:, 4] = 0.4 * grid + 1.0  # incommensurate drift decoheres from the cluster
    partition = cluster_partition(mean_coherence_matrix(trajectory))
    assert partition.count == 2
    assert partition.sizes.tolist() == [4, 1]
    assert partition.coherences[1] == 1.0  # singleton convention


def test_threshold_controls_cluster_granularity() -> None:
    # Two clusters 36 degrees apart: cos(36 deg) ~ 0.81 across-cluster. A loose threshold merges
    # them into one cluster, a tight threshold keeps them separate.
    trajectory = _planted([(3, 0.0), (3, _TWO_PI / 10)])
    matrix = mean_coherence_matrix(trajectory)
    assert cluster_count(matrix, threshold=0.7) == 1
    assert cluster_count(matrix, threshold=0.95) == 2


def test_labels_are_ordered_by_descending_size() -> None:
    trajectory = _planted([(1, 0.0), (5, np.pi / 2)])
    partition = cluster_partition(mean_coherence_matrix(trajectory))
    assert partition.sizes.tolist() == [5, 1]
    # the larger cluster owns label 0
    assert np.all(partition.labels[1:6] == 0)
    assert partition.labels[0] == 1


def test_partition_is_frozen() -> None:
    partition = ClusterPartition(
        labels=np.zeros(2, dtype=int),
        count=1,
        sizes=np.array([2]),
        coherences=np.array([1.0]),
    )
    with pytest.raises(AttributeError):
        partition.count = 2  # type: ignore[misc]


# --------------------------------------------------------------------------- validation


def test_phase_clusters_rejects_non_square_matrix() -> None:
    with pytest.raises(ValueError, match="non-empty square"):
        phase_clusters(np.zeros((3, 4)))


def test_cluster_partition_rejects_empty_matrix() -> None:
    with pytest.raises(ValueError, match="non-empty square"):
        cluster_partition(np.zeros((0, 0)))
