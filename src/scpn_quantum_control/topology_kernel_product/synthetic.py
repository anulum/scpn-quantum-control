# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Synthetic topology-kernel task
"""Deterministic graph controls and teacher-aligned synthetic data."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from .kernels import fidelity_kernel_matrix, topology_digest
from .schema import FloatArray, TopologyKernelConfig, TopologyKernelDataset


def _read_only_topology(value: NDArray[np.float64]) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def ring_topology(n_qubits: int) -> FloatArray:
    """Return an unweighted undirected cycle adjacency matrix.

    For two nodes the simple cycle collapses to their single undirected edge;
    no parallel-edge weight is introduced.
    """
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or n_qubits < 2:
        raise ValueError("n_qubits must be an integer of at least two")
    matrix = np.zeros((n_qubits, n_qubits), dtype=np.float64)
    for node in range(n_qubits):
        neighbour = (node + 1) % n_qubits
        matrix[node, neighbour] = 1.0
        matrix[neighbour, node] = 1.0
    return _read_only_topology(matrix)


def path_topology(n_qubits: int) -> FloatArray:
    """Return an unweighted undirected path adjacency matrix."""
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or n_qubits < 2:
        raise ValueError("n_qubits must be an integer of at least two")
    matrix = np.zeros((n_qubits, n_qubits), dtype=np.float64)
    for node in range(n_qubits - 1):
        matrix[node, node + 1] = 1.0
        matrix[node + 1, node] = 1.0
    return _read_only_topology(matrix)


def complete_topology(n_qubits: int) -> FloatArray:
    """Return an unweighted complete-graph adjacency matrix."""
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or n_qubits < 2:
        raise ValueError("n_qubits must be an integer of at least two")
    return _read_only_topology(np.ones((n_qubits, n_qubits)) - np.eye(n_qubits))


def zero_topology(n_qubits: int) -> FloatArray:
    """Return the all-zero topology used as a no-coupling control."""
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or n_qubits < 2:
        raise ValueError("n_qubits must be an integer of at least two")
    return _read_only_topology(np.zeros((n_qubits, n_qubits), dtype=np.float64))


def build_teacher_aligned_dataset(
    config: TopologyKernelConfig,
    *,
    seed: int = 880,
    candidate_count: int = 256,
    train_count: int = 32,
    test_count: int = 16,
) -> TopologyKernelDataset:
    """Build the frozen balanced teacher-aligned representability task.

    Two prototype vectors and a candidate pool are drawn uniformly from
    ``[-pi, pi]``. Candidates are ranked by the difference between their ring
    kernel similarities to the positive and negative prototype. Equal counts
    from the two tails are interleaved, then split without shuffling.

    Parameters
    ----------
    config:
        Kernel configuration. ``train_count`` and ``test_count`` must each fit
        its sample budget.
    seed:
        Non-negative NumPy generator seed. Committed evidence fixes this at 880.
    candidate_count:
        Candidate pool size in ``[train_count + test_count, 256]``.
    train_count, test_count:
        Positive even split sizes. Each split is exactly balanced.

    Returns
    -------
    TopologyKernelDataset
        Immutable features, labels, source candidate identifiers, prototypes,
        and SHA-256 custody.

    Notes
    -----
    Because labels and evaluation share the ring-kernel family, good accuracy
    is circular representability evidence rather than independent predictive
    validation.

    """
    if not isinstance(config, TopologyKernelConfig):
        raise ValueError("config must be a TopologyKernelConfig")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    counts = (candidate_count, train_count, test_count)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
        raise ValueError("dataset counts must be integers")
    if train_count < 2 or test_count < 2 or train_count % 2 or test_count % 2:
        raise ValueError("train_count and test_count must be positive even values")
    selected_count = train_count + test_count
    if candidate_count < selected_count or candidate_count > 256:
        raise ValueError("candidate_count must cover the selected data and be at most 256")
    if train_count > config.max_samples or test_count > config.max_samples:
        raise ValueError("train_count and test_count must fit the configured sample budget")

    generator = np.random.default_rng(seed)
    prototypes = generator.uniform(-np.pi, np.pi, size=(2, config.feature_dim))
    candidates = generator.uniform(-np.pi, np.pi, size=(candidate_count, config.feature_dim))
    candidate_config = replace(config, max_samples=max(config.max_samples, candidate_count))
    candidate_ids = tuple(f"candidate-{index:03d}" for index in range(candidate_count))
    prototype_ids = ("teacher-positive", "teacher-negative")
    topology = ring_topology(config.n_qubits)
    teacher_kernel = fidelity_kernel_matrix(
        candidates,
        prototypes,
        topology,
        candidate_config,
        row_ids=candidate_ids,
        column_ids=prototype_ids,
    )
    score = teacher_kernel.values[:, 0] - teacher_kernel.values[:, 1]
    per_class = selected_count // 2
    positive = np.flatnonzero(score > 0.0)
    negative = np.flatnonzero(score < 0.0)
    positive = positive[np.argsort(-score[positive], kind="stable")][:per_class]
    negative = negative[np.argsort(score[negative], kind="stable")][:per_class]
    if positive.size != per_class or negative.size != per_class:
        raise ValueError(
            "candidate pool does not contain enough examples from both teacher classes"
        )
    selected = np.column_stack((positive, negative)).ravel()
    labels = np.tile(np.asarray([1, -1], dtype=np.int64), per_class)
    selected_features = candidates[selected]
    selected_ids = tuple(candidate_ids[index] for index in selected)
    train_features = selected_features[:train_count]
    test_features = selected_features[train_count:]
    train_labels = labels[:train_count]
    test_labels = labels[train_count:]
    train_ids = selected_ids[:train_count]
    test_ids = selected_ids[train_count:]
    teacher_digest = topology_digest(topology, config)
    digest = hashlib.sha256()
    for array in (train_features, train_labels, test_features, test_labels, prototypes):
        dtype = "<i8" if np.issubdtype(array.dtype, np.integer) else "<f8"
        digest.update(np.ascontiguousarray(array, dtype=dtype).tobytes())
    digest.update("\x00".join(train_ids).encode())
    digest.update(b"\xff")
    digest.update("\x00".join(test_ids).encode())
    digest.update(teacher_digest.encode())
    return TopologyKernelDataset(
        train_features=train_features,
        train_labels=train_labels,
        train_ids=train_ids,
        test_features=test_features,
        test_labels=test_labels,
        test_ids=test_ids,
        teacher_prototypes=prototypes,
        teacher_topology_digest=teacher_digest,
        content_digest=digest.hexdigest(),
    )
