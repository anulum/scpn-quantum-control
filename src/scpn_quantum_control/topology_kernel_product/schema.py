# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Topology-aware quantum-kernel contracts
"""Immutable contracts for the bounded topology-kernel quantum-kernel product."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

TOPOLOGY_KERNEL_CLAIM_BOUNDARY = (
    "finite exact-statevector synthetic classification only; teacher labels are generated "
    "by the same ring-topology kernel and establish representability, not independent "
    "generalisation, quantum advantage, domain fitness, provider, QPU, or hardware results"
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


def _read_only_float(value: NDArray[np.float64]) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _read_only_int(value: NDArray[np.int64]) -> IntArray:
    result = np.array(value, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


def _require_digest(value: str, name: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_ids(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{name} must be non-empty and unique")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} entries must be non-empty strings")
    return tuple(value.strip() for value in values)


@dataclass(frozen=True, slots=True)
class TopologyKernelConfig:
    """Resource and numerical policy for topology-aware kernel evaluation.

    Parameters
    ----------
    n_qubits:
        Number of graph nodes/qubits. Dense simulation is restricted to
        ``[2, 8]``.
    evolution_time:
        Positive finite XY evolution time.
    trotter_reps:
        Lie–Trotter repetition count in ``[1, 16]``.
    max_samples:
        Maximum rows or columns accepted by a single kernel call.
    ridge:
        Positive diagonal regularisation for kernel ridge classification.

    """

    n_qubits: int = 4
    evolution_time: float = 0.8
    trotter_reps: int = 2
    max_samples: int = 64
    ridge: float = 1.0e-3

    def __post_init__(self) -> None:
        """Validate finite resource and numerical limits."""
        if (
            isinstance(self.n_qubits, bool)
            or not isinstance(self.n_qubits, int)
            or not 2 <= self.n_qubits <= 8
        ):
            raise ValueError("n_qubits must be an integer in [2, 8]")
        if not np.isfinite(self.evolution_time) or self.evolution_time <= 0.0:
            raise ValueError("evolution_time must be finite and positive")
        if (
            isinstance(self.trotter_reps, bool)
            or not isinstance(self.trotter_reps, int)
            or not 1 <= self.trotter_reps <= 16
        ):
            raise ValueError("trotter_reps must be an integer in [1, 16]")
        if (
            isinstance(self.max_samples, bool)
            or not isinstance(self.max_samples, int)
            or not 2 <= self.max_samples <= 256
        ):
            raise ValueError("max_samples must be an integer in [2, 256]")
        if not np.isfinite(self.ridge) or self.ridge <= 0.0:
            raise ValueError("ridge must be finite and positive")

    @property
    def feature_dim(self) -> int:
        """Return the number of canonical undirected graph edges."""
        return self.n_qubits * (self.n_qubits - 1) // 2


@dataclass(frozen=True, slots=True)
class TopologyKernelMatrix:
    """A custody-bound fidelity-kernel matrix.

    Parameters
    ----------
    values:
        Finite two-dimensional fidelity values in ``[0, 1]`` up to numerical
        tolerance. A defensive, read-only copy is stored.
    row_ids, column_ids:
        Unique sample identifiers matching the matrix axes.
    topology_digest:
        SHA-256 of the exact validated coupling matrix.
    content_digest:
        SHA-256 binding values, identifiers, and topology digest.
    claim_boundary:
        Explicit interpretation limit propagated with the result.

    """

    values: FloatArray
    row_ids: tuple[str, ...]
    column_ids: tuple[str, ...]
    topology_digest: str
    content_digest: str
    claim_boundary: str = TOPOLOGY_KERNEL_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate custody metadata and freeze a defensive matrix copy."""
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 2 or min(values.shape) < 1 or not np.all(np.isfinite(values)):
            raise ValueError("values must be a finite non-empty 2-D matrix")
        if np.any(values < -1.0e-10) or np.any(values > 1.0 + 1.0e-10):
            raise ValueError("fidelity values must lie in [0, 1] within tolerance")
        row_ids = _require_ids(self.row_ids, "row_ids")
        column_ids = _require_ids(self.column_ids, "column_ids")
        if values.shape != (len(row_ids), len(column_ids)):
            raise ValueError("values shape must match row_ids and column_ids")
        object.__setattr__(self, "values", _read_only_float(values))
        object.__setattr__(self, "row_ids", row_ids)
        object.__setattr__(self, "column_ids", column_ids)
        object.__setattr__(
            self, "topology_digest", _require_digest(self.topology_digest, "topology_digest")
        )
        object.__setattr__(
            self, "content_digest", _require_digest(self.content_digest, "content_digest")
        )
        if not isinstance(self.claim_boundary, str) or not self.claim_boundary.strip():
            raise ValueError("claim_boundary must be a non-empty string")
        object.__setattr__(self, "claim_boundary", self.claim_boundary.strip())


@dataclass(frozen=True, slots=True)
class TopologyKernelDataset:
    """Frozen balanced train/test split for teacher-aligned evidence.

    The labels are derived from two frozen prototype similarities under the
    primary ring kernel. Consequently this dataset tests whether the product
    can reproduce its own declared inductive bias; it is not an independently
    labelled scientific or application benchmark.

    Parameters
    ----------
    train_features, test_features:
        Finite edge-feature matrices with equal feature width.
    train_labels, test_labels:
        Balanced binary labels encoded as ``-1`` and ``+1``.
    train_ids, test_ids:
        Unique, mutually disjoint sample identifiers.
    teacher_prototypes:
        Two edge-feature rows defining positive and negative teacher anchors.
    teacher_topology_digest:
        SHA-256 of the ring topology used to generate labels.
    content_digest:
        SHA-256 binding all dataset contents and identifiers.

    """

    train_features: FloatArray
    train_labels: IntArray
    train_ids: tuple[str, ...]
    test_features: FloatArray
    test_labels: IntArray
    test_ids: tuple[str, ...]
    teacher_prototypes: FloatArray
    teacher_topology_digest: str
    content_digest: str
    claim_boundary: str = TOPOLOGY_KERNEL_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate the balanced split and freeze defensive array copies."""
        train = np.asarray(self.train_features, dtype=np.float64)
        test = np.asarray(self.test_features, dtype=np.float64)
        prototypes = np.asarray(self.teacher_prototypes, dtype=np.float64)
        if train.ndim != 2 or test.ndim != 2 or train.shape[1:] != test.shape[1:]:
            raise ValueError("train_features and test_features must be compatible 2-D matrices")
        if train.shape[0] < 2 or test.shape[0] < 2 or train.shape[1] < 1:
            raise ValueError("dataset splits must each contain at least two non-empty samples")
        if prototypes.shape != (2, train.shape[1]):
            raise ValueError("teacher_prototypes must have shape (2, feature_dim)")
        if not all(np.all(np.isfinite(array)) for array in (train, test, prototypes)):
            raise ValueError("dataset features and prototypes must be finite")
        train_labels = np.asarray(self.train_labels, dtype=np.int64)
        test_labels = np.asarray(self.test_labels, dtype=np.int64)
        if train_labels.shape != (train.shape[0],) or test_labels.shape != (test.shape[0],):
            raise ValueError("labels must match their feature split")
        for labels in (train_labels, test_labels):
            if set(labels.tolist()) != {-1, 1} or int(np.sum(labels == -1)) != int(
                np.sum(labels == 1)
            ):
                raise ValueError("each label split must be balanced and binary")
        train_ids = _require_ids(self.train_ids, "train_ids")
        test_ids = _require_ids(self.test_ids, "test_ids")
        if len(train_ids) != train.shape[0] or len(test_ids) != test.shape[0]:
            raise ValueError("sample identifiers must match their feature split")
        if set(train_ids) & set(test_ids):
            raise ValueError("train_ids and test_ids must be disjoint")
        object.__setattr__(self, "train_features", _read_only_float(train))
        object.__setattr__(self, "test_features", _read_only_float(test))
        object.__setattr__(self, "teacher_prototypes", _read_only_float(prototypes))
        object.__setattr__(self, "train_labels", _read_only_int(train_labels))
        object.__setattr__(self, "test_labels", _read_only_int(test_labels))
        object.__setattr__(self, "train_ids", train_ids)
        object.__setattr__(self, "test_ids", test_ids)
        object.__setattr__(
            self,
            "teacher_topology_digest",
            _require_digest(self.teacher_topology_digest, "teacher_topology_digest"),
        )
        object.__setattr__(
            self, "content_digest", _require_digest(self.content_digest, "content_digest")
        )
        if not isinstance(self.claim_boundary, str) or not self.claim_boundary.strip():
            raise ValueError("claim_boundary must be a non-empty string")
        object.__setattr__(self, "claim_boundary", self.claim_boundary.strip())


@dataclass(frozen=True, slots=True)
class KernelEvaluation:
    """Predictions and exact accuracy for one named kernel control.

    Parameters
    ----------
    name:
        Stable, non-empty control name.
    predictions, labels:
        Read-only binary arrays of equal non-zero length.
    correct, total, accuracy:
        Internally consistent classification counts and ratio.
    kernel_digest:
        SHA-256 of the evaluated cross-kernel matrix.

    """

    name: str
    predictions: IntArray
    labels: IntArray
    correct: int
    total: int
    accuracy: float
    kernel_digest: str

    def __post_init__(self) -> None:
        """Validate predictions, counts, accuracy, and digest custody."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        predictions = np.asarray(self.predictions, dtype=np.int64)
        labels = np.asarray(self.labels, dtype=np.int64)
        if predictions.ndim != 1 or predictions.size < 1 or predictions.shape != labels.shape:
            raise ValueError("predictions and labels must be equal non-empty vectors")
        if not set(predictions.tolist()) <= {-1, 1} or not set(labels.tolist()) <= {-1, 1}:
            raise ValueError("predictions and labels must be binary")
        observed = int(np.sum(predictions == labels))
        if self.total != predictions.size or self.correct != observed:
            raise ValueError("correct and total must match predictions")
        if not np.isclose(self.accuracy, observed / predictions.size, rtol=0.0, atol=1.0e-15):
            raise ValueError("accuracy must equal correct / total")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "predictions", _read_only_int(predictions))
        object.__setattr__(self, "labels", _read_only_int(labels))
        object.__setattr__(self, "accuracy", float(self.accuracy))
        object.__setattr__(
            self, "kernel_digest", _require_digest(self.kernel_digest, "kernel_digest")
        )
