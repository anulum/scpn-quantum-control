# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Topology-kernel schema tests
"""Contract and fail-closed tests for immutable topology-kernel records."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scpn_quantum_control.topology_kernel_product import (
    TOPOLOGY_KERNEL_CLAIM_BOUNDARY,
    KernelEvaluation,
    TopologyKernelConfig,
    TopologyKernelDataset,
    TopologyKernelMatrix,
)

DIGEST = "a" * 64


def _matrix() -> TopologyKernelMatrix:
    return TopologyKernelMatrix(
        values=np.eye(2),
        row_ids=("a", "b"),
        column_ids=("a", "b"),
        topology_digest=DIGEST,
        content_digest=DIGEST,
    )


def _dataset() -> TopologyKernelDataset:
    return TopologyKernelDataset(
        train_features=np.arange(12, dtype=float).reshape(4, 3),
        train_labels=np.array([1, -1, 1, -1]),
        train_ids=("a", "b", "c", "d"),
        test_features=np.arange(6, dtype=float).reshape(2, 3),
        test_labels=np.array([1, -1]),
        test_ids=("e", "f"),
        teacher_prototypes=np.zeros((2, 3)),
        teacher_topology_digest=DIGEST,
        content_digest=DIGEST,
    )


def test_config_defaults_and_feature_dimension_are_explicit() -> None:
    """Expose bounded defaults and the canonical edge-feature dimension."""
    config = TopologyKernelConfig()
    assert config.feature_dim == 6
    assert config.n_qubits == 4
    assert config.ridge == pytest.approx(1.0e-3)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_qubits", True),
        ("n_qubits", 1),
        ("n_qubits", 9),
        ("evolution_time", 0.0),
        ("evolution_time", np.inf),
        ("trotter_reps", True),
        ("trotter_reps", 0),
        ("trotter_reps", 17),
        ("max_samples", True),
        ("max_samples", 1),
        ("max_samples", 257),
        ("ridge", 0.0),
        ("ridge", np.nan),
    ],
)
def test_config_rejects_invalid_resource_policy(field: str, value: object) -> None:
    """Reject each resource-policy value outside its bounded contract."""
    with pytest.raises(ValueError):
        TopologyKernelConfig(**{field: value})  # type: ignore[arg-type]


def test_kernel_matrix_is_read_only_and_normalises_identifiers() -> None:
    """Freeze matrix custody and normalize surrounding identifier whitespace."""
    matrix = TopologyKernelMatrix(
        values=np.eye(2),
        row_ids=(" a ", "b"),
        column_ids=(" c ", "d"),
        topology_digest=DIGEST,
        content_digest=DIGEST,
    )
    assert matrix.row_ids == ("a", "b")
    assert matrix.column_ids == ("c", "d")
    assert matrix.claim_boundary == TOPOLOGY_KERNEL_CLAIM_BOUNDARY
    with pytest.raises(ValueError):
        matrix.values[0, 0] = 2.0


@pytest.mark.parametrize(
    "changes",
    [
        {"values": np.array([1.0])},
        {"values": np.array([[np.nan]])},
        {"values": np.array([[-0.1]])},
        {"values": np.array([[1.1]])},
        {"row_ids": ()},
        {"row_ids": ("a", "a")},
        {"row_ids": ("", "b")},
        {"column_ids": ()},
        {"values": np.ones((1, 2))},
        {"topology_digest": "bad"},
        {"content_digest": "g" * 64},
        {"claim_boundary": " "},
    ],
)
def test_kernel_matrix_rejects_invalid_contract(changes: dict[str, object]) -> None:
    """Reject malformed matrix values, custody metadata, and claim bounds."""
    with pytest.raises(ValueError):
        replace(_matrix(), **changes)


def test_dataset_is_balanced_disjoint_and_read_only() -> None:
    """Preserve balanced split custody through immutable defensive copies."""
    dataset = _dataset()
    assert dataset.train_features.shape == (4, 3)
    assert not dataset.train_features.flags.writeable
    assert not dataset.test_labels.flags.writeable
    with pytest.raises(ValueError):
        dataset.teacher_prototypes[0, 0] = 1.0


@pytest.mark.parametrize(
    "changes",
    [
        {"train_features": np.ones(3)},
        {"test_features": np.ones((2, 4))},
        {"train_features": np.ones((1, 3))},
        {"teacher_prototypes": np.ones((3, 3))},
        {"test_features": np.array([[1.0, np.nan, 2.0], [1.0, 2.0, 3.0]])},
        {"train_labels": np.array([1, -1])},
        {"train_labels": np.array([1, 1, 1, -1])},
        {"train_ids": ("a", "b")},
        {"test_ids": ("a", "f")},
        {"teacher_topology_digest": "bad"},
        {"content_digest": "bad"},
        {"claim_boundary": ""},
    ],
)
def test_dataset_rejects_invalid_contract(changes: dict[str, object]) -> None:
    """Reject malformed split geometry, labels, identifiers, and custody."""
    with pytest.raises(ValueError):
        replace(_dataset(), **changes)


def test_kernel_evaluation_recomputes_accuracy_and_freezes_arrays() -> None:
    """Normalize evaluation identity and freeze prediction evidence."""
    evaluation = KernelEvaluation(
        name=" ring ",
        predictions=np.array([1, -1]),
        labels=np.array([1, 1]),
        correct=1,
        total=2,
        accuracy=0.5,
        kernel_digest=DIGEST,
    )
    assert evaluation.name == "ring"
    assert not evaluation.predictions.flags.writeable
    assert evaluation.accuracy == pytest.approx(0.5)


@pytest.mark.parametrize(
    "changes",
    [
        {"name": ""},
        {"predictions": np.ones((1, 2), dtype=int)},
        {"labels": np.array([1])},
        {"predictions": np.array([0, 1])},
        {"labels": np.array([0, 1])},
        {"correct": 2},
        {"total": 3},
        {"accuracy": 0.4},
        {"kernel_digest": "bad"},
    ],
)
def test_kernel_evaluation_rejects_invalid_contract(changes: dict[str, object]) -> None:
    """Reject inconsistent evaluation identity, vectors, counts, and digest."""
    valid = KernelEvaluation(
        name="ring",
        predictions=np.array([1, -1]),
        labels=np.array([1, 1]),
        correct=1,
        total=2,
        accuracy=0.5,
        kernel_digest=DIGEST,
    )
    with pytest.raises(ValueError):
        replace(valid, **changes)
