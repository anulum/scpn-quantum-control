# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Synthetic topology-task tests
"""Frozen dataset, graph-control, and validation tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scpn_quantum_control.topology_kernel_product import (
    TopologyKernelConfig,
    build_teacher_aligned_dataset,
    complete_topology,
    path_topology,
    ring_topology,
    synthetic,
    zero_topology,
)


def test_graph_control_constructors_have_expected_edges() -> None:
    """Frozen graph controls have the expected simple undirected edges."""
    assert ring_topology(2).tolist() == [[0.0, 1.0], [1.0, 0.0]]
    assert int(np.sum(ring_topology(4)) // 2) == 4
    assert int(np.sum(path_topology(4)) // 2) == 3
    assert int(np.sum(complete_topology(4)) // 2) == 6
    assert np.count_nonzero(zero_topology(4)) == 0
    for topology in (
        ring_topology(4),
        path_topology(4),
        complete_topology(4),
        zero_topology(4),
    ):
        assert not topology.flags.writeable
        assert np.allclose(topology, topology.T)
        assert np.allclose(np.diag(topology), 0.0)


@pytest.mark.parametrize(
    "constructor",
    [ring_topology, path_topology, complete_topology, zero_topology],
)
@pytest.mark.parametrize("value", [True, 1, 1.5])
def test_graph_control_constructors_reject_bad_sizes(constructor: object, value: object) -> None:
    """Graph constructors reject non-integer and undersized node counts."""
    with pytest.raises(ValueError):
        constructor(value)  # type: ignore[operator]


def test_frozen_teacher_dataset_has_expected_custody_and_balance() -> None:
    """The frozen teacher task retains exact custody and class balance."""
    dataset = build_teacher_aligned_dataset(TopologyKernelConfig())
    assert (
        dataset.content_digest
        == "f86d0a3800dc28cddbd12d7385c1b0d4045c9cb7b33224d495892ab9af66b67b"
    )
    assert dataset.train_features.shape == (32, 6)
    assert dataset.test_features.shape == (16, 6)
    assert dataset.train_labels.tolist() == [1, -1] * 16
    assert dataset.test_labels.tolist() == [1, -1] * 8
    assert set(dataset.train_ids).isdisjoint(dataset.test_ids)
    assert dataset.train_ids[:2] == ("candidate-170", "candidate-186")
    assert not dataset.train_features.flags.writeable


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": True},
        {"seed": -1},
        {"candidate_count": True},
        {"train_count": 3},
        {"test_count": 1},
        {"candidate_count": 46},
        {"candidate_count": 257},
    ],
)
def test_dataset_builder_rejects_invalid_seed_and_counts(kwargs: dict[str, object]) -> None:
    """Invalid seeds and dataset counts fail closed."""
    with pytest.raises(ValueError):
        build_teacher_aligned_dataset(TopologyKernelConfig(), **kwargs)  # type: ignore[arg-type]


def test_dataset_builder_rejects_wrong_config_and_split_over_budget() -> None:
    """Wrong configurations and over-budget splits are rejected."""
    with pytest.raises(ValueError):
        build_teacher_aligned_dataset(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_teacher_aligned_dataset(TopologyKernelConfig(max_samples=16))


def test_dataset_builder_rejects_pool_without_both_teacher_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A candidate pool without both teacher classes cannot emit evidence."""
    monkeypatch.setattr(
        synthetic,
        "fidelity_kernel_matrix",
        lambda *_args, **_kwargs: SimpleNamespace(values=np.ones((4, 2))),
    )
    with pytest.raises(ValueError, match="both teacher classes"):
        build_teacher_aligned_dataset(
            TopologyKernelConfig(n_qubits=2, max_samples=2),
            candidate_count=4,
            train_count=2,
            test_count=2,
        )
