# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Topology-kernel numerical tests
"""Exact feature-map, fidelity, topology, and permutation tests."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.applications import (
    canonical_edge_pairs,
    encode_topology_edge_features,
)
from scpn_quantum_control.topology_kernel_product import (
    TopologyKernelConfig,
    fidelity_kernel_matrix,
    permute_edge_features,
    permute_topology,
    rbf_kernel_matrix,
    ring_topology,
    topology_digest,
    validate_feature_matrix,
    validate_topology,
)
from scpn_quantum_control.topology_kernel_product.kernels import _float_bytes


def test_kernel_custody_bytes_normalise_subprecision_and_signed_zero() -> None:
    """Normalize subprecision values and signed zero in custody bytes."""
    left = np.asarray([-0.0, 0.12345678901231], dtype=np.float64)
    right = np.asarray([1.0e-14, 0.12345678901229], dtype=np.float64)
    assert _float_bytes(left) == _float_bytes(right)


def test_canonical_edges_and_edge_encoder_contract() -> None:
    """Construct canonical edge order and a normalized encoded state."""
    assert canonical_edge_pairs(4) == ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    topology = ring_topology(4)
    state = encode_topology_edge_features(np.zeros(6), topology, 4)
    assert state.dim == 16
    assert np.linalg.norm(state.data) == pytest.approx(1.0)


@pytest.mark.parametrize("value", [True, 0, 1.5])
def test_canonical_edges_reject_invalid_qubit_count(value: object) -> None:
    """Reject noninteger and nonpositive canonical-edge qubit counts."""
    with pytest.raises(ValueError):
        canonical_edge_pairs(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_qubits": True},
        {"max_qubits": 0},
        {"max_qubits": 21},
        {"max_qubits": 1},
        {"K": np.array([[0.0, 1.0], [0.0, 0.0]])},
        {"K": np.ones((2, 2))},
        {"x": np.ones(2)},
        {"t": 0.0},
        {"t": np.inf},
        {"reps": True},
        {"reps": 0},
        {"reps": 17},
    ],
)
def test_edge_encoder_fails_closed(kwargs: dict[str, object]) -> None:
    """Reject invalid topology, feature, evolution, and allocation inputs."""
    arguments: dict[str, object] = {
        "x": np.ones(1),
        "K": np.array([[0.0, 1.0], [1.0, 0.0]]),
        "n_qubits": 2,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        encode_topology_edge_features(**arguments)  # type: ignore[arg-type]


def test_topology_and_feature_validation_return_defensive_copies() -> None:
    """Return read-only defensive topology and feature copies."""
    config = TopologyKernelConfig(n_qubits=3, max_samples=3)
    source_topology = ring_topology(3).copy()
    checked_topology = validate_topology(source_topology, config)
    source_topology[0, 1] = 7.0
    assert checked_topology[0, 1] == pytest.approx(1.0)
    source_features = np.arange(9, dtype=float).reshape(3, 3)
    checked_features = validate_feature_matrix(source_features, config)
    source_features[0, 0] = 99.0
    assert checked_features[0, 0] == pytest.approx(0.0)
    assert not checked_topology.flags.writeable
    assert not checked_features.flags.writeable


@pytest.mark.parametrize(
    "topology",
    [
        np.eye(2),
        np.ones((3, 2)),
        np.array([[0.0, np.nan], [np.nan, 0.0]]),
        np.array([[0.0, 1.0], [0.0, 0.0]]),
        np.ones((2, 2)),
    ],
)
def test_validate_topology_rejects_invalid_matrices(
    topology: NDArray[np.float64],
) -> None:
    """Reject malformed, nonfinite, asymmetric, or diagonal topologies."""
    with pytest.raises(ValueError):
        validate_topology(topology, TopologyKernelConfig(n_qubits=2))


def test_validate_topology_rejects_wrong_config_type() -> None:
    """Reject topology validation with an invalid policy object."""
    with pytest.raises(ValueError):
        validate_topology(np.zeros((2, 2)), object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "features",
    [
        np.ones(3),
        np.ones((2, 2)),
        np.empty((0, 3)),
        np.ones((4, 3)),
        np.array([[1.0, np.nan, 2.0]]),
    ],
)
def test_validate_feature_matrix_rejects_shape_budget_and_finiteness(
    features: NDArray[np.float64],
) -> None:
    """Reject invalid feature shapes, budgets, and nonfinite values."""
    with pytest.raises(ValueError):
        validate_feature_matrix(features, TopologyKernelConfig(n_qubits=3, max_samples=3))


def test_fidelity_kernel_is_symmetric_psd_and_custody_bound() -> None:
    """Bind a symmetric PSD fidelity matrix to topology and axis custody."""
    config = TopologyKernelConfig(n_qubits=3, max_samples=4)
    features = np.array([[0.1, 0.2, 0.3], [-0.2, 0.7, 0.4], [1.0, -0.5, 0.2]])
    ids = ("a", "b", "c")
    topology = ring_topology(3)
    kernel = fidelity_kernel_matrix(
        features,
        features,
        topology,
        config,
        row_ids=ids,
        column_ids=ids,
    )
    assert np.allclose(kernel.values, kernel.values.T, atol=1.0e-14, rtol=0.0)
    assert np.allclose(np.diag(kernel.values), 1.0, atol=1.0e-14, rtol=0.0)
    assert np.linalg.eigvalsh(kernel.values).min() >= -1.0e-12
    assert kernel.topology_digest == topology_digest(topology.astype(np.float32), config)
    assert not kernel.values.flags.writeable


def test_fidelity_and_rbf_kernels_require_axis_ids() -> None:
    """Require one row and column identifier per kernel axis."""
    config = TopologyKernelConfig(n_qubits=2)
    features = np.ones((2, 1))
    with pytest.raises(ValueError):
        fidelity_kernel_matrix(
            features,
            features,
            ring_topology(2),
            config,
            row_ids=("a",),
            column_ids=("a", "b"),
        )
    with pytest.raises(ValueError):
        rbf_kernel_matrix(
            features,
            features,
            config,
            gamma=1.0,
            row_ids=("a", "b"),
            column_ids=("a",),
        )


def test_rbf_kernel_matches_closed_form_and_rejects_gamma() -> None:
    """Match the RBF closed form and reject invalid bandwidths."""
    config = TopologyKernelConfig(n_qubits=2)
    rows = np.array([[0.0], [1.0]])
    kernel = rbf_kernel_matrix(
        rows,
        rows,
        config,
        gamma=0.5,
        row_ids=("a", "b"),
        column_ids=("a", "b"),
    )
    assert np.allclose(
        kernel.values,
        np.array([[1.0, np.exp(-0.5)], [np.exp(-0.5), 1.0]]),
        rtol=0.0,
        atol=1.0e-15,
    )
    assert topology_digest(ring_topology(2), config) != kernel.topology_digest
    for gamma in (0.0, np.inf):
        with pytest.raises(ValueError):
            rbf_kernel_matrix(
                rows,
                rows,
                config,
                gamma=gamma,
                row_ids=("a", "b"),
                column_ids=("a", "b"),
            )


def test_simultaneous_node_and_edge_permutation_preserves_kernel() -> None:
    """Preserve fidelity under simultaneous node and edge relabeling."""
    config = TopologyKernelConfig(n_qubits=4)
    features = np.array([[0.2, -0.3, 0.7, 0.1, -0.4, 0.9], [1.0, 0.2, -0.2, 0.3, 0.6, -0.8]])
    ids = ("a", "b")
    topology = ring_topology(4)
    permutation = (1, 2, 3, 0)
    original = fidelity_kernel_matrix(
        features, features, topology, config, row_ids=ids, column_ids=ids
    )
    relabelled_features = permute_edge_features(features, permutation, config)
    relabelled_topology = permute_topology(topology, permutation, config)
    relabelled = fidelity_kernel_matrix(
        relabelled_features,
        relabelled_features,
        relabelled_topology,
        config,
        row_ids=ids,
        column_ids=ids,
    )
    assert relabelled.values == pytest.approx(original.values, abs=1.0e-14)
    assert not relabelled_features.flags.writeable
    assert not relabelled_topology.flags.writeable


@pytest.mark.parametrize("function", [permute_edge_features, permute_topology])
def test_permutation_helpers_reject_non_permutations(function: object) -> None:
    """Reject repeated-node mappings in both permutation helpers."""
    config = TopologyKernelConfig(n_qubits=3)
    value = np.ones((2, 3)) if function is permute_edge_features else ring_topology(3)
    with pytest.raises(ValueError):
        function(value, (0, 0, 2), config)  # type: ignore[operator]
