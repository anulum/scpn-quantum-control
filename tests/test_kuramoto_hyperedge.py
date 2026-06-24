# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the structural hypergraph Kuramoto force and Jacobian
r"""Tests for :mod:`scpn_quantum_control.accel.kuramoto_hyperedge`.

The structural hypergraph force is checked against its hand-evaluated sum on a triangle, against
the symmetric networked Kuramoto force for an undirected arity-2 edge list (the lower-order
reduction), and for a mix of triangle and tetrahedron edges; the Jacobian is checked against a
central finite difference and for the zero row sums of the global-phase Goldstone mode. The edge
and weight validation is exercised across every malformed-input branch.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.accel.kuramoto_hyperedge import (
    hyperedge_force,
    hyperedge_jacobian,
)
from scpn_quantum_control.accel.networked_kuramoto import networked_kuramoto_force


def _central_difference_jacobian(theta, hyperedges, coupling, step=1e-6):
    count = theta.size
    numeric = np.empty((count, count))
    for column in range(count):
        forward = theta.copy()
        forward[column] += step
        backward = theta.copy()
        backward[column] -= step
        numeric[:, column] = (
            hyperedge_force(forward, hyperedges, coupling)
            - hyperedge_force(backward, hyperedges, coupling)
        ) / (2.0 * step)
    return numeric


# --------------------------------------------------------------------------- force


def test_force_matches_hand_sum_on_a_triangle() -> None:
    theta = np.array([0.2, 1.1, 2.3])
    weight = 0.8
    force = hyperedge_force(theta, [(0, 1, 2)], weight)
    expected = np.array(
        [
            weight * np.sin(theta[1] + theta[2] - 2 * theta[0]),
            weight * np.sin(theta[0] + theta[2] - 2 * theta[1]),
            weight * np.sin(theta[0] + theta[1] - 2 * theta[2]),
        ]
    )
    assert np.allclose(force, expected)


def test_pairwise_edges_reduce_to_symmetric_networked_force() -> None:
    rng = np.random.default_rng(0)
    count = 6
    theta = rng.uniform(0.0, 2.0 * np.pi, count)
    upper = np.triu(rng.normal(0.0, 0.5, (count, count)), 1)
    edges = [(i, j) for i in range(count) for j in range(i + 1, count)]
    weights = [upper[i, j] for (i, j) in edges]
    symmetric = upper + upper.T
    assert np.allclose(
        hyperedge_force(theta, edges, weights), networked_kuramoto_force(theta, symmetric)
    )


def test_scalar_weight_is_broadcast_over_all_edges() -> None:
    theta = np.array([0.0, 0.6, 1.4, 2.0])
    edges = [(0, 1, 2), (1, 2, 3)]
    scalar = hyperedge_force(theta, edges, 0.5)
    per_edge = hyperedge_force(theta, edges, [0.5, 0.5])
    assert np.allclose(scalar, per_edge)


def test_force_sums_over_edges_sharing_a_node() -> None:
    theta = np.array([0.1, 0.7, 1.5, 2.2])
    triangle = hyperedge_force(theta, [(0, 1, 2)], 1.0)
    tetra = hyperedge_force(theta, [(0, 1, 2, 3)], 1.0)
    both = hyperedge_force(theta, [(0, 1, 2), (0, 1, 2, 3)], 1.0)
    assert np.allclose(both, triangle + tetra)


def test_force_rejects_empty_theta() -> None:
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        hyperedge_force(np.empty(0), [(0, 1)], 1.0)


def test_force_rejects_non_vector_theta() -> None:
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        hyperedge_force(np.zeros((2, 2)), [(0, 1)], 1.0)


def test_force_rejects_too_small_hyperedge() -> None:
    with pytest.raises(ValueError, match="at least two node indices"):
        hyperedge_force(np.zeros(3), [(0,)], 1.0)


def test_force_rejects_two_dimensional_hyperedge() -> None:
    with pytest.raises(ValueError, match="at least two node indices"):
        hyperedge_force(np.zeros(3), [[(0, 1), (1, 2)]], 1.0)


def test_force_rejects_repeated_node_in_hyperedge() -> None:
    with pytest.raises(ValueError, match="repeated node index"):
        hyperedge_force(np.zeros(4), [(0, 1, 1)], 1.0)


def test_force_rejects_out_of_range_node() -> None:
    with pytest.raises(ValueError, match="outside"):
        hyperedge_force(np.zeros(3), [(0, 3)], 1.0)


def test_force_rejects_mismatched_weight_length() -> None:
    with pytest.raises(ValueError, match="scalar or a length-2 sequence"):
        hyperedge_force(np.zeros(4), [(0, 1), (1, 2)], [0.5])


# --------------------------------------------------------------------------- Jacobian


def test_jacobian_matches_central_difference_on_mixed_orders() -> None:
    rng = np.random.default_rng(3)
    count = 6
    theta = rng.uniform(0.0, 2.0 * np.pi, count)
    edges = [(0, 1, 2), (1, 3, 4, 5), (0, 2, 5), (2, 4)]
    weights = [0.8, -0.5, 1.2, 0.3]
    analytic = hyperedge_jacobian(theta, edges, weights)
    numeric = _central_difference_jacobian(theta, edges, weights)
    assert np.allclose(analytic, numeric, atol=1e-7)


def test_jacobian_rows_sum_to_zero() -> None:
    rng = np.random.default_rng(4)
    theta = rng.uniform(0.0, 2.0 * np.pi, 6)
    edges = [(0, 1, 2), (1, 3, 4, 5), (0, 5)]
    jacobian = hyperedge_jacobian(theta, edges, 0.9)
    assert np.allclose(jacobian.sum(axis=1), 0.0, atol=1e-12)


def test_jacobian_diagonal_carries_minus_order_weight() -> None:
    # A single triangle {0,1,2}: J_00 = -(m-1) w cos(arg) with m = 3, arg = θ1+θ2-2θ0.
    theta = np.array([0.3, 1.0, 2.1])
    weight = 1.4
    jacobian = hyperedge_jacobian(theta, [(0, 1, 2)], weight)
    arg = theta[1] + theta[2] - 2 * theta[0]
    assert jacobian[0, 0] == pytest.approx(-2.0 * weight * np.cos(arg))
    assert jacobian[0, 1] == pytest.approx(weight * np.cos(arg))


def test_jacobian_rejects_empty_theta() -> None:
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        hyperedge_jacobian(np.empty(0), [(0, 1)], 1.0)


def test_jacobian_rejects_non_vector_theta() -> None:
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        hyperedge_jacobian(np.zeros((3, 3)), [(0, 1)], 1.0)


def test_jacobian_rejects_bad_hyperedge() -> None:
    with pytest.raises(ValueError, match="outside"):
        hyperedge_jacobian(np.zeros(3), [(0, 9)], 1.0)


def test_jacobian_rejects_mismatched_weight_length() -> None:
    with pytest.raises(ValueError, match="scalar or a length-1 sequence"):
        hyperedge_jacobian(np.zeros(4), [(0, 1)], [0.5, 0.6])
