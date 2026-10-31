# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — sparse Kuramoto storage, aggregation and placement qualification
"""Sparse Kuramoto parity, duplicate policy, edge-linear storage and placement.

The dense oracle evaluates ``F_i = sum_j K_ij sin(theta_j - theta_i)`` with
explicit loops over a dense matrix, independently of the edge kernels. Parity
uses the frozen float64 budget ``1e-10 + 1e-8*|reference|``. Storage is
measured from the stored arrays and from tracemalloc peaks; no performance
claim is made here.
"""

from __future__ import annotations

import tracemalloc
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import sparse

import oscillatools as kuramoto
from oscillatools.accel import sparse_kuramoto as sk

FloatArray = NDArray[np.float64]


def _within_budget(actual: object, reference: object) -> bool:
    """Apply the frozen float64 test budget elementwise."""
    actual_array = np.asarray(actual, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    return bool(
        np.all(
            np.abs(actual_array - reference_array) <= 1.0e-10 + 1.0e-8 * np.abs(reference_array)
        )
    )


def _dense_force(theta: FloatArray, dense: FloatArray) -> FloatArray:
    """Evaluate the networked Kuramoto force with explicit dense loops."""
    count = theta.size
    force = np.zeros(count)
    for i in range(count):
        for j in range(count):
            force[i] += dense[i, j] * np.sin(theta[j] - theta[i])
    return force


def _signed_asymmetric_graph() -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return a 6-node directed graph with signed weights and its dense matrix."""
    row = np.array([0, 0, 1, 2, 3, 4, 5, 5, 2], dtype=np.intp)
    col = np.array([1, 3, 2, 0, 4, 5, 0, 2, 4], dtype=np.intp)
    weight = np.array([0.7, -1.2, 0.4, 0.9, -0.3, 1.5, 0.25, -0.8, 0.6])
    dense = np.zeros((6, 6))
    dense[row, col] = weight
    return row.astype(np.float64), col.astype(np.float64), weight, dense


def test_sparse_kuramoto_scaling_01() -> None:
    """Ring and signed asymmetric graphs match the independent dense force."""
    rng = np.random.default_rng(11)
    theta = rng.uniform(-np.pi, np.pi, 7)
    ring = kuramoto.ring_sparse_coupling(7, 0.8)
    dense_ring = 0.8 * (np.eye(7, k=1) + np.eye(7, k=-1))
    dense_ring[0, 6] = dense_ring[6, 0] = 0.8
    assert _within_budget(
        kuramoto.sparse_networked_kuramoto_force(theta, ring), _dense_force(theta, dense_ring)
    )

    row, col, weight, dense = _signed_asymmetric_graph()
    phases = rng.uniform(-np.pi, np.pi, 6)
    graph = kuramoto.sparse_coupling_from_edges(
        6, row.astype(np.intp), col.astype(np.intp), weight
    )
    via_scipy = kuramoto.sparse_coupling_from_scipy(sparse.csr_array(dense))
    oracle = _dense_force(phases, dense)
    for coupling in (graph, via_scipy):
        assert _within_budget(kuramoto.sparse_networked_kuramoto_force(phases, coupling), oracle)
    omega = rng.normal(size=6)
    step = kuramoto.sparse_kuramoto_euler_trajectory(phases, omega, graph, 0.01, 1)
    assert _within_budget(step[1], phases + 0.01 * (omega + oracle))


def test_sparse_kuramoto_scaling_02() -> None:
    """Repeated entries are summed by default and refused when declared."""
    row = np.array([0, 0, 1, 2, 1, 0])
    col = np.array([1, 1, 0, 2, 2, 1])
    weight = np.array([0.5, 0.25, -1.0, 3.0, 0.0, 0.125])
    snapshot = (row.copy(), col.copy(), weight.copy())
    summed = kuramoto.sparse_coupling_from_edges(3, row, col, weight)
    assert summed.row.tolist() == [0, 1]
    assert summed.col.tolist() == [1, 0]
    assert summed.weight.tolist() == [0.875, -1.0]
    for original, current in zip(snapshot, (row, col, weight), strict=True):
        np.testing.assert_array_equal(original, current)

    with pytest.raises(ValueError, match="repeats a \\(row, col\\) entry"):
        kuramoto.sparse_coupling_from_edges(3, row, col, weight, duplicates="refuse")
    unique = kuramoto.sparse_coupling_from_edges(
        3, [0, 1, 2], [1, 0, 2], [0.5, -1.0, 3.0], duplicates="refuse"
    )
    assert unique.nnz == 2

    repeated = sparse.coo_array((weight, (row, col)), shape=(3, 3))
    assert kuramoto.sparse_coupling_from_scipy(repeated).weight.tolist() == [0.875, -1.0]
    with pytest.raises(ValueError, match="duplicates='refuse'"):
        kuramoto.sparse_coupling_from_scipy(repeated, duplicates="refuse")
    for policy in ("max", "last"):
        with pytest.raises(ValueError, match="'sum' or 'refuse'"):
            kuramoto.sparse_coupling_from_edges(3, row, col, weight, duplicates=policy)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((3, [0, -1], [1, 0], [1.0, 1.0]), "non-negative"),
        ((3, [0, 3], [1, 0], [1.0, 1.0]), "within n_oscillators"),
        ((3, [0, 1], [1], [1.0, 1.0]), "identical one-dimensional shapes"),
        ((3, [0], [1], [np.nan]), "finite"),
        ((-1, [], [], []), "non-negative integer"),
    ],
)
def test_sparse_kuramoto_scaling_02_malformed_edges_refuse(
    arguments: tuple[Any, ...], message: str
) -> None:
    """Out-of-range, mismatched or non-finite edge lists refuse.

    Parameters
    ----------
    arguments
        ``(n_oscillators, row, col, weight)``.
    message
        Expected refusal text.

    """
    with pytest.raises(ValueError, match=message):
        kuramoto.sparse_coupling_from_edges(*arguments)


def _force_peak_bytes(count: int) -> int:
    """Return the tracemalloc peak of one force evaluation on a ring."""
    coupling = kuramoto.ring_sparse_coupling(count, 0.5)
    theta = np.linspace(0.0, 1.0, count)
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        force = kuramoto.sparse_networked_kuramoto_force(theta, coupling)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert force.shape == (count,)
    return peak


def test_sparse_kuramoto_scaling_03() -> None:
    """A large ring stores and evaluates in memory linear in its edges."""
    count = 200_000
    coupling = kuramoto.ring_sparse_coupling(count, 0.5)
    report = coupling.storage_report()
    index_bytes = np.dtype(np.intp).itemsize
    assert report.edges == 2 * count
    assert dict(report.component_bytes) == {
        "row": report.edges * index_bytes,
        "col": report.edges * index_bytes,
        "weight": report.edges * 8,
    }
    assert report.stored_bytes == report.edges * (2 * index_bytes + 8)
    assert report.bytes_per_edge == 2 * index_bytes + 8
    assert report.dense_equivalent_bytes == count * count * 8

    small, large = _force_peak_bytes(count), _force_peak_bytes(2 * count)
    # A handful of float64 edge and node temporaries, never an N x N matrix.
    assert small <= 12 * 8 * (count + report.edges)
    assert 1.5 <= large / small <= 2.5


def test_sparse_kuramoto_scaling_03_edgeless_storage() -> None:
    """An edgeless coupling stores nothing and reports zero bytes per edge."""
    report = kuramoto.ring_sparse_coupling(1).storage_report()
    assert (report.edges, report.stored_bytes, report.bytes_per_edge) == (0, 0, 0.0)


def test_sparse_kuramoto_scaling_04() -> None:
    """The kernels report CPU float64 placement and never a GPU result."""
    report = kuramoto.sparse_kuramoto_execution_report()
    assert report.backend == "cpu"
    assert report.precision == "float64"
    assert report.library == f"numpy {np.__version__}"
    assert report.gpu_status == "not_implemented"
    assert report.gpu_performance_claim is False
    assert kuramoto.sparse_kuramoto_execution_report() == report


def test_new_sparse_symbols_are_facade_capabilities() -> None:
    """The edge-list constructor and reports are public through both facades."""
    from oscillatools import accel

    for name in (
        "sparse_coupling_from_edges",
        "SparseStorageReport",
        "SparseExecutionReport",
        "sparse_kuramoto_execution_report",
    ):
        assert getattr(kuramoto, name) is getattr(sk, name) is getattr(accel, name)
        assert name in kuramoto.capabilities()["forces"]


def test_empty_edge_list_is_an_uncoupled_network() -> None:
    """No edges give zero force under either duplicate policy."""
    for policy in ("sum", "refuse"):
        empty = kuramoto.sparse_coupling_from_edges(4, [], [], [], duplicates=policy)
        assert empty.nnz == 0
        np.testing.assert_array_equal(
            kuramoto.sparse_networked_kuramoto_force(np.ones(4), empty), np.zeros(4)
        )
