# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — tests for KYMA v2 gated-coupling assembly
"""Tests for the base coupling, readout bridge, and gated assembly (fix 1)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from scpn_quantum_control.benchmarks.kyma_v2 import coupling, task  # noqa: E402


def test_ambient_matrix_uniform_zero_diagonal() -> None:
    mat = coupling.ambient_matrix(0.3)
    assert mat.shape == (task.N_OSC, task.N_OSC)
    assert not mat.diagonal().any()
    off = mat[~np.eye(task.N_OSC, dtype=bool)]
    assert np.allclose(off, 0.3)


def test_readout_bridge_is_symmetric_and_sparse() -> None:
    mat = coupling.readout_bridge_matrix(0.5)
    assert np.array_equal(mat, mat.T)
    partners = coupling._readout_bridge_partners()
    assert len(partners) == 2
    for p in partners:
        assert mat[task.READOUT_OSCILLATOR, p] == 0.5
    # exactly 2 undirected bridge edges → 4 nonzero entries
    assert np.count_nonzero(mat) == 2 * len(partners)


def test_bridge_partners_span_both_held_out_relations() -> None:
    partners = coupling._readout_bridge_partners()
    r1_clusters = set(task.PAIRS[task.HELD_OUT_R1_PAIR])
    r2_clusters = set(task.PAIRS[task.HELD_OUT_R2_PAIR])
    clusters_of = [next(c for c, m in enumerate(task.CLUSTERS) if p in m) for p in partners]
    # one partner from an R1 cluster, one from an R2 cluster — no anti-phase cancellation
    assert any(c in r1_clusters for c in clusters_of)
    assert any(c in r2_clusters for c in clusters_of)


def test_bridge_does_not_touch_intra_cluster_edges() -> None:
    # The bridge must not add coupling inside a cluster (would disturb the motif).
    mat = coupling.readout_bridge_matrix(1.0)
    for cluster in task.CLUSTERS:
        for i in cluster:
            for j in cluster:
                assert mat[i, j] == 0.0


def test_base_coupling_is_ambient_plus_bridge() -> None:
    base = coupling.base_coupling_matrix(0.1, 0.5)
    expected = coupling.ambient_matrix(0.1) + coupling.readout_bridge_matrix(0.5)
    assert np.allclose(base, expected)


def test_assemble_coupling_gates_by_code() -> None:
    base = jnp.asarray(coupling.base_coupling_matrix(0.0, 0.0))
    gates = np.zeros((task.N_RELATIONS, task.N_PAIRS, task.N_OSC, task.N_OSC))
    gates[0, 2, 1, 3] = gates[0, 2, 3, 1] = 7.0  # a distinctive motif edge
    gates_j = jnp.asarray(gates)
    # code activating (R1, pair 2) → that gate appears; inactive code → zero.
    active = task.encode(2, -1)[None]
    inactive = task.encode(4, -1)[None]
    k_active = np.asarray(coupling.assemble_coupling(jnp.asarray(active), gates_j, base))[0]
    k_inactive = np.asarray(coupling.assemble_coupling(jnp.asarray(inactive), gates_j, base))[0]
    assert k_active[1, 3] == pytest.approx(7.0)
    assert k_inactive[1, 3] == pytest.approx(0.0)


def test_symmetrise_projects_to_symmetric_zero_diagonal() -> None:
    raw = jnp.asarray(
        np.arange(task.N_RELATIONS * task.N_PAIRS * task.N_OSC * task.N_OSC, dtype=float).reshape(
            task.N_RELATIONS, task.N_PAIRS, task.N_OSC, task.N_OSC
        )
    )
    sym = np.asarray(coupling.symmetrise(raw))
    assert np.allclose(sym, np.swapaxes(sym, -1, -2))
    assert np.allclose(np.diagonal(sym, axis1=-2, axis2=-1), 0.0)
