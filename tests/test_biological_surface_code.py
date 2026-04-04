# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Project Configuration
"""Tests for Biological Surface Code and MWPM."""

import numpy as np

from scpn_quantum_control.qec.biological_surface_code import (
    BiologicalMWPMDecoder,
    BiologicalSurfaceCode,
)


def test_biological_surface_code_commutation():
    # Simple 4-node ring graph
    K = np.array(
        [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]
    )
    code = BiologicalSurfaceCode(K)

    assert code.num_data == 4  # 4 edges
    assert code.num_x_stabs == 4  # 4 nodes
    assert code.num_z_stabs == 1  # 1 fundamental cycle

    assert bool(code.verify_css_commutation()) is True


def test_biological_mwpm_decoder_single_error():
    # 4-node string (line graph)
    K = np.array(
        [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]
    )
    code = BiologicalSurfaceCode(K)
    decoder = BiologicalMWPMDecoder(code)

    # Introduce a Z error on the middle edge (node 1-2)
    err_z = np.zeros(code.num_data, dtype=np.int8)
    e_idx = code.edge_to_idx[(1, 2)]
    err_z[e_idx] = 1

    # Compute X syndrome
    syn_x = (code.Hx @ err_z) % 2

    # Decode
    correction = decoder.decode_z_errors(syn_x)

    # Check if correction matches error (or is equivalent up to stabilizer)
    assert np.array_equal(correction, err_z)


def test_mwpm_decoder_no_defects():
    """Empty syndrome → zero correction."""
    K = np.array(
        [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]
    )
    code = BiologicalSurfaceCode(K)
    decoder = BiologicalMWPMDecoder(code)
    syn_x = np.zeros(code.num_x_stabs, dtype=np.int8)
    correction = decoder.decode_z_errors(syn_x)
    assert np.all(correction == 0)


def test_mwpm_decoder_odd_defects():
    """Odd number of defects triggers truncation to even."""
    K = np.array(
        [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]
    )
    code = BiologicalSurfaceCode(K)
    decoder = BiologicalMWPMDecoder(code)
    # Manually create syndrome with 3 defects (odd)
    syn_x = np.zeros(code.num_x_stabs, dtype=np.int8)
    syn_x[0] = 1
    syn_x[1] = 1
    syn_x[2] = 1
    correction = decoder.decode_z_errors(syn_x)
    assert correction.shape == (code.num_data,)


def test_mwpm_decoder_disconnected_nodes():
    """Two disconnected components cannot form a path → NetworkXNoPath branch."""
    # Block-diagonal coupling: nodes {0,1} coupled, nodes {2,3} coupled, no cross
    K = np.array(
        [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]
    )
    code = BiologicalSurfaceCode(K)
    decoder = BiologicalMWPMDecoder(code)
    # Defects in different components: node 0 and node 2
    syn_x = np.zeros(code.num_x_stabs, dtype=np.int8)
    syn_x[0] = 1
    syn_x[2] = 1
    correction = decoder.decode_z_errors(syn_x)
    assert correction.shape == (code.num_data,)


def test_correction_clears_syndrome():
    """Error + correction must zero the syndrome (mod 2)."""
    K = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=float)
    code = BiologicalSurfaceCode(K)
    decoder = BiologicalMWPMDecoder(code)
    err_z = np.zeros(code.num_data, dtype=np.int8)
    e_idx = code.edge_to_idx[(1, 2)]
    err_z[e_idx] = 1
    syn_x = (code.Hx @ err_z) % 2
    correction = decoder.decode_z_errors(syn_x)
    residual_syn = (code.Hx @ ((err_z + correction) % 2)) % 2
    assert np.all(residual_syn == 0)


def test_complete_graph_css_commutation():
    """Complete graph (all-to-all) must still satisfy CSS commutation."""
    n = 5
    K = np.ones((n, n)) - np.eye(n)
    code = BiologicalSurfaceCode(K)
    assert code.verify_css_commutation()
    assert code.num_data == n * (n - 1) // 2


def test_two_error_correction():
    """Decoder handles two non-adjacent errors."""
    K = np.array(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0],
        ],
        dtype=float,
    )
    code = BiologicalSurfaceCode(K)
    decoder = BiologicalMWPMDecoder(code)
    err_z = np.zeros(code.num_data, dtype=np.int8)
    err_z[code.edge_to_idx[(0, 1)]] = 1
    err_z[code.edge_to_idx[(3, 4)]] = 1
    syn_x = (code.Hx @ err_z) % 2
    assert np.sum(syn_x) == 4  # 4 defects
    correction = decoder.decode_z_errors(syn_x)
    residual = (code.Hx @ ((err_z + correction) % 2)) % 2
    assert np.all(residual == 0)


def test_six_node_ring():
    """6-node ring has 6 edges, 6 X-stabilisers, and >= 1 Z-stabiliser."""
    n = 6
    K = np.zeros((n, n))
    for i in range(n):
        K[i, (i + 1) % n] = 1.0
        K[(i + 1) % n, i] = 1.0
    code = BiologicalSurfaceCode(K)
    assert code.num_data == 6
    assert code.num_x_stabs == 6
    assert code.num_z_stabs >= 1
    assert code.verify_css_commutation()


def test_threshold_filters_weak_edges():
    """Edges below threshold are excluded from the code."""
    K = np.array([[0, 1.0, 0.001], [1.0, 0, 0.5], [0.001, 0.5, 0]])
    code = BiologicalSurfaceCode(K, threshold=0.01)
    assert code.num_data == 2  # Only (0,1) and (1,2)


def test_no_edges_raises():
    """Zero-edge coupling matrix must raise ValueError."""
    import pytest

    K = np.zeros((3, 3))
    with pytest.raises(ValueError, match="no edges"):
        BiologicalSurfaceCode(K)
