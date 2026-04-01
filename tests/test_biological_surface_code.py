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
