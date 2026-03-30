# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Topology Auth Edge
"""Cover edge case in topology_auth.py: line 83 (no positive eigenvalues → entropy=0)."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.crypto.topology_auth import normalized_laplacian_fingerprint


def test_normalized_laplacian_zero_entropy():
    """1x1 self-loop K → L_sym = 0 → no positive eigenvalues → entropy = 0."""
    K = np.array([[1.0]])
    fp = normalized_laplacian_fingerprint(K)
    assert fp["spectral_entropy_norm"] == 0.0
