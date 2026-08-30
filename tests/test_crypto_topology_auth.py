# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Crypto Topology Auth
"""Tests for topology_auth: spectral fingerprint authentication."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.bridge import build_knm_paper27
from scpn_quantum_control.crypto.topology_auth import (
    spectral_fingerprint,
    topology_distance,
    verify_fingerprint,
)


def test_fingerprint_has_required_fields() -> None:
    """Return every public combinatorial fingerprint field."""
    K = build_knm_paper27(L=4)
    fp = spectral_fingerprint(K)
    assert "fiedler" in fp
    assert "gap_ratio" in fp
    assert "spectral_entropy" in fp
    assert "n_components" in fp
    assert "eigenvalues" in fp


def test_fiedler_positive_for_connected_graph() -> None:
    """Return positive algebraic connectivity for the K_nm graph."""
    K = build_knm_paper27(L=16)
    fp = spectral_fingerprint(K)
    assert fp["fiedler"] > 0, "K_nm graph should be connected"


def test_single_component() -> None:
    """Classify the connected K_nm graph as one component."""
    K = build_knm_paper27(L=16)
    fp = spectral_fingerprint(K)
    assert fp["n_components"] == 1


def test_verify_fingerprint_correct() -> None:
    """Verify an unchanged matrix against its own fingerprint."""
    K = build_knm_paper27(L=8)
    fp = spectral_fingerprint(K)
    assert verify_fingerprint(K, fp)


def test_verify_fingerprint_wrong_matrix() -> None:
    """Reject a rescaled matrix against the original fingerprint."""
    K = build_knm_paper27(L=8)
    fp = spectral_fingerprint(K)
    K_wrong = K * 2.0
    assert not verify_fingerprint(K_wrong, fp)


def test_topology_distance_self_zero() -> None:
    """Return zero distance for identical fingerprints."""
    K = build_knm_paper27(L=4)
    fp = spectral_fingerprint(K)
    assert topology_distance(fp, fp) < 1e-12


def test_topology_distance_different() -> None:
    """Return positive distance for rescaled coupling topology."""
    K1 = build_knm_paper27(L=4)
    K2 = 0.5 * K1
    fp1 = spectral_fingerprint(K1)
    fp2 = spectral_fingerprint(K2)
    assert topology_distance(fp1, fp2) > 0.1


def test_spectral_entropy_positive() -> None:
    """Return positive entropy for a nontrivial connected graph."""
    K = build_knm_paper27(L=8)
    fp = spectral_fingerprint(K)
    assert fp["spectral_entropy"] > 0


def test_spectral_fingerprint_keys() -> None:
    """Expose eigenvalues and entropy in the public fingerprint."""
    K = build_knm_paper27(L=4)
    fp = spectral_fingerprint(K)
    assert "eigenvalues" in fp
    assert "spectral_entropy" in fp


def test_spectral_fingerprint_deterministic() -> None:
    """Reproduce exact eigenvalues for the same matrix."""
    K = build_knm_paper27(L=4)
    fp1 = spectral_fingerprint(K)
    fp2 = spectral_fingerprint(K)
    np.testing.assert_array_equal(fp1["eigenvalues"], fp2["eigenvalues"])


def test_spectral_fingerprint_various_sizes() -> None:
    """Return one eigenvalue per graph node across supported sizes."""
    for L in [2, 4, 8]:
        K = build_knm_paper27(L=L)
        fp = spectral_fingerprint(K)
        assert len(fp["eigenvalues"]) == L


def test_zero_matrix_has_zero_spectral_entropy() -> None:
    """Return zero entropy when no positive Laplacian eigenvalue exists."""
    fp = spectral_fingerprint(np.zeros((3, 3), dtype=np.float64))
    assert fp["spectral_entropy"] == 0.0
