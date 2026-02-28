"""Tests for topology_auth: spectral fingerprint authentication."""

from __future__ import annotations

from scpn_quantum_control.bridge import build_knm_paper27
from scpn_quantum_control.crypto.topology_auth import (
    spectral_fingerprint,
    topology_distance,
    verify_fingerprint,
)


def test_fingerprint_has_required_fields():
    K = build_knm_paper27(L=4)
    fp = spectral_fingerprint(K)
    assert "fiedler" in fp
    assert "gap_ratio" in fp
    assert "spectral_entropy" in fp
    assert "n_components" in fp
    assert "eigenvalues" in fp


def test_fiedler_positive_for_connected_graph():
    K = build_knm_paper27(L=16)
    fp = spectral_fingerprint(K)
    assert fp["fiedler"] > 0, "K_nm graph should be connected"


def test_single_component():
    K = build_knm_paper27(L=16)
    fp = spectral_fingerprint(K)
    assert fp["n_components"] == 1


def test_verify_fingerprint_correct():
    K = build_knm_paper27(L=8)
    fp = spectral_fingerprint(K)
    assert verify_fingerprint(K, fp)


def test_verify_fingerprint_wrong_matrix():
    K = build_knm_paper27(L=8)
    fp = spectral_fingerprint(K)
    K_wrong = K * 2.0
    assert not verify_fingerprint(K_wrong, fp)


def test_topology_distance_self_zero():
    K = build_knm_paper27(L=4)
    fp = spectral_fingerprint(K)
    assert topology_distance(fp, fp) < 1e-12


def test_topology_distance_different():
    K1 = build_knm_paper27(L=4)
    K2 = 0.5 * K1
    fp1 = spectral_fingerprint(K1)
    fp2 = spectral_fingerprint(K2)
    assert topology_distance(fp1, fp2) > 0.1


def test_spectral_entropy_positive():
    K = build_knm_paper27(L=8)
    fp = spectral_fingerprint(K)
    assert fp["spectral_entropy"] > 0
