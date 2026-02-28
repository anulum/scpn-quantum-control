"""Tests for hierarchical_keys: SCPN layer key derivation tree."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.bridge import build_knm_paper27
from scpn_quantum_control.crypto.hierarchical_keys import (
    derive_layer_key,
    derive_master_key,
    key_hierarchy,
    verify_key_chain,
)


def test_master_key_deterministic():
    K = build_knm_paper27(L=4)
    k1 = derive_master_key(K, R_global=0.8, nonce=b"test")
    k2 = derive_master_key(K, R_global=0.8, nonce=b"test")
    assert k1 == k2


def test_master_key_changes_with_R():
    K = build_knm_paper27(L=4)
    k1 = derive_master_key(K, R_global=0.8)
    k2 = derive_master_key(K, R_global=0.5)
    assert k1 != k2


def test_master_key_changes_with_nonce():
    K = build_knm_paper27(L=4)
    k1 = derive_master_key(K, R_global=0.8, nonce=b"a")
    k2 = derive_master_key(K, R_global=0.8, nonce=b"b")
    assert k1 != k2


def test_master_key_length():
    K = build_knm_paper27(L=4)
    key = derive_master_key(K, R_global=0.8)
    assert len(key) == 32  # SHA-256


def test_layer_keys_differ():
    K = build_knm_paper27(L=4)
    phases = np.array([0.1, 0.2, 0.3, 0.4])
    k0 = derive_layer_key(K, 0, phases[:1])
    k1 = derive_layer_key(K, 1, phases[1:2])
    assert k0 != k1


def test_key_hierarchy_completeness():
    K = build_knm_paper27(L=4)
    phases = np.array([0.1, 0.2, 0.3, 0.4])
    h = key_hierarchy(K, phases, R_global=0.7)
    assert "master" in h
    assert "layers" in h
    assert len(h["layers"]) == 4
    for i in range(4):
        assert i in h["layers"]
        assert len(h["layers"][i]) == 32


def test_verify_key_chain_correct():
    K = build_knm_paper27(L=4)
    phases = np.array([0.1, 0.2, 0.3, 0.4])
    h = key_hierarchy(K, phases, R_global=0.7, nonce=b"verify")
    assert verify_key_chain(h["master"], h["layers"], K, phases, 0.7, b"verify")


def test_verify_key_chain_wrong_K():
    K = build_knm_paper27(L=4)
    phases = np.array([0.1, 0.2, 0.3, 0.4])
    h = key_hierarchy(K, phases, R_global=0.7)
    K_wrong = K * 1.1
    assert not verify_key_chain(h["master"], h["layers"], K_wrong, phases, 0.7)


def test_verify_key_chain_wrong_R():
    K = build_knm_paper27(L=4)
    phases = np.array([0.1, 0.2, 0.3, 0.4])
    h = key_hierarchy(K, phases, R_global=0.7)
    assert not verify_key_chain(h["master"], h["layers"], K, phases, 0.9)
