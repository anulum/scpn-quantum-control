# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Crypto Hierarchical Keys
"""Tests for hierarchical_keys: SCPN layer key derivation tree."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from scpn_quantum_control.bridge import build_knm_paper27
from scpn_quantum_control.crypto.hierarchical_keys import (
    derive_layer_key,
    derive_master_key,
    evolve_key_phases,
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


def test_evolve_key_phases_ode_failure():
    """ODE solver failure raises RuntimeError (line 138)."""
    K = build_knm_paper27(L=4)
    omega = np.array([1.0, 2.0, 3.0, 4.0])
    theta_0 = np.zeros(4)

    failed_sol = SimpleNamespace(status=-1, message="step size too small")

    with (
        patch(
            "scpn_quantum_control.crypto.hierarchical_keys.solve_ivp",
            return_value=failed_sol,
        ),
        pytest.raises(RuntimeError, match="Phase evolution failed"),
    ):
        evolve_key_phases(K, omega, theta_0, t_window=1.0)


# ---------------------------------------------------------------------------
# Cryptographic invariants: key uniqueness and entropy
# ---------------------------------------------------------------------------


def test_all_layer_keys_unique():
    """All layer keys in a hierarchy must be distinct."""
    K = build_knm_paper27(L=8)
    phases = np.linspace(0, 2 * np.pi, 8)
    h = key_hierarchy(K, phases, R_global=0.5)
    keys = list(h["layers"].values())
    assert len(set(keys)) == 8  # all unique


def test_master_key_differs_from_all_layer_keys():
    K = build_knm_paper27(L=4)
    phases = np.array([0.1, 0.2, 0.3, 0.4])
    h = key_hierarchy(K, phases, R_global=0.7)
    for layer_key in h["layers"].values():
        assert h["master"] != layer_key


def test_evolve_key_phases_finite():
    """Evolved phases must be finite for valid inputs."""
    K = build_knm_paper27(L=4)
    omega = np.array([1.0, 2.0, 3.0, 4.0])
    theta_0 = np.zeros(4)
    traj = evolve_key_phases(K, omega, theta_0, t_window=0.5, n_samples=10)
    assert np.all(np.isfinite(traj))
    assert traj.shape == (4, 10)


# ---------------------------------------------------------------------------
# Pipeline: Knm → key hierarchy → verify → wired
# ---------------------------------------------------------------------------


def test_pipeline_knm_to_key_verification():
    """Full pipeline: build_knm → key hierarchy → verify chain.
    Verifies cryptographic key module is wired end-to-end.
    """
    import time

    from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16

    K = build_knm_paper27(L=4)
    phases = OMEGA_N_16[:4]  # use frequencies as phases
    R = 0.8

    t0 = time.perf_counter()
    h = key_hierarchy(K, phases, R, nonce=b"pipeline")
    verified = verify_key_chain(h["master"], h["layers"], K, phases, R, nonce=b"pipeline")
    dt = (time.perf_counter() - t0) * 1000

    assert verified is True
    assert len(h["master"]) == 32
    assert len(h["layers"]) == 4

    print(f"\n  PIPELINE Knm→KeyHierarchy→Verify (4 layers): {dt:.2f} ms")
    print(f"  Master key: {h['master'][:8].hex()}...")
