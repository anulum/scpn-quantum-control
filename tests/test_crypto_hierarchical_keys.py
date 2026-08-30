# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Crypto Hierarchical Keys
"""Tests for hierarchical_keys: SCPN layer key derivation tree."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.bridge import build_knm_paper27
from scpn_quantum_control.crypto.hierarchical_keys import (
    derive_layer_key,
    derive_master_key,
    evolve_key_phases,
    key_hierarchy,
    verify_key_chain,
)


def test_master_key_deterministic() -> None:
    """Derive identical master bytes from identical inputs."""
    K = build_knm_paper27(L=4)
    k1 = derive_master_key(K, R_global=0.8, nonce=b"test")
    k2 = derive_master_key(K, R_global=0.8, nonce=b"test")
    assert k1 == k2


def test_master_key_changes_with_R() -> None:
    """Bind the global order parameter into the master key."""
    K = build_knm_paper27(L=4)
    k1 = derive_master_key(K, R_global=0.8)
    k2 = derive_master_key(K, R_global=0.5)
    assert k1 != k2


def test_master_key_changes_with_nonce() -> None:
    """Bind the caller-supplied nonce into the master key."""
    K = build_knm_paper27(L=4)
    k1 = derive_master_key(K, R_global=0.8, nonce=b"a")
    k2 = derive_master_key(K, R_global=0.8, nonce=b"b")
    assert k1 != k2


def test_master_key_length() -> None:
    """Return a full 32-byte SHA-256 master digest."""
    K = build_knm_paper27(L=4)
    key = derive_master_key(K, R_global=0.8)
    assert len(key) == 32  # SHA-256


def test_layer_keys_differ() -> None:
    """Derive distinct keys for distinct coupling rows and layers."""
    K = build_knm_paper27(L=4)
    phases = np.array([0.1, 0.2, 0.3, 0.4])
    k0 = derive_layer_key(K, 0, phases[:1])
    k1 = derive_layer_key(K, 1, phases[1:2])
    assert k0 != k1


def test_key_hierarchy_completeness() -> None:
    """Produce a master key and one subkey for every layer."""
    K = build_knm_paper27(L=4)
    phases = np.array([0.1, 0.2, 0.3, 0.4])
    h = key_hierarchy(K, phases, R_global=0.7)
    assert "master" in h
    assert "layers" in h
    assert len(h["layers"]) == 4
    for i in range(4):
        assert i in h["layers"]
        assert len(h["layers"][i]) == 32


def test_verify_key_chain_correct() -> None:
    """Accept a freshly derived master and layer-key chain."""
    K = build_knm_paper27(L=4)
    phases = np.array([0.1, 0.2, 0.3, 0.4])
    h = key_hierarchy(K, phases, R_global=0.7, nonce=b"verify")
    assert verify_key_chain(h["master"], h["layers"], K, phases, 0.7, b"verify")


def test_verify_key_chain_wrong_K() -> None:
    """Reject a key chain recomputed from a different coupling matrix."""
    K = build_knm_paper27(L=4)
    phases = np.array([0.1, 0.2, 0.3, 0.4])
    h = key_hierarchy(K, phases, R_global=0.7)
    K_wrong = K * 1.1
    assert not verify_key_chain(h["master"], h["layers"], K_wrong, phases, 0.7)


def test_verify_key_chain_wrong_R() -> None:
    """Reject a key chain recomputed from a different order parameter."""
    K = build_knm_paper27(L=4)
    phases = np.array([0.1, 0.2, 0.3, 0.4])
    h = key_hierarchy(K, phases, R_global=0.7)
    assert not verify_key_chain(h["master"], h["layers"], K, phases, 0.9)


def test_evolve_key_phases_ode_failure() -> None:
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


def test_all_layer_keys_unique() -> None:
    """All layer keys in a hierarchy must be distinct."""
    K = build_knm_paper27(L=8)
    phases = cast(NDArray[np.float64], np.linspace(0, 2 * np.pi, 8))
    h = key_hierarchy(K, phases, R_global=0.5)
    keys = list(h["layers"].values())
    assert len(set(keys)) == 8  # all unique


def test_master_key_differs_from_all_layer_keys() -> None:
    """Keep the master digest distinct from every layer digest."""
    K = build_knm_paper27(L=4)
    phases = np.array([0.1, 0.2, 0.3, 0.4])
    h = key_hierarchy(K, phases, R_global=0.7)
    for layer_key in h["layers"].values():
        assert h["master"] != layer_key


def test_evolve_key_phases_finite() -> None:
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


def test_pipeline_knm_to_key_verification() -> None:
    """Run the full coupling-matrix to verified-key-chain pipeline.

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
