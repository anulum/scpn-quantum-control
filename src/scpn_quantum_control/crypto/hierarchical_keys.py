# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Hierarchical Keys
"""SCPN layer hierarchy to key derivation tree.

The 16-layer SCPN hierarchy maps to a key tree where:
- Master key = hash(K_nm_full || R_global || nonce)
- Layer-n subkey = hash(K_nm[n,:] || phase_sequence_n || nonce)

Time-varying keys: Kuramoto phase sequences add temporal entropy.
Different time windows produce different keys from the same K_nm.

Derived keys are held in ordinary Python ``bytes`` with no memory
zeroisation, so they must be assumed to persist in process memory until
interpreter exit.

Ref: Improved group QKD with multi-party collaboration, Sci. Reports 2025
"""

from __future__ import annotations

import hashlib
import hmac
import struct
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp


def derive_master_key(
    K: NDArray[np.float64],
    R_global: float,
    nonce: bytes = b"",
) -> bytes:
    """Derive a master key from the coupling matrix and order parameter.

    Parameters
    ----------
    K
        Full coupling matrix in canonical row-major representation.
    R_global
        Global order parameter bound into the digest.
    nonce
        Optional caller-supplied session nonce.

    Returns
    -------
    bytes
        The 32-byte SHA-256 digest of ``K || R_global || nonce``.

    """
    h = hashlib.sha256()
    h.update(K.tobytes())
    h.update(struct.pack("!d", R_global))
    h.update(nonce)
    return h.digest()


def derive_layer_key(
    K: NDArray[np.float64],
    layer_idx: int,
    phase_sequence: NDArray[np.float64],
    nonce: bytes = b"",
) -> bytes:
    """Derive a layer subkey from one coupling row and phase trajectory.

    Parameters
    ----------
    K
        Full coupling matrix; the selected row is bound into the digest.
    layer_idx
        Zero-indexed layer number.
    phase_sequence
        Phase values for the selected layer over the time window.
    nonce
        Optional caller-supplied session nonce.

    Returns
    -------
    bytes
        The 32-byte SHA-256 layer-key digest.

    """
    h = hashlib.sha256()
    h.update(K[layer_idx, :].tobytes())
    h.update(struct.pack("!i", layer_idx))
    h.update(phase_sequence.tobytes())
    h.update(nonce)
    return h.digest()


def key_hierarchy(
    K: NDArray[np.float64],
    phases: NDArray[np.float64],
    R_global: float,
    nonce: bytes = b"",
) -> dict[str, Any]:
    """Derive the master key and every layer subkey.

    Parameters
    ----------
    K
        Square coupling matrix.
    phases
        Current phase value for every layer.
    R_global
        Global order parameter bound into the master key.
    nonce
        Optional session nonce bound into every key.

    Returns
    -------
    dict[str, Any]
        Mapping with the master key under ``master`` and indexed layer keys
        under ``layers``.

    """
    master = derive_master_key(K, R_global, nonce)
    n = K.shape[0]
    layers = {}
    for i in range(n):
        layers[i] = derive_layer_key(K, i, phases[i : i + 1], nonce)
    return {"master": master, "layers": layers}


def verify_key_chain(
    master: bytes,
    layer_keys: dict[int, bytes],
    K: NDArray[np.float64],
    phases: NDArray[np.float64],
    R_global: float,
    nonce: bytes = b"",
) -> bool:
    """Verify the master and layer keys against their derivation inputs.

    Parameters
    ----------
    master
        Candidate master-key bytes.
    layer_keys
        Candidate keys indexed by layer.
    K
        Full coupling matrix used for recomputation.
    phases
        Current phase value for every layer.
    R_global
        Global order parameter used for recomputation.
    nonce
        Session nonce used for recomputation.

    Returns
    -------
    bool
        Whether the master and every supplied layer key match recomputation.

    """
    expected = key_hierarchy(K, phases, R_global, nonce)
    if master != expected["master"]:
        return False
    return all(key == expected["layers"].get(idx) for idx, key in layer_keys.items())


# --- Time-Evolving Key Rotation ---


def _kuramoto_rhs(
    t: float, theta: NDArray[np.float64], K: NDArray[np.float64], omega: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Kuramoto ODE right-hand side: dθ/dt = ω + Σ K_nm sin(θ_m - θ_n)."""
    n = len(theta)
    dtheta: NDArray[np.float64] = omega.copy()
    for i in range(n):
        for j in range(n):
            dtheta[i] += K[i, j] * np.sin(theta[j] - theta[i])
    return dtheta


def evolve_key_phases(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    theta_0: NDArray[np.float64],
    t_window: float,
    n_samples: int = 32,
) -> NDArray[np.float64]:
    """Evolve Kuramoto dynamics and sample the phase trajectory.

    Parameters
    ----------
    K
        Coupling matrix for the layer oscillators.
    omega
        Intrinsic angular frequency of every layer.
    theta_0
        Initial phase of every layer.
    t_window
        Positive integration-window duration.
    n_samples
        Number of evenly spaced trajectory samples.

    Returns
    -------
    numpy.ndarray
        Phase array shaped ``(n_layers, n_samples)``.

    Raises
    ------
    RuntimeError
        If the numerical integrator does not complete successfully.

    """
    t_eval = np.linspace(0, t_window, n_samples)
    sol = solve_ivp(
        _kuramoto_rhs,
        (0, t_window),
        theta_0,
        args=(K, omega),
        t_eval=t_eval,
        method="RK45",
    )
    if sol.status != 0:
        raise RuntimeError(f"Phase evolution failed: {sol.message}")
    result: NDArray[np.float64] = np.asarray(sol.y)
    return result  # shape (n_layers, n_samples)


def rotating_key_schedule(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    theta_0: NDArray[np.float64],
    n_windows: int = 4,
    window_duration: float = 1.0,
) -> list[dict[str, Any]]:
    """Generate a sequence of key hierarchies from evolving Kuramoto dynamics.

    Parameters
    ----------
    K
        Coupling matrix for the layer oscillators.
    omega
        Intrinsic angular frequency of every layer.
    theta_0
        Initial phase of every layer.
    n_windows
        Number of consecutive rotation windows.
    window_duration
        Integration duration of each window.

    Returns
    -------
    list[dict[str, Any]]
        One hierarchy per window with its index, keys, order parameter, and
        final phases. Each window starts from the preceding final phases.

    """
    theta = theta_0.copy()
    schedule = []

    for w in range(n_windows):
        trajectory = evolve_key_phases(K, omega, theta, window_duration)
        # R_global from final phases
        final_phases = trajectory[:, -1]
        R = float(abs(np.mean(np.exp(1j * final_phases))))
        nonce = struct.pack("!i", w)
        h = key_hierarchy(K, final_phases, R, nonce)
        schedule.append(
            {
                "window": w,
                "master": h["master"],
                "layers": h["layers"],
                "R_global": R,
                "final_phases": final_phases,
            }
        )
        theta = final_phases  # chain windows

    return schedule


# --- Group Key Agreement ---


def group_key(
    K: NDArray[np.float64],
    member_layers: list[int],
    phases: NDArray[np.float64],
    nonce: bytes = b"",
) -> bytes:
    """Derive a shared key for a subset of SCPN layers.

    Parameters
    ----------
    K
        Full coupling matrix.
    member_layers
        Layer indices included in the key group.
    phases
        Current phase value for every layer.
    nonce
        Optional session nonce.

    Returns
    -------
    bytes
        The 32-byte SHA-256 digest of the selected submatrix, sorted member
        indices, selected phases, and nonce.

    """
    sub_K = K[np.ix_(member_layers, member_layers)]
    sub_phases = phases[member_layers]
    h = hashlib.sha256()
    h.update(sub_K.tobytes())
    for idx in sorted(member_layers):
        h.update(struct.pack("!i", idx))
    h.update(sub_phases.tobytes())
    h.update(nonce)
    return h.digest()


def hmac_verify_key(key: bytes, message: bytes, expected_mac: bytes) -> bool:
    """Verify an HMAC-SHA256 tag with constant-time digest comparison.

    Parameters
    ----------
    key
        HMAC key bytes.
    message
        Authenticated message bytes.
    expected_mac
        Candidate HMAC-SHA256 tag.

    Returns
    -------
    bool
        Whether the computed and supplied tags match.

    """
    computed = hmac.new(key, message, hashlib.sha256).digest()
    return hmac.compare_digest(computed, expected_mac)


def hmac_sign(key: bytes, message: bytes) -> bytes:
    """Produce an HMAC-SHA256 authentication tag.

    Parameters
    ----------
    key
        HMAC key bytes.
    message
        Message bytes to authenticate.

    Returns
    -------
    bytes
        The 32-byte HMAC-SHA256 tag.

    """
    return hmac.new(key, message, hashlib.sha256).digest()
