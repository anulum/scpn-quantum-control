# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Topology Auth
"""Spectral fingerprint authentication for K_nm topology.

The Laplacian spectrum of K_nm provides a public authentication token.
Co-spectral graphs (different K_nm with same spectrum) exist, so
publishing the spectrum doesn't reveal K_nm — but any party with the
true K_nm can verify consistency.

The secret K_nm is used directly as HMAC key material and lives in
ordinary NumPy arrays with no memory zeroisation, so it must be assumed
to persist in process memory until interpreter exit.

Refs:
- Van Dam & Haemers (2003), "Which graphs are determined by their spectrum?"
- AAAI Symposium 2025, "Quantum Network Science: Graph Structure to Entanglement Performance"
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import entropy as scipy_entropy

EIGENVALUE_ZERO_ATOL = 1e-12  # absolute tolerance for Laplacian zero eigenvalues
EIGENVALUE_ZERO_RTOL = 1e-8  # relative tolerance for Laplacian eigenvalue ratios


def spectral_fingerprint(K: NDArray[np.float64]) -> dict[str, Any]:
    """Compute public spectral fingerprint of coupling matrix.

    Parameters
    ----------
    K
        Square coupling matrix.

    Returns
    -------
    dict
        Fiedler value, gap ratio, spectral entropy, component count, and sorted
        combinatorial-Laplacian eigenvalues.

    """
    n = K.shape[0]
    D = np.diag(K.sum(axis=1))
    L = D - K

    eigvals = np.sort(np.linalg.eigvalsh(L))

    fiedler = float(eigvals[1]) if n > 1 else 0.0
    gap_ratio = (
        float(eigvals[1] / eigvals[2]) if n > 2 and eigvals[2] > EIGENVALUE_ZERO_ATOL else 0.0
    )

    pos_eigvals = eigvals[eigvals > EIGENVALUE_ZERO_ATOL]
    if len(pos_eigvals) > 0:
        p = pos_eigvals / pos_eigvals.sum()
        s_entropy = float(scipy_entropy(p, base=2))
    else:
        s_entropy = 0.0

    n_components = int(np.sum(eigvals < EIGENVALUE_ZERO_RTOL))

    return {
        "fiedler": fiedler,
        "gap_ratio": gap_ratio,
        "spectral_entropy": s_entropy,
        "n_components": n_components,
        "eigenvalues": eigvals.tolist(),
    }


def normalized_laplacian_fingerprint(K: NDArray[np.float64]) -> dict[str, Any]:
    """Fingerprint from the normalized Laplacian L_sym = I - D^{-1/2} K D^{-1/2}.

    More robust to degree heterogeneity than the combinatorial Laplacian.
    Eigenvalues lie in [0, 2] for connected graphs.

    Parameters
    ----------
    K
        Square coupling matrix.

    Returns
    -------
    dict
        Normalized Fiedler value, spectral entropy, eigenvalues, and radius.

    """
    n = K.shape[0]
    d = K.sum(axis=1)
    d_inv_sqrt = np.zeros_like(d, dtype=float)
    positive_degree = d > EIGENVALUE_ZERO_ATOL
    d_inv_sqrt[positive_degree] = 1.0 / np.sqrt(d[positive_degree])
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_sym = np.eye(n) - D_inv_sqrt @ K @ D_inv_sqrt

    eigvals = np.sort(np.linalg.eigvalsh(L_sym))
    fiedler = float(eigvals[1]) if n > 1 else 0.0

    pos_eigvals = eigvals[eigvals > EIGENVALUE_ZERO_ATOL]
    if len(pos_eigvals) > 0:
        p = pos_eigvals / pos_eigvals.sum()
        s_entropy = float(scipy_entropy(p, base=2))
    else:
        s_entropy = 0.0

    return {
        "fiedler_norm": fiedler,
        "spectral_entropy_norm": s_entropy,
        "eigenvalues_norm": eigvals.tolist(),
        "spectral_radius": float(eigvals[-1]),
    }


def verify_fingerprint(
    K: NDArray[np.float64], fingerprint: dict[str, Any], tol: float = 1e-6
) -> bool:
    """Check a coupling matrix against a claimed spectral fingerprint.

    Parameters
    ----------
    K
        Coupling matrix to fingerprint.
    fingerprint
        Claimed public fingerprint fields.
    tol
        Absolute tolerance for scalar fingerprint fields.

    Returns
    -------
    bool
        Whether scalar fields and the component count match.

    """
    computed = spectral_fingerprint(K)
    return bool(
        abs(computed["fiedler"] - fingerprint["fiedler"]) < tol
        and abs(computed["spectral_entropy"] - fingerprint["spectral_entropy"]) < tol
        and computed["n_components"] == fingerprint["n_components"]
    )


def topology_distance(fp1: dict[str, Any], fp2: dict[str, Any]) -> float:
    """L2 distance between two spectral fingerprints.

    Useful for detecting calibration drift or K_nm tampering.

    Parameters
    ----------
    fp1
        First spectral fingerprint.
    fp2
        Second spectral fingerprint.

    Returns
    -------
    float
        Euclidean eigenvalue distance, or infinity for unequal dimensions.

    """
    e1 = np.array(fp1["eigenvalues"])
    e2 = np.array(fp2["eigenvalues"])
    if len(e1) != len(e2):
        return float("inf")
    return float(np.linalg.norm(e1 - e2))


# --- Challenge-Response Authentication ---


def topology_commitment(K: NDArray[np.float64], nonce: bytes = b"") -> bytes:
    """Commit to K_nm without revealing it.

    Returns SHA-256(K_nm_bytes || nonce). The commitment binds the prover
    to a specific K_nm. Later, the prover opens by revealing K_nm + nonce,
    and the verifier recomputes the hash.

    Parameters
    ----------
    K
        Coupling matrix serialized as commitment material.
    nonce
        Optional public commitment nonce.

    Returns
    -------
    bytes
        SHA-256 commitment digest.

    """
    h = hashlib.sha256()
    h.update(K.tobytes())
    h.update(nonce)
    return h.digest()


def verify_commitment(K: NDArray[np.float64], nonce: bytes, commitment: bytes) -> bool:
    """Verify that a coupling matrix matches a prior commitment.

    Parameters
    ----------
    K
        Candidate coupling matrix.
    nonce
        Nonce used to create the commitment.
    commitment
        Claimed SHA-256 commitment digest.

    Returns
    -------
    bool
        Whether recomputation matches the supplied digest.

    """
    return topology_commitment(K, nonce) == commitment


def challenge_response_prove(K: NDArray[np.float64], challenge: bytes) -> bytes:
    """Prover: compute HMAC(K_nm, challenge) as proof of K_nm knowledge.

    The challenge is a random nonce from the verifier. The response
    proves the prover knows K_nm without transmitting it.

    Parameters
    ----------
    K
        Secret coupling matrix used as HMAC key material.
    challenge
        Verifier-issued challenge bytes.

    Returns
    -------
    bytes
        SHA-256 HMAC response.

    """
    return hmac.new(K.tobytes(), challenge, hashlib.sha256).digest()


def challenge_response_verify(
    K: NDArray[np.float64],
    challenge: bytes,
    response: bytes,
) -> bool:
    """Verify a response against the HMAC for a topology challenge.

    Parameters
    ----------
    K : NDArray[np.float64]
        Secret coupling matrix used as the HMAC key material.
    challenge : bytes
        Verifier-issued challenge bytes.
    response : bytes
        Claimed SHA-256 HMAC response.

    Returns
    -------
    bool
        ``True`` when the response matches the expected HMAC.

    """
    expected = hmac.new(K.tobytes(), challenge, hashlib.sha256).digest()
    return hmac.compare_digest(response, expected)


# --- Noise Tolerance ---


def fingerprint_noise_tolerance(
    K: NDArray[np.float64], n_trials: int = 100, sigma: float = 0.01
) -> dict[str, Any]:
    """Estimate fingerprint stability under small perturbations to K.

    Adds Gaussian noise N(0, sigma²) to K, recomputes fingerprint,
    measures drift. Returns mean and max drift across trials.

    Parameters
    ----------
    K
        Reference coupling matrix.
    n_trials
        Number of deterministic perturbation trials.
    sigma
        Standard deviation of the symmetric Gaussian perturbation.

    Returns
    -------
    dict
        Noise scale and mean, maximum, standard-deviation, and 99th-percentile
        fingerprint drift statistics.

    """
    fp_ref = spectral_fingerprint(K)
    rng = np.random.default_rng(42)
    drifts = []

    for _ in range(n_trials):
        noise = rng.normal(0, sigma, K.shape)
        noise = (noise + noise.T) / 2  # keep symmetric
        K_noisy = np.maximum(K + noise, 0)  # keep non-negative
        np.fill_diagonal(K_noisy, 0)  # keep zero diagonal
        fp_noisy = spectral_fingerprint(K_noisy)
        drifts.append(topology_distance(fp_ref, fp_noisy))

    return {
        "sigma": sigma,
        "mean_drift": float(np.mean(drifts)),
        "max_drift": float(np.max(drifts)),
        "std_drift": float(np.std(drifts)),
        "safe_tol": float(np.percentile(drifts, 99)),
    }


def row_hash_fingerprint(K: NDArray[np.float64]) -> list[bytes]:
    """Per-row SHA-256 hashes of K_nm.

    Enables selective verification: prove knowledge of specific coupling
    rows without revealing the full matrix. Useful for hierarchical
    authentication where different parties control different SCPN layers.

    Parameters
    ----------
    K
        Coupling matrix whose rows are committed independently.

    Returns
    -------
    list[bytes]
        SHA-256 digest for each matrix row in input order.

    """
    return [hashlib.sha256(K[i, :].tobytes()).digest() for i in range(K.shape[0])]


def verify_row_hash(K: NDArray[np.float64], row_idx: int, expected_hash: bytes) -> bool:
    """Verify one coupling-matrix row against its hash.

    Parameters
    ----------
    K
        Coupling matrix containing the candidate row.
    row_idx
        Row index to hash.
    expected_hash
        Claimed SHA-256 row digest.

    Returns
    -------
    bool
        Whether the row digest matches the claim.

    """
    return hashlib.sha256(K[row_idx, :].tobytes()).digest() == expected_hash
